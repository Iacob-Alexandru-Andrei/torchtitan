# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helper utilities for consistent optimizer metric reductions.

These helpers attach replica counts to optimizer metrics and provide common
logic for preparing metrics before distributed reductions. They make it
possible to combine contributions from both sharded and fully replicated
parameter layouts without over- or under-counting.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor

METRIC_COUNT_PREFIX = "__tt_metric_count__"
METRIC_SQ_PREFIX = "__tt_metric_sq__"


def metric_count_key(metric_name: str) -> str:
    """Return the auxiliary key used to store the replica count for ``metric_name``."""
    return f"{METRIC_COUNT_PREFIX}{metric_name}"


def metric_sq_key(metric_name: str) -> str:
    """Return the auxiliary key used to store squared values for std calculation."""
    return f"{METRIC_SQ_PREFIX}{metric_name}"


def _infer_default_device(metrics: dict[str, Any]) -> torch.device:
    for value in metrics.values():
        if isinstance(value, DTensor):
            return value.device
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.tensor(0.0).device


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if isinstance(value, torch.Tensor):
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype=dtype)
        return value.to(device=device)
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(float(value), device=device, dtype=dtype)


def prepare_metrics_for_reduction(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Square local L2 metrics and register replica counts for later reductions."""
    device = _infer_default_device(optimizer_metrics)
    for metric_name, value in list(optimizer_metrics.items()):
        tensor_value = _as_tensor(value, device=device)
        optimizer_metrics[metric_name] = tensor_value

        count_key = metric_count_key(metric_name)
        if count_key not in optimizer_metrics:
            optimizer_metrics[count_key] = torch.ones(
                (),
                device=tensor_value.device,
                dtype=tensor_value.dtype,
            )
        else:
            optimizer_metrics[count_key] = _as_tensor(
                optimizer_metrics[count_key],
                device=tensor_value.device,
                dtype=tensor_value.dtype,
            )

        if metric_name.startswith("l2_norm"):
            optimizer_metrics[metric_name] = tensor_value.pow(2)

        # Track squared values for basis_similarity to compute std across replicas
        if metric_name == "mean/basis_similarity":
            sq_key = metric_sq_key(metric_name)
            # Skip NaN values from the std calculation
            if not torch.isnan(tensor_value):
                optimizer_metrics[sq_key] = tensor_value.pow(2)
            else:
                optimizer_metrics[sq_key] = torch.tensor(0.0, device=tensor_value.device, dtype=tensor_value.dtype)
                optimizer_metrics[count_key] = torch.tensor(0.0, device=tensor_value.device, dtype=tensor_value.dtype)

    return optimizer_metrics


def reduce_metrics_across_ranks(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce prepared metrics across all workers with c10d collectives."""
    world_size = c10d.get_world_size() if c10d.is_initialized() else 1

    local_keys: Iterable[str]
    if world_size > 1:
        local_keys = list(optimizer_metrics.keys())
        gathered_keys: list[list[str] | None] = [None for _ in range(world_size)]
        c10d.all_gather_object(gathered_keys, local_keys)
        keys: set[str] = set()
        for key_list in gathered_keys:
            if key_list is not None:
                keys.update(key_list)
        process_keys = sorted(
            k for k in keys
            if not k.startswith(METRIC_COUNT_PREFIX) and not k.startswith(METRIC_SQ_PREFIX)
        )
    else:
        process_keys = [
            key for key in optimizer_metrics.keys()
            if not key.startswith(METRIC_COUNT_PREFIX) and not key.startswith(METRIC_SQ_PREFIX)
        ]

    device = _infer_default_device(optimizer_metrics)

    for metric in process_keys:
        value = _as_tensor(optimizer_metrics.get(metric, 0.0), device=device)
        count_key = metric_count_key(metric)
        count = _as_tensor(
            optimizer_metrics.get(count_key, 0.0),
            device=value.device,
            dtype=value.dtype,
        )

        if world_size > 1:
            if metric.startswith("l2_norm"):
                c10d.all_reduce(value, op=c10d.ReduceOp.SUM)
                c10d.all_reduce(count, op=c10d.ReduceOp.SUM)
                denom = torch.clamp(count, min=1.0)
                optimizer_metrics[metric] = torch.sqrt(value / denom)
            elif metric.startswith("min"):
                c10d.all_reduce(value, op=c10d.ReduceOp.MIN)
                optimizer_metrics[metric] = value
            elif metric.startswith("max"):
                c10d.all_reduce(value, op=c10d.ReduceOp.MAX)
                optimizer_metrics[metric] = value
            elif metric.startswith("zero_count"):
                c10d.all_reduce(value, op=c10d.ReduceOp.SUM)
                optimizer_metrics[metric] = value
            else:
                c10d.all_reduce(value, op=c10d.ReduceOp.SUM)
                c10d.all_reduce(count, op=c10d.ReduceOp.SUM)
                denom = torch.clamp(count, min=1.0)
                mean_value = value / denom
                optimizer_metrics[metric] = mean_value

                # Compute std for basis_similarity metrics
                if metric == "mean/basis_similarity":
                    sq_key = metric_sq_key(metric)
                    sq_value = _as_tensor(optimizer_metrics.get(sq_key, 0.0), device=device)
                    c10d.all_reduce(sq_value, op=c10d.ReduceOp.SUM)
                    mean_sq = sq_value / denom
                    variance = torch.clamp(mean_sq - mean_value.pow(2), min=0.0)
                    std_key = "mean/basis_similarity_std"
                    optimizer_metrics[std_key] = torch.sqrt(variance)
        else:
            if metric.startswith("l2_norm"):
                optimizer_metrics[metric] = torch.sqrt(value)
            else:
                optimizer_metrics[metric] = value

                # For single-replica case, std is 0 for basis_similarity
                if metric == "mean/basis_similarity":
                    std_key = "mean/basis_similarity_std"
                    optimizer_metrics[std_key] = torch.tensor(0.0, device=value.device)

        optimizer_metrics.pop(count_key, None)

    # Remove any stray count keys that may remain (e.g., when no values were present).
    for key in list(optimizer_metrics.keys()):
        if key.startswith(METRIC_COUNT_PREFIX) or key.startswith(METRIC_SQ_PREFIX):
            optimizer_metrics.pop(key)

    return optimizer_metrics
