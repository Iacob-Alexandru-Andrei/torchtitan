"""Optimizer monitoring utilities for Mosaic/TorchTitan examples.

These utilities are mirrored from the private Photon project so that users can
experiment with Mosaic streaming jobs without modifying the core TorchTitan
library.  They register a Composer callback that logs gradient norms, optimizer
statistics, and optional curvature metrics after each optimizer step.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.utils import dist
from llmfoundry.registry import callbacks

__all__ = [
    "OptimizerMonitor",
    "accumulate_curvature_metrics",
    "finalize_curvature_metrics",
]


@callbacks.register_class("optimizer_monitor")
class OptimizerMonitor(Callback):
    """Monitor gradients, optimizer state, and curvature metrics."""

    def __init__(
        self,
        interval: int | str | Time = "10ba",
        *,
        only_global: bool = False,
        log_optimizer_metrics: bool = True,
        report_curvature: bool = False,
    ) -> None:
        self.log_optimizer_metrics = log_optimizer_metrics
        self.only_global = only_global

        if isinstance(interval, int):
            self.interval = Time(interval, TimeUnit.BATCH)
        elif isinstance(interval, str):
            self.interval = Time.from_timestring(interval)
        else:
            self.interval = interval

        if (
            self.interval.unit == TimeUnit.BATCH
            and self.interval < Time.from_timestring("10ba")
        ):
            warnings.warn(
                "OptimizerMonitor interval %s is below the recommended 10ba and may"
                " reduce throughput." % self.interval,
                stacklevel=2,
            )

        if self.interval.unit not in {TimeUnit.BATCH, TimeUnit.EPOCH}:
            msg = f"Invalid time unit for parameter interval: {self.interval.unit}"
            raise ValueError(msg)
        self.report_curvature = report_curvature

    def batch_end(self, state: State, logger: Logger) -> None:  # noqa: C901, PLR0915, PLR0912
        current_time_value = state.timestamp.get(self.interval.unit).value
        if current_time_value % self.interval.value != 0:
            return

        optimizer_metrics: dict[str, Any] = {}

        for name, param in state.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            metric_reporter: Callable[[Any, Any, Any], dict[str, Any]] | None = getattr(
                state.optimizers[0],
                "report_per_parameter_metrics",
                None,
            )
            if callable(metric_reporter) and self.log_optimizer_metrics:
                optimizer_metrics.update(metric_reporter(param, name, optimizer_metrics))

            if f"l2_norm/grad/{name}" not in optimizer_metrics:
                optimizer_metrics[f"l2_norm/grad/{name}"] = torch.linalg.vector_norm(
                    param.grad
                )

        if state.fsdp_enabled and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            pre_reduce_metrics: Callable[[Any], dict[str, Any]] | None = getattr(
                state.optimizers[0],
                "pre_reduce_metrics",
                None,
            )
            if callable(pre_reduce_metrics):
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            dist_reduce_metrics: Callable[[Any], dict[str, Any]] | None = getattr(
                state.optimizers[0],
                "dist_reduce_metrics",
                None,
            )
            if callable(dist_reduce_metrics):
                optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        agg_types = ["l2_norm", "min", "max"]
        agg_dict: dict[tuple[str, str], Any] = {}
        for metric in optimizer_metrics:
            parts = metric.split("/")
            if len(parts) < 2:
                continue
            agg_type, metric_name = parts[0], parts[1]
            if agg_type not in agg_types:
                continue
            key = (agg_type, metric_name)
            if agg_type == "l2_norm" and key not in agg_dict:
                agg_dict[key] = 0.0
            elif agg_type == "min" and key not in agg_dict:
                agg_dict[key] = float("inf")
            elif agg_type == "max" and key not in agg_dict:
                agg_dict[key] = float("-inf")

        for metric, value in optimizer_metrics.items():
            parts = metric.split("/")
            if len(parts) < 2:
                continue
            agg_type, metric_name = parts[0], parts[1]
            if agg_type not in agg_types:
                continue
            key = (agg_type, metric_name)
            if agg_type == "l2_norm":
                agg_dict[key] += value**2
            elif agg_type == "min":
                agg_dict[key] = min(agg_dict[key], value)
            elif agg_type == "max":
                agg_dict[key] = max(agg_dict[key], value)

        for (agg_type, metric_name), agg_value in agg_dict.items():
            if agg_type == "l2_norm":
                optimizer_metrics[f"{agg_type}/{metric_name}/global"] = agg_value**0.5
            else:
                optimizer_metrics[f"{agg_type}/{metric_name}/global"] = agg_value

        if self.only_global:
            optimizer_metrics = {
                k: v for k, v in optimizer_metrics.items() if k.endswith("/global")
            }

        if self.report_curvature:
            curvature_acc = accumulate_curvature_metrics(optimizer_metrics)
            curvature_stats = finalize_curvature_metrics(curvature_acc)
            optimizer_metrics.update(curvature_stats)

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()
        logger.log_metrics(optimizer_metrics)


def accumulate_curvature_metrics(
    optimizer_metrics: dict[str, torch.Tensor],
    metric_name_prefixes: tuple[str, ...] = (
        "curvature/param_diff_norm",
        "curvature/grad_diff_norm",
        "curvature/long_bb",
        "curvature/short_bb",
        "curvature/l2_norm/second_to_first_derivative_estimate_ratio",
        "curvature/l2_norm/second_derivative_estimate",
        "curvature/local_lipschitz",
    ),
) -> dict[str, Any]:
    sums_for_norms = {
        "curvature/param_diff_norm": 0.0,
        "curvature/grad_diff_norm": 0.0,
    }
    list_for_stats: dict[str, list[float]] = {
        "curvature/long_bb": [],
        "curvature/short_bb": [],
        "curvature/l2_norm/first_to_second_derivative_estimate_ratio": [],
        "curvature/l2_norm/second_derivative_estimate": [],
        "curvature/local_lipschitz": [],
    }

    for metric_name, metric_value in optimizer_metrics.items():
        for prefix in metric_name_prefixes:
            if not metric_name.startswith(prefix):
                continue
            val = (
                metric_value.item()
                if isinstance(metric_value, torch.Tensor)
                else metric_value
            )
            if prefix in sums_for_norms:
                sums_for_norms[prefix] += val**2
            elif prefix in list_for_stats:
                list_for_stats[prefix].append(val)

    return {
        "sums_for_norms": sums_for_norms,
        "list_for_stats": list_for_stats,
    }


def finalize_curvature_metrics(acc: dict[str, Any]) -> dict[str, float]:
    final_metrics: dict[str, float] = {}
    sums_for_norms = acc["sums_for_norms"]
    for prefix, squared_sum in sums_for_norms.items():
        final_metrics[f"{prefix}/global"] = squared_sum**0.5

    grad_diff_g = final_metrics.get("curvature/grad_diff_norm/global", 0.0)
    param_diff_g = final_metrics.get("curvature/param_diff_norm/global", 1.0)
    final_metrics["curvature/local_lipschitz/global"] = (
        grad_diff_g / param_diff_g if param_diff_g != 0.0 else float("inf")
    )

    list_for_stats = acc["list_for_stats"]
    for prefix, values in list_for_stats.items():
        if not values:
            continue
        tensor_vals = torch.tensor(values, dtype=torch.float32)
        finite_vals = tensor_vals[torch.isfinite(tensor_vals)]
        if finite_vals.numel() == 0:
            continue
        final_metrics[f"{prefix}/mean"] = finite_vals.mean().item()
        final_metrics[f"{prefix}/median"] = finite_vals.median().item()
        final_metrics[f"{prefix}/min"] = finite_vals.min().item()
        final_metrics[f"{prefix}/max"] = finite_vals.max().item()

    return final_metrics
