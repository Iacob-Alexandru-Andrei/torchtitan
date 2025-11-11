# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Construction utilities for Mosaic-aware dataloaders used in FL experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchtitan.experiments.fl.metrics import get_or_create_unigram_manager

from .dataset_factory import (
    _normalize_mosaic_dataloader_config,
    build_dataset_for_rank,
    MosaicRuntimeConfig,
    NormalizedMosaicConfig,
)
from .parallel import (
    MosaicParallelAwareDataloader,
    ParallelDataLoaderRequest,
    titan_collate_fn,
)
from .streams import _extract_streams
from .unigram import setup_unigram_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.experiments.fl.configs.config import MosaicJobConfig
    from torchtitan.experiments.fl.metrics import PureUnigramCrossEntropy
else:
    from collections import abc as _abc

    Callable = _abc.Callable


@dataclass(frozen=True)
class DataloaderBuildRequest:
    """Input parameters required to build a Mosaic dataloader.

    Args:
        job_config: Parsed Mosaic job configuration describing dataset settings.
        tokenizer: Tokenizer instance used to encode text samples.
        dp_world_size: Total number of data parallel ranks.
        dp_rank: Data parallel rank handled by the current process.
        split: Dataset split to load (for example ``"train"`` or ``"val"``).
        default_drop_last: Whether batches should drop the remainder by default.
    """

    job_config: MosaicJobConfig
    tokenizer: BaseTokenizer
    dp_world_size: int
    dp_rank: int
    split: str
    default_drop_last: bool


def _apply_split_overrides(
    normalized: NormalizedMosaicConfig, *, job_config: MosaicJobConfig, split: str
) -> NormalizedMosaicConfig:
    """Apply per-split runtime overrides from the job configuration.

    Args:
        normalized: Normalized dataloader configuration produced for the split.
        job_config: Full Mosaic job configuration with optional overrides.
        split: Dataset split for which overrides should be resolved.

    Returns:
        The normalized configuration updated with any matching overrides.
    """
    mosaic_cfg = job_config.mosaic_dataloader
    if not mosaic_cfg:
        return normalized

    overrides = mosaic_cfg.get_split_overrides(split)
    if not overrides:
        return normalized

    runtime = normalized.runtime
    updated_runtime = MosaicRuntimeConfig(
        num_workers=overrides.get("num_workers", runtime.num_workers),
        prefetch_factor=overrides.get("prefetch_factor", runtime.prefetch_factor),
        pin_memory=overrides.get("pin_memory", runtime.pin_memory),
        persistent_workers=overrides.get("persistent_workers", runtime.persistent_workers),
        drop_last=overrides.get("drop_last", runtime.drop_last),
        batch_size=runtime.batch_size,
    )
    return NormalizedMosaicConfig(
        dataset_config=normalized.dataset_config,
        runtime=updated_runtime,
        isolate_grouped_streams=normalized.isolate_grouped_streams,
    )


def _resolve_replica_identifier(job_config: MosaicJobConfig) -> str | None:
    """Resolve a stable identifier for the current replica if available."""
    run_uuid = getattr(job_config, "run_uuid", None) or os.getenv("RUN_UUID")
    run_uuid_str: str | None = None
    if run_uuid not in (None, ""):
        run_uuid_str = str(run_uuid)
        job_config.run_uuid = run_uuid_str

    candidate: int | str | None = getattr(job_config.fault_tolerance, "replica_id", None)
    if candidate in (None, "", -1):
        for env_var in (
            "TORCHFT_REPLICA_ID",
            "FAULT_TOLERANCE_REPLICA_ID",
            "FT_REPLICA_ID",
            "REPLICA_ID",
        ):
            env_value = os.getenv(env_var)
            if env_value not in (None, "", "-1"):
                candidate = env_value
                break
    replica_str: str | None = None
    if candidate not in (None, "", -1):
        replica_str = str(candidate)

    if run_uuid_str and replica_str:
        return f"{run_uuid_str}-rep{replica_str}"
    if run_uuid_str:
        return run_uuid_str
    return replica_str


def _build_mosaic_dataloader(
    request: DataloaderBuildRequest,
    *,
    register_unigram_metric: Callable[[PureUnigramCrossEntropy], None] | None = None,
) -> MosaicParallelAwareDataloader:
    """Construct a :class:`MosaicParallelAwareDataloader` for the request.

    Args:
        request: Fully-populated dataloader build request.
        register_unigram_metric: Optional callback used to register the unigram
            metric that powers the tokenizer-aware loss monitor.

    Returns:
        A dataloader configured for the requested split and data parallel rank.
    """
    normalized = _normalize_mosaic_dataloader_config(
        request.job_config,
        split=request.split,
        default_drop_last=request.default_drop_last,
    )
    normalized = _apply_split_overrides(normalized, job_config=request.job_config, split=request.split)

    extraction = _extract_streams(dict(normalized.dataset_config))
    replica_identifier = _resolve_replica_identifier(request.job_config)
    namespace_base = f"rep{replica_identifier}" if replica_identifier is not None else f"pid{os.getpid()}"
    shared_memory_namespace = f"{namespace_base}-{request.split}-dp{request.dp_rank}"
    dataset, assignment = build_dataset_for_rank(
        normalized,
        extraction,
        dp_rank=request.dp_rank,
        dp_world_size=request.dp_world_size,
        tokenizer=request.tokenizer,
        split=request.split,
        shared_memory_namespace=shared_memory_namespace,
    )

    unigram_manager = get_or_create_unigram_manager(request.job_config)
    unigram_setup = setup_unigram_metric(
        assignment,
        job_config=request.job_config,
        split=request.split,
        tokenizer=request.tokenizer,
        collate_fn=titan_collate_fn,
        manager=unigram_manager,
        register_unigram_metric=register_unigram_metric,
    )

    loader_request = ParallelDataLoaderRequest(
        dp_rank=request.dp_rank,
        dp_world_size=request.dp_world_size,
        runtime=normalized.runtime,
        collate_fn=unigram_setup.collate_fn,
        group_key=unigram_setup.group_key,
        unigram_handle=unigram_setup.handle,
    )
    return MosaicParallelAwareDataloader(dataset, loader_request)


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
    register_unigram_metric: Callable[[PureUnigramCrossEntropy], None] | None = None,
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the training split.

    Args:
        dp_world_size: Total number of data parallel ranks.
        dp_rank: Data parallel rank handled by the current process.
        tokenizer: Tokenizer used to encode text samples.
        job_config: Full Mosaic job configuration containing dataset options.
        register_unigram_metric: Optional callback used to register the unigram
            metric for monitoring loss skew.

    Returns:
        A :class:`MosaicParallelAwareDataloader` configured for training.
    """
    request = DataloaderBuildRequest(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="train",
        default_drop_last=True,
    )
    return _build_mosaic_dataloader(request, register_unigram_metric=register_unigram_metric)


def build_mosaic_validation_dataloader(  # noqa: PLR0913
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
    infinite: bool = False,  # noqa: ARG001 - kept for compatibility
    register_unigram_metric: Callable[[PureUnigramCrossEntropy], None] | None = None,
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the validation split.

    Args:
        dp_world_size: Total number of data parallel ranks.
        dp_rank: Data parallel rank handled by the current process.
        tokenizer: Tokenizer used to encode text samples.
        job_config: Full Mosaic job configuration containing dataset options.
        infinite: Historical parameter retained for compatibility; ignored.
        register_unigram_metric: Optional callback used to register the unigram
            metric for monitoring loss skew.

    Returns:
        A :class:`MosaicParallelAwareDataloader` configured for validation.
    """
    request = DataloaderBuildRequest(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="val",
        default_drop_last=False,
    )
    return _build_mosaic_dataloader(request, register_unigram_metric=register_unigram_metric)


__all__ = [
    "MosaicParallelAwareDataloader",
    "build_mosaic_dataloader",
    "build_mosaic_validation_dataloader",
    "titan_collate_fn",
]
