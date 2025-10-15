"""Public entry points for constructing Mosaic streaming dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from torchtitan.experiments.fl.metrics import get_or_create_unigram_manager

from .dataset_factory import (
    MosaicRuntimeConfig,
    NormalizedMosaicConfig,
    build_dataset_for_rank,
    _normalize_mosaic_dataloader_config,
)
from .parallel import MosaicParallelAwareDataloader, ParallelDataLoaderRequest, titan_collate_fn
from .streams import _extract_streams
from .unigram import setup_unigram_metric

if TYPE_CHECKING:
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.experiments.fl.configs.config import MosaicJobConfig
    from torchtitan.experiments.fl.metrics import PureUnigramCrossEntropy


@dataclass(frozen=True)
class DataloaderBuildRequest:
    """Input parameters required to build a Mosaic dataloader."""

    job_config: MosaicJobConfig
    tokenizer: BaseTokenizer
    dp_world_size: int
    dp_rank: int
    split: str
    default_drop_last: bool


def _apply_split_overrides(
    normalized: NormalizedMosaicConfig, *, job_config: MosaicJobConfig, split: str
) -> NormalizedMosaicConfig:
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
    return NormalizedMosaicConfig(dataset_config=normalized.dataset_config, runtime=updated_runtime)


def _build_mosaic_dataloader(
    request: DataloaderBuildRequest,
    *,
    register_unigram_metric: Callable[["PureUnigramCrossEntropy"], None] | None = None,
) -> MosaicParallelAwareDataloader:
    normalized = _normalize_mosaic_dataloader_config(
        request.job_config,
        split=request.split,
        default_drop_last=request.default_drop_last,
    )
    normalized = _apply_split_overrides(normalized, job_config=request.job_config, split=request.split)

    extraction = _extract_streams(dict(normalized.dataset_config))
    dataset, assignment = build_dataset_for_rank(
        normalized,
        extraction,
        dp_rank=request.dp_rank,
        dp_world_size=request.dp_world_size,
        tokenizer=request.tokenizer,
        split=request.split,
    )

    unigram_manager = get_or_create_unigram_manager(request.job_config)
    unigram_setup = setup_unigram_metric(
        assignment,
        job_config=request.job_config,
        split=request.split,
        tokenizer=request.tokenizer,
        collate_fn=titan_collate_fn,
        manager=unigram_manager,
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
    register_unigram_metric: Callable[["PureUnigramCrossEntropy"], None] | None = None,
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the training split."""

    request = DataloaderBuildRequest(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="train",
        default_drop_last=True,
    )
    return _build_mosaic_dataloader(request, register_unigram_metric=register_unigram_metric)


def build_mosaic_validation_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
    infinite: bool = False,  # noqa: ARG001 - kept for compatibility
    register_unigram_metric: Callable[["PureUnigramCrossEntropy"], None] | None = None,
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the validation split."""

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
