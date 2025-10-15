"""Dataset factory helpers for Mosaic streaming dataloaders."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

try:
    from llmfoundry.data.text_data import StreamingTextDataset
    from streaming import StreamingDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc

if TYPE_CHECKING:
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.experiments.fl.configs.config import MosaicJobConfig

from .parallel import StatefulStreamingTextDataset
from .streams import StreamAssignment, StreamExtractionResult, _select_stream_subset


@dataclass(frozen=True)
class MosaicRuntimeConfig:
    """Runtime settings for the Mosaic dataloader workers."""

    num_workers: int
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    batch_size: int


@dataclass(frozen=True)
class NormalizedMosaicConfig:
    """Normalized configuration payload passed between dataloader helpers."""

    dataset_config: dict[str, Any]
    runtime: MosaicRuntimeConfig


@dataclass(frozen=True)
class DatasetFactoryConfig:
    """Keyword arguments used to instantiate the StatefulStreamingTextDataset."""

    kwargs: dict[str, Any]


def _select_dataset_config(dataset_cfg: Mapping[str, Any] | None, split: str) -> dict[str, Any]:
    if not dataset_cfg:
        return {}

    cfg = dict(dataset_cfg)
    if "common" in cfg or split in cfg:
        merged: dict[str, Any] = {}
        merged.update(cfg.pop("common", {}) or {})
        merged.update(cfg.pop(split, {}) or {})
        return merged
    return cfg


def _normalize_mosaic_dataloader_config(
    job_config: MosaicJobConfig,
    *,
    split: str,
    default_drop_last: bool,
) -> NormalizedMosaicConfig:
    """Normalize high-level Mosaic dataloader configuration into typed payloads."""

    mosaic_cfg = job_config.mosaic_dataloader
    if not mosaic_cfg:
        msg = "mosaic_dataloader config must be set."
        raise ValueError(msg)

    cfg = mosaic_cfg
    dataset_cfg = _select_dataset_config(cfg.dataset, split)

    num_workers = cfg.num_workers
    prefetch_factor = cfg.prefetch_factor
    pin_memory = cfg.pin_memory
    persistent_workers = cfg.persistent_workers
    drop_last = cfg.drop_last if cfg.drop_last is not None else default_drop_last

    batch_size = (
        job_config.validation.local_batch_size
        if split == "val"
        else job_config.training.local_batch_size
    )

    runtime = MosaicRuntimeConfig(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        batch_size=batch_size,
    )

    return NormalizedMosaicConfig(dataset_config=dataset_cfg, runtime=runtime)


def _prepare_dataset_kwargs(
    dataset_cfg: dict[str, Any],
    *,
    dataset_split_remote: str | None,
) -> DatasetFactoryConfig:
    """Prepare keyword arguments for the streaming dataset factory."""

    valid_params = {
        *inspect.signature(StreamingTextDataset).parameters,
        *inspect.signature(StreamingDataset).parameters,
    }
    dataset_kwargs = {k: v for k, v in dataset_cfg.items() if k in valid_params}

    subset_num_samples = dataset_cfg.get("subset_num_samples")
    if subset_num_samples is not None:
        dataset_kwargs["epoch_size"] = subset_num_samples

    if dataset_split_remote is not None:
        dataset_kwargs["split"] = dataset_split_remote

    return DatasetFactoryConfig(kwargs=dataset_kwargs)


def create_streaming_dataset(
    *,
    assignment: StreamAssignment,
    tokenizer: BaseTokenizer,
    dataset_config: DatasetFactoryConfig,
    batch_size: int,
    split: str,
) -> StatefulStreamingTextDataset:
    """Instantiate the stateful streaming dataset for the resolved stream subset."""

    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    return StatefulStreamingTextDataset(
        tokenizer=hf_tokenizer,
        streams=assignment.streams,
        batch_size=batch_size,
        **dataset_config.kwargs,
    )


def build_dataset_for_rank(
    normalized: NormalizedMosaicConfig,
    extraction: StreamExtractionResult,
    *,
    dp_rank: int,
    dp_world_size: int,
    tokenizer: BaseTokenizer,
    split: str,
) -> tuple[StatefulStreamingTextDataset, StreamAssignment]:
    """Create a dataset for the current rank from normalized configuration."""

    assignment = _select_stream_subset(
        extraction,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    dataset_factory_config = _prepare_dataset_kwargs(
        extraction.dataset_config,
        dataset_split_remote=assignment.dataset_split_remote,
    )
    dataset = create_streaming_dataset(
        assignment=assignment,
        tokenizer=tokenizer,
        dataset_config=dataset_factory_config,
        batch_size=normalized.runtime.batch_size,
        split=split,
    )
    return dataset, assignment


__all__ = [
    "DatasetFactoryConfig",
    "MosaicRuntimeConfig",
    "NormalizedMosaicConfig",
    "build_dataset_for_rank",
    "create_streaming_dataset",
    "_normalize_mosaic_dataloader_config",
]
