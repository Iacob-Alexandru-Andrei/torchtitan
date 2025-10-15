"""Dataset factory helpers for Mosaic streaming dataloaders."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
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
    from collections.abc import Mapping
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
    """Extract configuration for the requested split.

    Args:
        dataset_cfg: Mapping of dataset split names to configuration payloads.
        split: Name of the split being loaded.

    Returns:
        A dictionary containing the merged configuration for the split. When the
        configuration is missing, an empty dictionary is returned.
    """
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
    """Normalize high-level Mosaic dataloader configuration into typed payloads.

    Args:
        job_config: Full Mosaic job configuration describing dataloaders.
        split: Dataset split that is being prepared.
        default_drop_last: Fallback flag determining whether batches should drop
            incomplete final batches.

    Returns:
        The normalized dataset and runtime configuration for the requested split.
    """
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
    """Prepare keyword arguments for the streaming dataset factory.

    Args:
        dataset_cfg: Split-specific dataset configuration extracted from the job
            configuration.
        dataset_split_remote: Optional remote split name used by streaming
            backends.

    Returns:
        A :class:`DatasetFactoryConfig` containing sanitized keyword arguments.
    """
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
    """Instantiate the streaming dataset for the resolved stream subset.

    Args:
        assignment: Stream assignment describing the subset for this rank.
        tokenizer: Tokenizer used to encode the dataset samples.
        dataset_config: Normalized dataset configuration containing keyword
            arguments accepted by the streaming dataset.
        batch_size: Local batch size for the current rank.
        split: Dataset split name. Present for API consistency with future hooks.

    Returns:
        A :class:`StatefulStreamingTextDataset` configured for the rank's subset.
    """
    del split  # split is reserved for future streaming dataset hooks
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    return StatefulStreamingTextDataset(
        tokenizer=hf_tokenizer,
        streams=assignment.streams,
        batch_size=batch_size,
        **dataset_config.kwargs,
    )


def build_dataset_for_rank(  # noqa: PLR0913
    normalized: NormalizedMosaicConfig,
    extraction: StreamExtractionResult,
    *,
    dp_rank: int,
    dp_world_size: int,
    tokenizer: BaseTokenizer,
    split: str,
) -> tuple[StatefulStreamingTextDataset, StreamAssignment]:
    """Create a dataset for the current rank from normalized configuration.

    Args:
        normalized: Normalized Mosaic configuration for the job and split.
        extraction: Stream extraction results derived from the dataset config.
        dp_rank: Data parallel rank handled by the current process.
        dp_world_size: Total number of data parallel ranks.
        tokenizer: Tokenizer used to encode dataset samples.
        split: Dataset split name.

    Returns:
        Tuple containing the instantiated dataset and the resolved stream
        assignment.
    """
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
    "_normalize_mosaic_dataloader_config",
    "build_dataset_for_rank",
    "create_streaming_dataset",
]
