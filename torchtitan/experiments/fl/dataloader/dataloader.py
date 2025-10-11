# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this- source tree.

"""Adapters for using Mosaic streaming dataloaders with TorchTitan."""

from __future__ import annotations

import inspect
import pickle
import os
import posixpath
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

try:
    from llmfoundry.data.text_data import StreamingTextDataset
    from streaming import Stream, StreamingDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    ) from exc

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.experiments.fl.configs.config import MosaicJobConfig
from torchtitan.tools.logging import logger


def _is_uri(path: str | None) -> bool:
    return bool(path and "://" in path)


def _join_remote_path(root: str | None, path: str | None) -> str | None:
    if path is None or _is_uri(path):
        return path
    if root is None:
        return path
    if _is_uri(root):
        return f"{root.rstrip('/')}/{path.lstrip('/')}"
    return posixpath.join(root, path)


def _join_local_path(root: str | None, path: str | None) -> str | None:
    if path is None or root is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _flatten_stream_configs(streams_cfg: Any) -> dict[str, dict[str, Any]]:
    """Flatten nested stream configurations into a simple mapping."""

    flattened: dict[str, dict[str, Any]] = {}

    def _collect(config: Any, parent_key: str | None = None) -> None:
        if isinstance(config, Mapping):
            if "remote" in config or "local" in config:
                flattened_key = config.get("name") or parent_key or f"stream_{len(flattened)}"
                flattened[flattened_key] = dict(config)
                flattened[flattened_key].setdefault("name", flattened_key)
                return

            for key, value in config.items():
                if key == "client_streams":
                    _collect(value)
                elif isinstance(value, Mapping) and ("remote" in value or "local" in value):
                    flattened[key] = dict(value)
                    flattened[key].setdefault("name", key)
                else:
                    _collect(value, key)
        elif isinstance(config, (list, tuple)):
            for item in config:
                _collect(item)

    _collect(streams_cfg)
    return flattened


def _select_dataset_config(dataset_cfg: Mapping[str, Any] | None, split: str) -> dict[str, Any]:
    if not dataset_cfg:
        return {}

    cfg = deepcopy(dict(dataset_cfg))
    if "common" in cfg or split in cfg:
        merged: dict[str, Any] = {}
        merged.update(cfg.pop("common", {}) or {})
        merged.update(cfg.pop(split, {}) or {})
        return merged
    return cfg


def _extract_streams(dataset_cfg: dict[str, Any]) -> tuple[list[Stream] | None, dict[str, Any]]:
    root_remote = dataset_cfg.pop("root_remote", None)
    root_local = dataset_cfg.pop("root_local", None)
    streams_cfg = dataset_cfg.pop("streams", None)

    if not streams_cfg:
        return None, dataset_cfg

    flattened = _flatten_stream_configs(streams_cfg)
    streams: list[Stream] = []
    for name, stream_cfg in flattened.items():
        stream_kwargs = dict(stream_cfg)
        stream_kwargs = {key: value for key, value in stream_kwargs.items() if value is not None}
        stream_kwargs.pop("name", None)

        if "remote" in stream_kwargs:
            stream_kwargs["remote"] = _join_remote_path(root_remote, stream_kwargs["remote"])
        elif root_remote is not None:
            logger.warning(
                "Stream %s is missing a remote path; root_remote was provided.",
                name,
            )

        if "local" in stream_kwargs:
            stream_kwargs["local"] = _join_local_path(root_local, stream_kwargs["local"])
        elif root_local is not None:
            logger.warning("Stream %s is missing a local path; root_local was provided.", name)

        streams.append(Stream(**stream_kwargs))

    logger.info("Built %d streams for Mosaic dataloader", len(streams))
    return streams, dataset_cfg


class StatefulStreamingTextDataset(StreamingTextDataset):
    """
    A stateful wrapper around StreamingTextDataset that internally tracks the number
    of samples yielded. This makes it compatible with dataloaders like TorchTitan's
    StatefulDataLoader that do not pass arguments to the dataset's state_dict method.

    Args:
        *args: Positional arguments to pass to StreamingTextDataset.
        **kwargs: Keyword arguments to pass to StreamingTextDataset.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._num_samples_yielded = 0

    def __getitem__(self, idx: int) -> dict[str, list[int]] | torch.Tensor:
        """
        Overrides the parent method to increment the internal sample counter
        each time an item is fetched.
        """
        self._num_samples_yielded += 1
        return super().__getitem__(idx)

    def state_dict(self, num_samples: int | None = None, from_beginning: bool = True) -> dict[str, Any]:
        """
        Saves the dataset's state.

        If `num_samples` is not provided by the caller, it defaults to using
        the internal `_num_samples_yielded` counter. This makes it compatible
        with the StatefulDataLoader.
        """
        effective_num_samples = num_samples if num_samples is not None else self._num_samples_yielded
        parent_state = super().state_dict(num_samples=effective_num_samples, from_beginning=from_beginning)
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """
        Restores the dataset's state from a state dictionary.
        """
        self._num_samples_yielded = obj.pop("_num_samples_yielded", 0)
        super().load_state_dict(obj)


class MosaicParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader for Mosaic StreamingTextDataset with distributed data parallelism support.

    This dataloader inherits from torchdata's StatefulDataLoader to provide
    full checkpointing support with prefetching capabilities for streaming datasets,
    following the same pattern as the standard ParallelAwareDataloader.

    Args:
        dataset: The StreamingTextDataset instance to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
        collate_fn: Optional function to collate samples in a batch.
        num_workers: Number of worker processes for data loading.
        prefetch_factor: Number of batches to prefetch per worker.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Whether to keep workers alive between epochs.
        drop_last: Whether to drop the last incomplete batch.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: StatefulStreamingTextDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Callable | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        )
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        """Save dataloader state for checkpointing."""
        return {
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state from checkpoint."""
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"DataLoader state is empty for dp rank {self.dp_rank}, expected key {self._rank_id}")
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
        )
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def titan_collate_fn(batch: list[Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Collates samples from StreamingTextDataset and formats them for the
    TorchTitan training loop.

    Args:
        batch: A list of samples from the dataset.

    Returns:
        A tuple where the first element is an `input_dict` and the second is a
        tensor of corresponding labels.
    """
    if isinstance(batch[0], dict):
        input_ids_list = [item["input_ids"] for item in batch]
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        else:
            input_ids_tensor = torch.stack(input_ids_list)
    elif isinstance(batch[0], torch.Tensor):
        input_ids_tensor = torch.stack(batch)
    else:
        raise TypeError(f"Unsupported batch item type from dataset: {type(batch[0])}")

    model_inputs = input_ids_tensor[:, :-1].contiguous()
    labels = input_ids_tensor[:, 1:].contiguous()
    input_dict = {"input": model_inputs}
    return input_dict, labels


def _build_mosaic_dataloader(
    *,
    job_config: MosaicJobConfig,
    tokenizer: BaseTokenizer,
    dp_world_size: int,
    dp_rank: int,
    split: str,
    default_drop_last: bool,
) -> MosaicParallelAwareDataloader:
    mosaic_cfg = job_config.mosaic_dataloader
    if not mosaic_cfg:
        raise ValueError("mosaic_dataloader config must be set.")

    cfg = deepcopy(mosaic_cfg)
    dataset_cfg_raw = cfg.pop("dataset", {})
    dataset_cfg = _select_dataset_config(dataset_cfg_raw, split)

    # Extract dataloader-specific config
    num_workers = cfg.get("num_workers", 8)
    prefetch_factor = cfg.get("prefetch_factor", 2)
    pin_memory = cfg.get("pin_memory", True)
    persistent_workers = cfg.get("persistent_workers", True)
    drop_last = cfg.get("drop_last", default_drop_last)

    # Allow per-split overrides
    split_overrides = cfg.get(split, {})
    if isinstance(split_overrides, Mapping):
        num_workers = split_overrides.get("num_workers", num_workers)
        prefetch_factor = split_overrides.get("prefetch_factor", prefetch_factor)
        pin_memory = split_overrides.get("pin_memory", pin_memory)
        persistent_workers = split_overrides.get("persistent_workers", persistent_workers)
        drop_last = split_overrides.get("drop_last", drop_last)

    streams, dataset_cfg = _extract_streams(dataset_cfg)

    # Filter dataset config to only include valid StreamingTextDataset parameters
    valid_params = {
        *inspect.signature(StreamingTextDataset).parameters,
        *inspect.signature(StreamingDataset).parameters,
    }
    dataset_config_filtered = {k: v for k, v in dataset_cfg.items() if k in valid_params}

    # Resolve optional subset configuration
    subset_num_samples = dataset_cfg.pop("subset_num_samples", None)
    if subset_num_samples is not None:
        dataset_config_filtered["epoch_size"] = subset_num_samples

    # The tokenizer is expected to be a HuggingFace tokenizer or a wrapper.
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    batch_size = job_config.validation.local_batch_size if split == "val" else job_config.training.local_batch_size

    logger.info(
        "Building StreamingTextDataset (%s split) with config: %s",
        split,
        dataset_config_filtered,
    )
    text_dataset = StatefulStreamingTextDataset(
        tokenizer=hf_tokenizer,
        streams=streams,
        batch_size=batch_size,
        **dataset_config_filtered,
    )

    return MosaicParallelAwareDataloader(
        dataset=text_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=titan_collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
) -> MosaicParallelAwareDataloader:
    """Build a Mosaic dataloader for the training split."""

    return _build_mosaic_dataloader(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="train",
        default_drop_last=True,
    )


def build_mosaic_validation_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
    infinite: bool = False,  # noqa: ARG001 - kept for compatibility
) -> MosaicParallelAwareDataloader:
    """
    Build a Mosaic dataloader for the validation split.

    Parameters
    ----------
    dp_world_size : int
        Data parallel world size.
    dp_rank : int
        Data parallel rank.
    tokenizer : BaseTokenizer
        Tokenizer instance.
    job_config : MosaicJobConfig
        Job configuration.
    infinite : bool, optional
        Unused parameter kept for compatibility with previous versions of the API.
        It may be removed in a future release; downstream callers should not rely
        on it.
    """

    return _build_mosaic_dataloader(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        split="val",
        default_drop_last=False,
    )


__all__ = [
    "MosaicParallelAwareDataloader",
    "build_mosaic_dataloader",
    "build_mosaic_validation_dataloader",
]
