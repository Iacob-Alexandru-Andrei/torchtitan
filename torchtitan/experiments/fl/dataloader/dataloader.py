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
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

try:
    from llmfoundry.data.text_data import StreamingTextDataset
    from streaming import Stream, StreamingDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    missing_dependency_msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(missing_dependency_msg) from exc

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.components.tokenizer import BaseTokenizer
    from torchtitan.experiments.fl.configs.config import MosaicJobConfig


class StatefulStreamingTextDataset(StreamingTextDataset):
    """Track how many samples the streaming dataset has yielded.

    This makes the dataset compatible with dataloaders like TorchTitan's
    ``StatefulDataLoader`` that do not pass arguments to the dataset's
    ``state_dict`` method.

    Args:
        *args: Positional arguments to pass to ``StreamingTextDataset``.
        **kwargs: Keyword arguments to pass to ``StreamingTextDataset``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_samples_yielded = 0

    def __getitem__(self, idx: int) -> dict[str, list[int]] | torch.Tensor:
        """Increment the internal sample counter each time an item is fetched."""
        self._num_samples_yielded += 1
        return super().__getitem__(idx)

    def state_dict(
        self,
        num_samples: int | None = None,
        *,
        from_beginning: bool = True,
    ) -> dict[str, Any]:
        """Saves the dataset's state.

        If `num_samples` is not provided by the caller, it defaults to using
        the internal `_num_samples_yielded` counter. This makes it compatible
        with the StatefulDataLoader.
        """
        effective_num_samples = (
            num_samples if num_samples is not None else self._num_samples_yielded
        )
        parent_state = super().state_dict(
            num_samples=effective_num_samples, from_beginning=from_beginning
        )
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Restore the dataset's state from a state dictionary."""
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

    def __init__(  # noqa: PLR0913
        self,
        dataset: StatefulStreamingTextDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Callable | None = None,
        num_workers: int = 0,
        *,
        prefetch_factor: int | None = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
    ) -> None:
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
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        value = state_dict[self._rank_id]
        if not isinstance(value, bytes):
            logger.warning(
                f"Expected bytes for DataLoader state at key {self._rank_id}, got {type(value)}. Aborting load_state_dict."
            )
            return
        super().load_state_dict(pickle.loads(value))  # noqa: S301


def titan_collate_fn(batch: list[Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate samples from ``StreamingTextDataset`` for the TorchTitan loop.

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
        batch_type_msg = f"Unsupported batch item type from dataset: {type(batch[0])}"
        raise TypeError(batch_type_msg)

    model_inputs = input_ids_tensor[:, :-1].contiguous()
    labels = input_ids_tensor[:, 1:].contiguous()
    input_dict = {"input": model_inputs}
    return input_dict, labels


def _get_stream_arg_names() -> set[str]:
    """Return the valid keyword arguments accepted by :class:`~streaming.Stream`."""
    if not hasattr(_get_stream_arg_names, "_cache"):
        stream_params = inspect.signature(Stream.__init__).parameters
        _get_stream_arg_names._cache = {
            name for name in stream_params if name != "self"
        }
    return _get_stream_arg_names._cache  # type: ignore[attr-defined]


def _build_streams_from_config(
    streams_cfg: Mapping[str, Any] | None,
    *,
    base_streams: Mapping[str, Any] | None = None,
) -> list[Stream] | None:
    """Construct :class:`Stream` objects from configuration dictionaries.

    Args:
        streams_cfg: Mapping of stream names to their configuration dictionaries.
        base_streams: Optional mapping of base stream definitions that can be
            referenced by name. When provided, the configuration for a stream is
            merged on top of the corresponding base configuration sharing the
            same key.

    Returns:
        A list of instantiated :class:`Stream` objects or ``None`` if
        ``streams_cfg`` is ``None``.
    """
    if not streams_cfg:
        return None

    valid_keys = _get_stream_arg_names()
    resolved_streams: list[Stream] = []
    for stream_name, stream_config in streams_cfg.items():
        if not isinstance(stream_config, Mapping):
            type_error_msg = (
                "Stream configuration entries must be mappings. "
                f"Got {type(stream_config)!r} for stream '{stream_name}'."
            )
            raise TypeError(type_error_msg)

        merged_config: dict[str, Any] = {}
        if base_streams and stream_name in base_streams:
            merged_config.update(deepcopy(base_streams[stream_name]))
        merged_config.update(deepcopy(stream_config))

        # Filter out any unsupported keys before instantiating the Stream.
        stream_kwargs = {
            key: merged_config[key] for key in valid_keys if key in merged_config
        }
        resolved_streams.append(Stream(**stream_kwargs))

    return resolved_streams


def _select_non_iid_streams_from_mapping(
    config: Mapping[str, Any], dp_rank: int
) -> Mapping[str, Any] | None:
    """Resolve the non-IID configuration from a mapping structure."""
    candidate_keys = (
        str(dp_rank),
        f"dp_rank_{dp_rank}",
        f"rank_{dp_rank}",
        f"worker_{dp_rank}",
    )
    for key in candidate_keys:
        if key in config:
            return config[key]
    return config.get("default")


def _select_non_iid_streams_from_sequence(
    config: Sequence[Mapping[str, Any]], dp_rank: int
) -> Mapping[str, Any] | None:
    """Resolve the non-IID configuration from a sequence of mappings."""
    default_entry: Mapping[str, Any] | None = None
    for entry in config:
        if not isinstance(entry, Mapping):
            continue

        ranks = (
            entry.get("dp_ranks")
            or entry.get("ranks")
            or entry.get("workers")
            or entry.get("clients")
        )
        if ranks is None:
            if entry.get("default"):
                default_entry = entry
            continue

        if isinstance(ranks, int):
            ranks = [ranks]

        if dp_rank in ranks:
            return entry.get("streams") or entry.get("client_streams")

    if default_entry is not None:
        return default_entry.get("streams") or default_entry.get("client_streams")
    return None


def _select_non_iid_streams_config(
    non_iid_cfg: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    dp_rank: int,
) -> Mapping[str, Any] | None:
    """Pick the appropriate stream configuration for a given data-parallel rank."""
    if non_iid_cfg is None:
        return None

    if isinstance(non_iid_cfg, Mapping):
        return _select_non_iid_streams_from_mapping(non_iid_cfg, dp_rank)

    if isinstance(non_iid_cfg, Sequence):
        return _select_non_iid_streams_from_sequence(non_iid_cfg, dp_rank)

    return None


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: MosaicJobConfig,
) -> MosaicParallelAwareDataloader:
    """Build a dataloader using Mosaic's StreamingTextDataset.

    This function constructs a StreamingTextDataset and wraps it in a
    MosaicParallelAwareDataloader that supports prefetching and distributed
    data parallelism.
    """
    mosaic_cfg = job_config.mosaic_dataloader
    if mosaic_cfg is None:
        missing_cfg_msg = "mosaic_dataloader config must be set."
        raise ValueError(missing_cfg_msg)

    # The tokenizer is expected to be a HuggingFace tokenizer or a wrapper.
    # We attempt to extract it here.
    hf_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    cfg = deepcopy(mosaic_cfg)
    dataset_cfg = cfg.get("dataset", {})

    # Extract dataloader-specific config
    num_workers = cfg.get("num_workers", 8)
    prefetch_factor = cfg.get("prefetch_factor", 2)
    pin_memory = cfg.get("pin_memory", True)
    persistent_workers = cfg.get("persistent_workers", True)
    drop_last = cfg.get("drop_last", True)

    # Build streams
    streams_cfg = dataset_cfg.pop("streams", None)
    base_streams = (
        {name: deepcopy(config) for name, config in streams_cfg.items()}
        if isinstance(streams_cfg, Mapping)
        else {}
    )
    non_iid_streams_cfg = dataset_cfg.pop("non_iid_streams", None)

    selected_non_iid_cfg = _select_non_iid_streams_config(
        non_iid_streams_cfg, dp_rank=dp_rank
    )
    streams = None

    if selected_non_iid_cfg is not None:
        streams = _build_streams_from_config(
            selected_non_iid_cfg, base_streams=base_streams
        )
        if streams is None:
            empty_stream_msg = (
                "non_iid_streams configuration for rank "
                f"{dp_rank} must define at least one stream."
            )
            raise ValueError(empty_stream_msg)
        logger.info(
            "Using non-IID stream configuration for dp_rank %s with %s streams",
            dp_rank,
            len(streams),
        )
    elif isinstance(streams_cfg, Mapping):
        streams = _build_streams_from_config(streams_cfg)
        if streams:
            logger.info(f"Built {len(streams)} streams for Mosaic dataloader")

    # Filter dataset config to only include valid StreamingTextDataset parameters
    valid_params = {
        *inspect.signature(StreamingTextDataset).parameters,
        *inspect.signature(StreamingDataset).parameters,
    }
    dataset_config_filtered = {
        k: v for k, v in dataset_cfg.items() if k in valid_params
    }

    # Build the StreamingTextDataset
    logger.info(
        "Building StreamingTextDataset with config: %s", dataset_config_filtered
    )
    text_dataset = StatefulStreamingTextDataset(
        tokenizer=hf_tokenizer,
        streams=streams,
        batch_size=job_config.training.local_batch_size,
        **dataset_config_filtered,
    )

    return MosaicParallelAwareDataloader(
        dataset=text_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        collate_fn=titan_collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


__all__ = ["MosaicParallelAwareDataloader", "build_mosaic_dataloader"]
