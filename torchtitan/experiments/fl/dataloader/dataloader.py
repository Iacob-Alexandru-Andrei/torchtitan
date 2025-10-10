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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

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


def _coerce_rank(value: Any) -> int:
    """Return ``value`` as an integer data parallel rank."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Invalid dp_rank value {value!r} in non_iid configuration"
        ) from exc


def _collect_candidate_ranks(entry: Mapping[str, Any], fallback_index: int) -> set[int]:
    """Extract all candidate data parallel ranks from a non-IID sequence entry."""
    candidate_ranks: set[int] = set()
    explicit_rank = False

    if "dp_rank" in entry:
        candidate_ranks.add(_coerce_rank(entry["dp_rank"]))
        explicit_rank = True

    if "dp_ranks" in entry:
        ranks_value = entry["dp_ranks"]
        if isinstance(ranks_value, Sequence) and not isinstance(
            ranks_value, (str, bytes)
        ):
            candidate_ranks.update(_coerce_rank(rank) for rank in ranks_value)
        else:
            candidate_ranks.add(_coerce_rank(ranks_value))
        explicit_rank = True

    if not explicit_rank:
        candidate_ranks.add(fallback_index)

    return candidate_ranks


def _select_non_iid_streams_from_sequence(
    sequence: Sequence[Mapping[str, Any]], dp_rank: int
) -> dict[str, Any] | None:
    """Select streams for ``dp_rank`` from a sequential non-IID definition."""
    for index, entry in enumerate(sequence):
        if not isinstance(entry, Mapping):  # pragma: no cover - defensive
            logger.warning(
                "Skipping non_iid entry at index %s because it is not a mapping: %r",
                index,
                entry,
            )
            continue

        candidate_ranks = _collect_candidate_ranks(entry, index)
        if dp_rank in candidate_ranks:
            return {
                "streams": entry.get("streams"),
                "client_streams": entry.get("client_streams"),
            }

    return None


def _select_non_iid_streams(
    non_iid_cfg: Mapping[str, Any] | None, *, dp_rank: int
) -> dict[str, Any] | None:
    """Select non-IID stream definitions for the provided ``dp_rank``."""
    if not non_iid_cfg:
        return None

    by_dp_rank = non_iid_cfg.get("by_dp_rank")
    if isinstance(by_dp_rank, Mapping):
        entry: Any | None = None
        if str(dp_rank) in by_dp_rank:
            entry = by_dp_rank[str(dp_rank)]
        elif dp_rank in by_dp_rank:
            entry = by_dp_rank[dp_rank]
        elif "default" in by_dp_rank:
            entry = by_dp_rank["default"]

        if entry is not None:
            if not isinstance(entry, Mapping):  # pragma: no cover - defensive
                raise TypeError(
                    "non_iid.by_dp_rank entries must be mappings containing "
                    "streams/client_streams definitions"
                )
            return {
                "streams": entry.get("streams"),
                "client_streams": entry.get("client_streams"),
            }

    sequence = non_iid_cfg.get("sequence")
    if sequence is None:
        return None
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
        raise TypeError(
            "non_iid.sequence must be a sequence of mappings containing "
            "streams/client_streams definitions"
        )

    return _select_non_iid_streams_from_sequence(sequence, dp_rank)


def _build_stream_objects(
    stream_configs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[Stream] | None:
    """Materialise configured stream dictionaries into ``Stream`` objects."""
    if stream_configs is None:
        return None

    if isinstance(stream_configs, Mapping):
        configs_iter = stream_configs.values()
    elif isinstance(stream_configs, Sequence) and not isinstance(
        stream_configs, (str, bytes)
    ):
        configs_iter = stream_configs
    else:  # pragma: no cover - defensive
        raise TypeError(
            "streams configuration must be a mapping or sequence of mappings"
        )

    return [
        Stream(
            remote=stream_config.get("remote"),
            local=stream_config.get("local"),
            proportion=stream_config.get("proportion", 1.0),
            download_retry=stream_config.get("download_retry", 2),
            download_timeout=stream_config.get("download_timeout", 60),
        )
        for stream_config in configs_iter
    ]


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

    def state_dict(
        self, num_samples: int | None = None, from_beginning: bool = True
    ) -> dict[str, Any]:
        """
        Saves the dataset's state.

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
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                f"expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
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
        raise ValueError("mosaic_dataloader config must be set.")

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

    # Build streams and optionally apply non-IID selections
    streams_cfg = dataset_cfg.pop("streams", None)
    client_streams_cfg = dataset_cfg.pop("client_streams", None)
    non_iid_cfg = dataset_cfg.pop("non_iid", None)

    if non_iid_cfg:
        selection = _select_non_iid_streams(non_iid_cfg, dp_rank=dp_rank)
        if selection is not None:
            logger.info("Applying non-IID stream selection for dp_rank %s", dp_rank)
            if selection.get("streams") is not None:
                streams_cfg = selection.get("streams")
            if selection.get("client_streams") is not None:
                client_streams_cfg = selection.get("client_streams")

    if client_streams_cfg is not None:
        dataset_cfg["client_streams"] = client_streams_cfg

    streams = _build_stream_objects(streams_cfg)
    if streams is not None:
        logger.info("Built %s streams for Mosaic dataloader", len(streams))

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
