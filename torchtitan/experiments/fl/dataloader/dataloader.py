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
from copy import deepcopy
from functools import partial
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


def titan_collate_fn(
    batch: list[Any], *, eos_id: int | None
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
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

    if eos_id is not None:
        sequence_boundaries = input_ids_tensor == eos_id
        sequence_boundaries[:, -1] = True
        cumulative_boundaries = torch.cumsum(
            sequence_boundaries.to(dtype=torch.int32), dim=1
        )
        sequence_id_full = torch.zeros_like(
            cumulative_boundaries, dtype=torch.int32
        )
        sequence_id_full[:, 1:] = cumulative_boundaries[:, :-1]
    else:
        sequence_id_full = torch.zeros(
            input_ids_tensor.shape, dtype=torch.int32, device=input_ids_tensor.device
        )

    model_inputs = input_ids_tensor[:, :-1].contiguous()
    labels = input_ids_tensor[:, 1:].contiguous()
    sequence_id = sequence_id_full[:, :-1].contiguous()

    input_dict = {"input": model_inputs, "sequence_id": sequence_id}
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

    # Build streams
    streams_cfg = dataset_cfg.pop("streams", None)
    streams = None
    if streams_cfg:
        streams = [
            Stream(
                remote=stream_config.get("remote"),
                local=stream_config.get("local"),
                proportion=stream_config.get("proportion", 1.0),
                download_retry=stream_config.get("download_retry", 2),
                download_timeout=stream_config.get("download_timeout", 60),
            )
            for stream_config in streams_cfg.values()
        ]
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
        collate_fn=partial(titan_collate_fn, eos_id=tokenizer.eos_id),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


__all__ = ["MosaicParallelAwareDataloader", "build_mosaic_dataloader"]
