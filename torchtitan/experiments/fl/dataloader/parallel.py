"""Parallel-aware dataloader utilities for Mosaic streaming datasets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.experiments.fl.metrics import UnigramMetricHandle

try:
    from llmfoundry.data.text_data import StreamingTextDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc


class StatefulStreamingTextDataset(StreamingTextDataset):
    """Stateful wrapper that keeps track of yielded samples."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_samples_yielded = 0

    def __getitem__(self, idx: int) -> dict[str, list[int]] | torch.Tensor:
        """Increment the internal counter each time an item is fetched."""

        self._num_samples_yielded += 1
        return super().__getitem__(idx)

    def state_dict(
        self, num_samples: int | None = None, *, from_beginning: bool = True
    ) -> dict[str, Any]:
        """Saves the dataset's state compatible with TorchTitan dataloaders."""

        effective_num_samples = num_samples if num_samples is not None else self._num_samples_yielded
        parent_state = super().state_dict(num_samples=effective_num_samples, from_beginning=from_beginning)
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Restores the dataset's state from a state dictionary."""

        self._num_samples_yielded = obj.pop("_num_samples_yielded", 0)
        super().load_state_dict(obj)


@dataclass(frozen=True)
class ParallelDataLoaderRequest:
    """Parameters required to construct a :class:`MosaicParallelAwareDataloader`."""

    dp_rank: int
    dp_world_size: int
    runtime: "MosaicRuntimeConfig"
    collate_fn: Any | None = None
    group_key: str | None = None
    unigram_handle: "UnigramMetricHandle" | None = None


class MosaicParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader for Mosaic StreamingTextDataset with distributed data parallelism support."""

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(self, dataset: StatefulStreamingTextDataset, request: ParallelDataLoaderRequest) -> None:
        runtime = request.runtime
        self.dp_world_size = request.dp_world_size
        self.dp_rank = request.dp_rank
        self.batch_size = runtime.batch_size
        super().__init__(
            dataset,
            batch_size=runtime.batch_size,
            collate_fn=request.collate_fn,
            num_workers=runtime.num_workers,
            prefetch_factor=runtime.prefetch_factor,
            pin_memory=runtime.pin_memory,
            persistent_workers=runtime.persistent_workers,
            drop_last=runtime.drop_last,
        )
        self._rank_id = f"dp_rank_{request.dp_rank}"
        self.unigram_group_key = request.group_key
        self.unigram_metric_handle = request.unigram_handle

    def state_dict(self) -> dict[str, Any]:
        """Save dataloader state for checkpointing."""

        return {
            self._rank_id: deepcopy(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state from checkpoint."""

        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                "DataLoader state is empty for dp rank %s, expected key %s",
                self.dp_rank,
                self._rank_id,
            )
            return

        assert (
            self.dp_world_size == state_dict["world_size"]
        ), "dp_degree is inconsistent before and after checkpoint, dataloader resharding is not supported yet."
        loader_state = state_dict[self._rank_id]
        super().load_state_dict(deepcopy(loader_state))

    def close(self) -> None:
        """Close the dataloader and release resources."""

        if self.unigram_metric_handle is not None:
            self.unigram_metric_handle.close()
            self.unigram_metric_handle = None


def titan_collate_fn(batch: list[Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate samples from :class:`StreamingTextDataset` for TorchTitan."""

    if isinstance(batch[0], dict):
        input_ids_list = [item["input_ids"] for item in batch]
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        else:
            input_ids_tensor = torch.stack(input_ids_list)
    elif isinstance(batch[0], torch.Tensor):
        input_ids_tensor = torch.stack(batch)
    else:
        msg = f"Unsupported batch item type from dataset: {type(batch[0])}"
        raise TypeError(msg)

    model_inputs = input_ids_tensor[:, :-1].contiguous()
    labels = input_ids_tensor[:, 1:].contiguous()
    input_dict = {"input": model_inputs}
    return input_dict, labels


__all__ = [
    "MosaicParallelAwareDataloader",
    "ParallelDataLoaderRequest",
    "StatefulStreamingTextDataset",
    "titan_collate_fn",
]
