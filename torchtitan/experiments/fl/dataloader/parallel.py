"""Parallel dataloader utilities tailored for Mosaic streaming datasets."""

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

    from .dataset_factory import MosaicRuntimeConfig

try:
    from llmfoundry.data.text_data import StreamingTextDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc


class StatefulStreamingTextDataset(StreamingTextDataset):
    """Stateful wrapper that keeps track of yielded samples.

    Args:
        *args: Positional arguments forwarded to ``StreamingTextDataset``.
        **kwargs: Keyword arguments forwarded to ``StreamingTextDataset``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_samples_yielded = 0

    def __getitem__(self, idx: int) -> dict[str, list[int]] | torch.Tensor:
        """Increment the internal counter each time an item is fetched.

        Args:
            idx: Index of the sample requested by the dataloader.

        Returns:
            The sample retrieved from the parent dataset implementation.
        """
        self._num_samples_yielded += 1
        return super().__getitem__(idx)

    def state_dict(
        self, num_samples: int | None = None, *, from_beginning: bool = True
    ) -> dict[str, Any]:
        """Save the dataset state in a TorchTitan-compatible format.

        Args:
            num_samples: Optional number of samples to persist.
            from_beginning: Whether to rewind the iterator before saving.

        Returns:
            Serialized dataset state, including the yielded sample count.
        """
        effective_num_samples = num_samples if num_samples is not None else self._num_samples_yielded
        parent_state = super().state_dict(num_samples=effective_num_samples, from_beginning=from_beginning)
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Restore the dataset state from a serialized representation.

        Args:
            obj: State dictionary produced by :meth:`state_dict`.
        """
        self._num_samples_yielded = obj.pop("_num_samples_yielded", 0)
        super().load_state_dict(obj)


@dataclass(frozen=True)
class ParallelDataLoaderRequest:
    """Parameters used to construct a :class:`MosaicParallelAwareDataloader`.

    Args:
        dp_rank: Data parallel rank handled by the current process.
        dp_world_size: Total number of data parallel ranks.
        runtime: Runtime configuration describing worker behavior.
        collate_fn: Optional collate function applied to dataset samples.
        group_key: Optional key used to group unigram metrics by stream.
        unigram_handle: Optional handle to a registered unigram metric.
    """

    dp_rank: int
    dp_world_size: int
    runtime: MosaicRuntimeConfig
    collate_fn: Any | None = None
    group_key: str | None = None
    unigram_handle: UnigramMetricHandle | None = None


class MosaicParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader for Mosaic streaming datasets with distributed DP support.

    Args:
        dataset: Streaming dataset backed by Mosaic streaming primitives.
        request: Request describing runtime parameters for the dataloader.
    """

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
        """Serialize dataloader state for checkpointing.

        Returns:
            Mapping containing the state keyed by the rank identifier.
        """
        return {
            self._rank_id: deepcopy(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state from a checkpoint payload.

        Args:
            state_dict: Serialized state previously returned by :meth:`state_dict`.
        """
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
    """Collate samples from :class:`StreamingTextDataset` for TorchTitan.

    Args:
        batch: Batch generated by :class:`StreamingTextDataset` instances.

    Returns:
        Tuple consisting of the input dictionary and label tensor expected by the
        training loop.

    Raises:
        TypeError: If an unsupported batch structure is encountered.
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
