# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel dataloader utilities tailored for Mosaic streaming datasets."""

from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.experiments.fl.metrics import UnigramMetricHandle

    from .dataset_factory import MosaicRuntimeConfig

try:
    from llmfoundry.data.text_data import StreamingTextDataset
    from streaming import Stream
except ImportError as exc:  # pragma: no cover - optional dependency
    msg = (
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    )
    raise RuntimeError(msg) from exc


_DIST_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@contextmanager
def _sanitized_dist_environment() -> Any:
    """Temporarily remove torch.distributed-related environment variables."""
    saved: dict[str, str | None] = {}
    for key in _DIST_ENV_VARS:
        saved[key] = os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _patch_streaming_distributed() -> Any:
    """Temporarily patch streaming.base.distributed to disable collectives."""
    try:
        import streaming.base.dataset as streaming_dataset
        import streaming.base.distributed as streaming_dist
    except ModuleNotFoundError:  # pragma: no cover - optional dependency absent (e.g., unit tests)
        yield
        return

    replacements = {
        "get_rank": lambda: 0,
        "get_world_size": lambda: 1,
        "get_local_rank": lambda: 0,
        "get_local_world_size": lambda: 1,
        "barrier": lambda: None,
        "broadcast": lambda tensor, src: None,
        "all_gather": lambda tensor_list, tensor: None,
        "maybe_init_dist": lambda: False,
    }

    saved: dict[str, Any] = {}
    for name, func in replacements.items():
        saved[name] = getattr(streaming_dist, name, None)
        setattr(streaming_dist, name, func)
        if hasattr(streaming_dataset, name):
            saved[f"dataset::{name}"] = getattr(streaming_dataset, name)
            setattr(streaming_dataset, name, func)
    try:
        yield
    finally:
        for name, func in saved.items():
            if name.startswith("dataset::"):
                attr = name.split("::", 1)[1]
                if func is None:
                    delattr(streaming_dataset, attr)
                else:
                    setattr(streaming_dataset, attr, func)
            elif func is None:
                delattr(streaming_dist, name)
            else:
                setattr(streaming_dist, name, func)


@contextmanager
def _patch_torch_dist_barrier() -> Any:
    """Temporarily disable torch.distributed barrier/destroy calls."""
    try:
        import torch.distributed as torch_dist
    except ImportError:  # pragma: no cover - defensive
        yield
        return

    saved_barrier = getattr(torch_dist, "barrier", None)
    saved_destroy = getattr(torch_dist, "destroy_process_group", None)

    def _noop(*args, **kwargs) -> None:
        return None

    if saved_barrier is not None:
        torch_dist.barrier = _noop  # type: ignore[attr-defined]
    if saved_destroy is not None:
        torch_dist.destroy_process_group = _noop  # type: ignore[attr-defined]

    try:
        yield
    finally:
        if saved_barrier is not None:
            torch_dist.barrier = saved_barrier  # type: ignore[attr-defined]
        if saved_destroy is not None:
            torch_dist.destroy_process_group = saved_destroy  # type: ignore[attr-defined]


_PREFIX_PATCHED = False


def _patch_streaming_prefix_once() -> None:
    global _PREFIX_PATCHED
    if _PREFIX_PATCHED:
        return

    try:
        import streaming.base.dataset as streaming_dataset
        import streaming.base.shared as streaming_shared
        import streaming.base.shared.prefix as streaming_prefix
    except ModuleNotFoundError:  # pragma: no cover - optional dependency absent
        _PREFIX_PATCHED = True
        return

    _check_self = streaming_prefix._check_self
    _check_and_find_retrying = streaming_prefix._check_and_find_retrying
    SharedMemory = streaming_prefix.SharedMemory
    _get_path = streaming_prefix._get_path
    _pack_locals = streaming_prefix._pack_locals
    _unpack_locals = streaming_prefix._unpack_locals
    SHM_TO_CLEAN = streaming_prefix.SHM_TO_CLEAN
    LOCALS = streaming_prefix.LOCALS

    def get_shm_prefix_no_barrier(streams_local, streams_remote, world, retry=100):
        _check_self(streams_local)

        prefix_int = max(
            [
                _check_and_find_retrying(
                    streams_local, streams_remote, shm_name=shm_name, retry=retry
                )
                for shm_name in SHM_TO_CLEAN
            ]
        )

        shm = None
        if world.is_local_leader:
            name = _get_path(prefix_int, LOCALS)
            data = _pack_locals(streams_local, prefix_int)
            shm = SharedMemory(name, True, len(data))
            shm.buf[: len(data)] = data

        if not world.is_local_leader:
            name = _get_path(prefix_int, LOCALS)
            try:
                shm = SharedMemory(name, False)
            except FileNotFoundError:
                msg = "Internal error: shared memory prefix was not registered by local leader."
                raise RuntimeError(msg)

            their_locals, their_prefix_int = _unpack_locals(bytes(shm.buf))
            if streams_local != their_locals or prefix_int != their_prefix_int:
                msg = "Internal error: shared memory registered does not match local leader"
                raise RuntimeError(msg)
        assert shm is not None
        return prefix_int, shm

    streaming_prefix.get_shm_prefix = get_shm_prefix_no_barrier
    streaming_shared.get_shm_prefix = get_shm_prefix_no_barrier
    streaming_dataset.get_shm_prefix = get_shm_prefix_no_barrier
    _PREFIX_PATCHED = True


def serialize_streams(streams: list[Stream]) -> list[dict[str, Any]]:
    """Serialize Stream objects into dictionaries for cross-process transfer."""
    serialized: list[dict[str, Any]] = []
    for stream in streams:
        serialized.append(
            {
                "remote": getattr(stream, "remote", None),
                "local": getattr(stream, "local", None),
                "split": getattr(stream, "split", None),
                "proportion": getattr(stream, "_proportion", None),
                "repeat": getattr(stream, "_repeat", None),
                "choose": getattr(stream, "_choose", None),
                "download_retry": getattr(stream, "_download_retry", None),
                "download_timeout": getattr(stream, "_download_timeout", None),
                "validate_hash": getattr(stream, "validate_hash", None),
                "keep_zip": getattr(stream, "_keep_zip", None),
            }
        )
    return serialized


def _deserialize_streams(stream_payload: list[dict[str, Any]]) -> list[Stream]:
    """Deserialize stream configuration dictionaries back into Stream objects."""
    return [Stream(**payload) for payload in stream_payload]


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
        effective_num_samples = (
            num_samples if num_samples is not None else self._num_samples_yielded
        )
        parent_state = super().state_dict(
            num_samples=effective_num_samples, from_beginning=from_beginning
        )
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """Restore the dataset state from a serialized representation.

        Args:
            obj: State dictionary produced by :meth:`state_dict`.
        """
        self._num_samples_yielded = obj.pop("_num_samples_yielded", 0)
        super().load_state_dict(obj)


def _with_streaming_patch(fn):
    def wrapper(self, *args, **kwargs):
        with _patch_streaming_distributed(), _patch_torch_dist_barrier():
            return fn(self, *args, **kwargs)

    return wrapper


class BarrierFreeStreamingTextDataset(StatefulStreamingTextDataset):
    """StatefulStreamingTextDataset variant that disables streaming barriers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _patch_streaming_prefix_once()
        with _patch_streaming_distributed(), _patch_torch_dist_barrier():
            super().__init__(*args, **kwargs)

    @_with_streaming_patch
    def __iter__(self):
        return super().__iter__()

    @_with_streaming_patch
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().state_dict(*args, **kwargs)

    @_with_streaming_patch
    def load_state_dict(self, obj: dict[str, Any]) -> None:
        super().load_state_dict(obj)

    @_with_streaming_patch
    def close(self) -> None:
        close_fn = getattr(super(), "close", None)
        if callable(close_fn):
            close_fn()


class IsolatedStreamingTextDataset(IterableDataset):
    """Dataset wrapper that constructs StreamingTextDataset inside the consuming process."""

    def __init__(
        self,
        *,
        dataset_kwargs: dict[str, Any],
        serialized_streams: list[dict[str, Any]],
        tokenizer: Any,
        batch_size: int,
    ) -> None:
        self._dataset_kwargs = dict(dataset_kwargs)
        self._serialized_streams = list(serialized_streams)
        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self._dataset: StatefulStreamingTextDataset | None = None
        self._num_samples_yielded = 0

    def _ensure_dataset(self) -> StatefulStreamingTextDataset:
        if self._dataset is None:
            _patch_streaming_prefix_once()
            streams = (
                _deserialize_streams(self._serialized_streams)
                if self._serialized_streams
                else None
            )
            with _sanitized_dist_environment(), _patch_streaming_distributed(), _patch_torch_dist_barrier():
                self._dataset = BarrierFreeStreamingTextDataset(
                    tokenizer=self._tokenizer,
                    streams=streams,
                    batch_size=self.batch_size,
                    **self._dataset_kwargs,
                )
        return self._dataset

    def __iter__(self):
        dataset = self._ensure_dataset()
        return iter(dataset)

    def state_dict(
        self, num_samples: int | None = None, *, from_beginning: bool = True
    ) -> dict[str, Any]:
        dataset = self._ensure_dataset()
        return dataset.state_dict(
            num_samples=num_samples, from_beginning=from_beginning
        )

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        dataset = self._ensure_dataset()
        dataset.load_state_dict(obj)

    def close(self) -> None:
        dataset = self._dataset
        if dataset is not None and hasattr(dataset, "close"):
            try:
                dataset.close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to close dataset cleanly", exc_info=True)
        self._dataset = None

    def __del__(self) -> None:  # pragma: no cover - cleanup best effort
        with suppress(Exception):
            self.close()

    def __getstate__(self) -> dict[str, Any]:  # pragma: no cover - simple state hook
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    def __setstate__(
        self, state: dict[str, Any]
    ) -> None:  # pragma: no cover - simple state hook
        self.__dict__.update(state)


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

    def __init__(
        self, dataset: IterableDataset, request: ParallelDataLoaderRequest
    ) -> None:
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
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "close"):
            try:
                dataset.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to close dataset cleanly", exc_info=True)


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
    "IsolatedStreamingTextDataset",
    "MosaicParallelAwareDataloader",
    "ParallelDataLoaderRequest",
    "StatefulStreamingTextDataset",
    "serialize_streams",
    "titan_collate_fn",
]
