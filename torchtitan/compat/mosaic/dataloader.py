# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapters for using Mosaic streaming dataloaders with TorchTitan."""

from __future__ import annotations

import inspect
import pickle
from copy import deepcopy
from typing import Any

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

try:
    from llmfoundry.data.text_data import StreamingTextDataset  # type: ignore[import]
    from streaming import StreamingDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "llm-foundry and streaming are required to build Mosaic dataloaders. "
        "Please install llm-foundry, mosaicml-streaming, and composer to enable this integration."
    ) from exc

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
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
        # Use the externally provided num_samples if available, otherwise use our tracker.
        effective_num_samples = (
            num_samples if num_samples is not None else self._num_samples_yielded
        )

        # Get the state dictionary from the parent StreamingDataset
        parent_state = super().state_dict(
            num_samples=effective_num_samples, from_beginning=from_beginning
        )

        # Add our internal tracker to the state for correct restoration.
        parent_state["_num_samples_yielded"] = self._num_samples_yielded
        return parent_state

    def load_state_dict(self, obj: dict[str, Any]) -> None:
        """
        Restores the dataset's state from a state dictionary.
        """
        # Restore our internal sample counter first.
        self._num_samples_yielded = obj.pop("_num_samples_yielded", 0)

        # Now, pass the rest of the state to the parent class to handle.
        # The parent's load_state_dict will correctly restore the stream position.
        super().load_state_dict(obj)


def _maybe_extract_hf_tokenizer(tokenizer: BaseTokenizer | Any | None) -> Any:
    """Return a HuggingFace tokenizer if ``tokenizer`` wraps one."""

    if tokenizer is None:
        return None

    try:  # Import lazily so TorchTitan does not hard depend on transformers.
        from transformers import PreTrainedTokenizerBase  # type: ignore
    except Exception:  # pragma: no cover - transformers may be absent
        PreTrainedTokenizerBase = ()  # type: ignore[assignment]

    if isinstance(tokenizer, PreTrainedTokenizerBase):
        return tokenizer

    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if isinstance(inner_tokenizer, PreTrainedTokenizerBase):
        return inner_tokenizer

    return tokenizer


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
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state from checkpoint."""
        # State being empty is valid.
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
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def titan_collate_fn(batch: list[Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Collates samples from StreamingTextDataset and formats them for the
    TorchTitan training loop.

    This function takes a batch of token sequences, creates the input and label
    tensors for next-token prediction, and formats them into the
    (input_dict, labels) tuple.

    Args:
        batch: A list of samples from the dataset. Each sample is either a
               dictionary containing 'input_ids' or a single tensor of token IDs.

    Returns:
        A tuple where the first element is an `input_dict` (containing a tensor
        of input IDs) and the second is a tensor of corresponding labels.
    """
    # 1. Extract token IDs and stack them into a single batch tensor
    if isinstance(batch[0], dict):
        # Case: tokenizer returns a dict {'input_ids': ..., 'attention_mask': ...}
        # Note: We are ignoring the attention_mask here as the model might not need it,
        # but it could be added to input_dict if required.
        input_ids_list = [item["input_ids"] for item in batch]
        # The tokenizer might return lists, so we convert them to tensors
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        else:
            input_ids_tensor = torch.stack(input_ids_list)

    elif isinstance(batch[0], torch.Tensor):
        # Case: dataset returns a single tensor of tokens
        input_ids_tensor = torch.stack(batch)
    else:
        raise TypeError(f"Unsupported batch item type from dataset: {type(batch[0])}")

    # 2. Create the model inputs and labels for next-token prediction
    # Inputs are all tokens except the last one
    model_inputs = input_ids_tensor[:, :-1].contiguous()
    # Labels are all tokens except the first one (they are shifted by one)
    labels = input_ids_tensor[:, 1:].contiguous()

    # 3. Format into the (input_dict, labels) structure
    input_dict = {"input": model_inputs}

    return input_dict, labels


def build_mosaic_dataloader(
    *,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
) -> MosaicParallelAwareDataloader:
    """Build a dataloader using Mosaic's StreamingTextDataset directly.

    This function constructs a StreamingTextDataset and wraps it in a
    MosaicParallelAwareDataloader that supports prefetching and distributed
    data parallelism, similar to how build_hf_dataloader works.
    """

    mosaic_cfg = job_config.training.mosaic.mosaic_dataloader
    if mosaic_cfg is None:
        raise ValueError(
            "job_config.training.mosaic.mosaic_dataloader must be set before constructing "
            "the Trainer. Use the Mosaic example CLI to attach the configuration "
            "prior to instantiating TorchTitan's Trainer."
        )

    # Get the HuggingFace tokenizer
    hf_tokenizer = _maybe_extract_hf_tokenizer(tokenizer)
    if hf_tokenizer is None:
        # Try to build tokenizer from mosaic config if provided
        mosaic_tokenizer_cfg = job_config.training.mosaic.mosaic_tokenizer
        if mosaic_tokenizer_cfg and "name" in mosaic_tokenizer_cfg:
            try:
                from transformers import AutoTokenizer

                tokenizer_name = mosaic_tokenizer_cfg["name"]
                tokenizer_kwargs = mosaic_tokenizer_cfg.get("kwargs", {})
                hf_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, **tokenizer_kwargs
                )
                logger.info(f"Loaded tokenizer from mosaic config: {tokenizer_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from mosaic config: {e}")
                raise ValueError(
                    "Could not extract or build HuggingFace tokenizer"
                ) from e
        else:
            raise ValueError(
                "Mosaic dataloader requires a HuggingFace tokenizer, but none was provided "
                "and no tokenizer configuration was found in training.mosaic.mosaic_tokenizer"
            )

    cfg = deepcopy(mosaic_cfg)
    dataset_cfg = cfg.get("dataset", {})

    # Extract dataloader-specific config
    num_workers = cfg.get("num_workers", 8)
    prefetch_factor = cfg.get("prefetch_factor", 2)
    pin_memory = cfg.get("pin_memory", True)
    persistent_workers = cfg.get("persistent_workers", True)
    drop_last = cfg.get("drop_last", True)

    # Build streams if present
    streams_cfg = dataset_cfg.pop("streams", None)
    streams = None
    if streams_cfg:
        try:
            from streaming import Stream

            streams = []
            for stream_name, stream_config in streams_cfg.items():
                stream = Stream(
                    remote=stream_config.get("remote"),
                    local=stream_config.get("local"),
                    proportion=stream_config.get("proportion", 1.0),
                    download_retry=stream_config.get("download_retry", 2),
                    download_timeout=stream_config.get("download_timeout", 60),
                )
                streams.append(stream)
            logger.info(f"Built {len(streams)} streams for Mosaic dataloader")
        except Exception as e:
            logger.error(f"Failed to build streams: {e}")
            raise

    # Filter dataset config to only include valid StreamingTextDataset parameters
    valid_streaming_text_params = inspect.signature(StreamingTextDataset).parameters
    valid_base_dataset_params = inspect.signature(StreamingDataset).parameters

    dataset_config_filtered = {
        k: v
        for k, v in dataset_cfg.items()
        if k in valid_streaming_text_params or k in valid_base_dataset_params
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

    # Get collate function if needed
    collate_fn = titan_collate_fn
    # Most StreamingTextDataset configurations handle collation internally
    # If you need a custom collate_fn, you can add logic here

    # Wrap in our parallel-aware dataloader with prefetching support
    return MosaicParallelAwareDataloader(
        dataset=text_dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


__all__ = ["MosaicParallelAwareDataloader", "build_mosaic_dataloader"]
