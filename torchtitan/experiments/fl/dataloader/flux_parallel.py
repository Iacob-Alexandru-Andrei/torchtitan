# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FL dataloader that mirrors Mosaic's worker-friendly wrapper for Flux."""

from __future__ import annotations

import pickle
from copy import deepcopy
from typing import Any

from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader


class FluxParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Parallel-aware dataloader with worker/prefetch controls for Flux."""

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        collate_fn: Any | None = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        self._rank_id = f"dp_rank_{dp_rank}"

        # Disable worker-specific flags when num_workers == 0 to avoid warnings.
        if num_workers <= 0:
            num_workers = 0
            prefetch_factor = None
            persistent_workers = False

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

    def state_dict(self) -> dict[str, Any]:
        """Serialize dataloader state for checkpointing."""
        return {
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore dataloader state from a checkpoint payload."""
        if not state_dict:
            return
        if self._rank_id not in state_dict:
            return
        assert (
            self.dp_world_size == state_dict["world_size"]
        ), "dp_degree changed; dataloader resharding is not supported."
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))

    def close(self) -> None:
        """Close the underlying dataset if it exposes a close method."""
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "close"):
            close_fn = getattr(dataset, "close")
            try:
                close_fn()
            except Exception:
                # best effort cleanup
                pass
