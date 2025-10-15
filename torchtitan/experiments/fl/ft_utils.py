# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities specific to FL experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.components.checkpoint import LR_SCHEDULER, MODEL, OPTIMIZER
from torchtitan.tools.logging import logger

if TYPE_CHECKING:  # pragma: no cover
    from torchft.manager import Manager as TorchFTManager

    from torchtitan.train import Trainer


_INIT_SYNC_KEY = "fl_initial_sync"


def ensure_torchft_init_sync(trainer: Trainer) -> None:
    """Register a TorchFT state handler so replicas share the same initial weights.

    TorchFT relies on user-provided state accessors when broadcasting state to
    new replicas (for both init-sync and recovery).  DDP implicitly broadcasts
    parameters from rank 0, so without this hook TorchFT replicas can diverge
    at initialization.  We reuse the checkpoint manager's wrapped components to
    keep the registration consistent with the rest of the system.
    """
    ft_manager_wrapper = getattr(trainer, "ft_manager", None)
    if ft_manager_wrapper is None or not getattr(ft_manager_wrapper, "enabled", False):
        return

    checkpointer = getattr(trainer, "checkpointer", None)
    if checkpointer is None:
        logger.debug("TorchFT init sync skipped: trainer has no checkpointer")
        return

    torchft_manager: TorchFTManager | None = getattr(checkpointer, "ft_manager", None)
    if torchft_manager is None:
        logger.debug("TorchFT init sync skipped: TorchFT manager unavailable")
        return

    # torchft.Manager keeps user registrations in _user_state_dicts.
    # Skip registration if something else already set up the helper.
    already_registered = _INIT_SYNC_KEY in getattr(
        torchft_manager, "_user_state_dicts", {}
    )
    if already_registered:
        return

    states: dict[str, object] | None = getattr(checkpointer, "states", None)
    if not states:
        logger.debug("TorchFT init sync skipped: no checkpoint states to register")
        return

    tracked_entries = {}
    for key in (MODEL, OPTIMIZER, LR_SCHEDULER):
        entry = states.get(key)
        if entry is not None:
            tracked_entries[key] = entry

    if MODEL not in tracked_entries:
        logger.warning(
            "TorchFT init sync skipped because the model state wrapper is missing"
        )
        return

    first_load = True

    def _state_dict():
        return {key: entry.state_dict() for key, entry in tracked_entries.items()}

    def _load_state_dict(payload) -> None:
        nonlocal first_load
        for key, entry in tracked_entries.items():
            if key in payload:
                entry.load_state_dict(payload[key])
        if first_load:
            logger.info("TorchFT init sync state applied to local training components")
            first_load = False

    torchft_manager.register_state_dict_fn(
        _INIT_SYNC_KEY, _load_state_dict, _state_dict
    )
    logger.info("Registered TorchFT initial sync handler for FL experiments")
