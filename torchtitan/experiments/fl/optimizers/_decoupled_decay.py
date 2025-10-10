"""Shared helpers for decoupled weight decay handling."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_decoupled_weight_decay_factor(
    lr: float | Tensor, initial_lr: float | Tensor | None
) -> float | Tensor:
    """Return the scaling factor used for decoupled weight decay.

    The logic is shared across multiple optimizers and handles ``initial_lr``
    being ``None`` or ``0`` (as a Python ``float`` or :class:`~torch.Tensor`).
    In those cases the decay is treated as fully decoupled and the factor is ``1``.
    """
    if initial_lr is None:
        return 1.0

    if isinstance(initial_lr, Tensor):
        if torch.any(initial_lr == 0):
            return 1.0
        return lr / initial_lr

    if initial_lr == 0.0:
        return 1.0

    return lr / initial_lr


__all__ = ["compute_decoupled_weight_decay_factor"]

