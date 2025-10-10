# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Custom optimizer hyperparameters for decoupled and quasi-hyperbolic optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from torchtitan.config import Optimizer as BaseOptimizer


@dataclass
class DesLocConfig:
    """Configuration options for the Desynchronized Local SGD strategy."""

    enabled: bool = False
    """Whether to enable DES-LOC synchronization."""

    param_sync_every: int = 1
    """Number of optimizer steps between parameter synchronizations."""

    optimizer_sync_every: int | list[int] | dict[str, int] | None = None
    """Synchronization frequency for optimizer states.

    If ``None`` the parameter synchronization cadence is reused. A single integer
    applies to every optimizer state tensor. A list specifies the cadence per
    discovered state (ordered alphabetically), while a dict maps explicit state
    names (e.g. ``{"exp_avg": 4}``).
    """

    backup_device: str | torch.device | None = "cpu"
    """Device used to keep fault-tolerance copies of parameters and optimizer state."""

    pin_memory: bool = True
    """Whether to pin the CPU buffers used for the DES-LOC backups."""

    def resolved_backup_device(self) -> torch.device | None:
        """Convert the configured ``backup_device`` into a ``torch.device``."""
        device = self.backup_device
        if device is None:
            return None
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            return torch.device(device)
        msg = (
            "backup_device must be a string, torch.device, or None; "
            f"received {type(device)!r}"
        )
        raise TypeError(msg)

    def normalized_optimizer_sync(self) -> int | list[int] | dict[str, int] | None:
        """Return the optimizer sync specification in a stable format."""
        spec = self.optimizer_sync_every
        if spec is None:
            return None
        if isinstance(spec, dict):
            return {str(k): int(v) for k, v in spec.items()}
        if isinstance(spec, list):
            return [int(v) for v in spec]
        return int(spec)


@dataclass
class MosaicOptimizerConfig(BaseOptimizer):
    """Mosaic-specific optimizer config with additional hyperparameters."""

    vs: tuple[float, ...] = (0.7,)
    """vs hyperparameters for quasi-hyperbolic optimizers (each optimizer extracts as many as needed)"""

    decouple: bool = True
    """Whether to decouple the learning rate from the weight decay"""

    desloc: DesLocConfig = field(default_factory=DesLocConfig)
    """Configuration block for the DES-LOC synchronization strategy."""
