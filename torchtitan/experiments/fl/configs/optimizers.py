# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Custom optimizer hyperparameters for decoupled and quasi-hyperbolic optimizers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, cast, Literal

import torch

from torchtitan.config import Optimizer as BaseOptimizer

from torch.optim import Optimizer

# Default values from BaseOptimizer
_MIN_BETAS_LENGTH = 2


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

    quorum_timeout_seconds: int = 60
    """Timeout (seconds) to wait for TorchFT quorum formation during DES-LOC sync."""

    outer_optimizer: "DesLocOuterOptimizerConfig | None" = None
    """Optional optimizer to apply averaged pseudo-gradients to global parameters."""

    log_outer_metrics: bool = False
    """Whether to log DES-LOC outer optimizer pseudo-gradient and momentum norms."""

    def resolved_backup_device(self) -> torch.device | None:
        """Convert the configured ``backup_device`` into a ``torch.device``."""
        device = self.backup_device
        if device is None:
            return None
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            return torch.device(device)
        msg = f"backup_device must be a string, torch.device, or None; received {type(device)!r}"
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

    def normalized_outer_optimizer(self) -> "DesLocOuterOptimizerConfig | None":
        """Return a normalized outer optimizer configuration if provided."""
        outer = self.outer_optimizer
        if outer is None:
            return None
        if isinstance(outer, DesLocOuterOptimizerConfig):
            if outer.target is None:
                if outer.kwargs:
                    msg = "desloc.outer_optimizer.kwargs requires a target optimizer."
                    raise ValueError(msg)
                return None
            return outer
        if isinstance(outer, dict):
            target = outer.get("target")
            kwargs = outer.get("kwargs", {})
            if target is None:
                if kwargs:
                    msg = "desloc.outer_optimizer.kwargs requires a target optimizer."
                    raise ValueError(msg)
                return None
            if not isinstance(kwargs, dict):
                msg = "desloc.outer_optimizer.kwargs must be a mapping."
                raise TypeError(msg)
            return DesLocOuterOptimizerConfig(target=target, kwargs=dict(kwargs))
        msg = (
            "desloc.outer_optimizer must be a DesLocOuterOptimizerConfig, mapping, or None; "
            f"received {type(outer)!r}."
        )
        raise TypeError(msg)


@dataclass
class MosaicOptimizerConfig(BaseOptimizer):
    """Mosaic-specific optimizer config with additional hyperparameters."""

    desloc: DesLocConfig = field(default_factory=DesLocConfig)
    """Optional DES-LOC configuration."""

    vs: tuple[float, ...] = (0.7,)
    """vs hyperparameters for quasi-hyperbolic optimizers (each optimizer extracts as many as needed)"""

    decouple: bool = True
    """Whether to decouple the learning rate from the weight decay"""

    betas: tuple[float, ...] | None = None
    """
    Optional explicit betas tuple for AggMo optimizers.
    If provided, must have length = number of non-zero vs + 1 (last element is beta2).
    If None, betas will be constructed from beta1 and beta2 fields for compatibility.
    Example: For vs=(0.7, 0.2), betas=(0.9, 0.99, 0.95) means beta1_1=0.9, beta1_2=0.99, beta2=0.95.
    """

    builder: Literal["mosaic", "default"] = "mosaic"
    """Selector for the optimizer builder.

    * ``"mosaic"`` uses the FL-specific builder with Mosaic optimizers and DES-LOC support.
    * ``"default"`` delegates to the core TorchTitan optimizer builder.
    """

    norm: str = "Auto"
    """Norm backend to use with Scion optimizers."""

    norm_kwargs: dict[str, Any] | None = None
    """Additional keyword arguments for the selected norm backend."""

    scale: float = 1.0
    """Scalar multiplier applied to Scion updates."""

    unconstrained: bool = False
    """Disable the multiplicative shrinkage used by Scion when True."""

    scion_v: float | None = None
    """Quasi-hyperbolic mixing parameter for :class:`ScionQH`."""

    scion_momentums: tuple[float, ...] = (1.0,)
    """First-moment beta coefficients for :class:`ScionAggMo`."""

    scion_weights: tuple[float, ...] | None = None
    """Combination weights for :class:`ScionAggMo`."""

    galore_rank: int | None = None
    """Default low-rank size for GaLore. None disables projection."""

    galore_update_proj_gap: int = 200
    """Number of steps between GaLore projector refreshes."""

    galore_scale: float = 1.0
    """Scaling factor applied when projecting back to full rank."""

    galore_proj_type: str = "std"
    """Projection strategy for GaLore (std, reverse_std, left, right, full)."""

    galore_dim: int = 2
    """Expected tensor dimensionality for GaLore projections."""

    galore_v1: float = 0.0
    """Quasi-hyperbolic coefficient for GaLore first momentum."""

    def __post_init__(self) -> None:
        """Auto-initialize beta1 and beta2 from betas if betas is provided."""
        builder = self.builder.lower()
        if builder not in {"mosaic", "default"}:
            msg = "optimizer.builder must be either 'mosaic' or 'default'"
            raise ValueError(msg)
        self.builder = cast("Literal['mosaic', 'default']", builder)

        if isinstance(self.desloc, dict):
            self.desloc = DesLocConfig(**self.desloc)
        if self.desloc.quorum_timeout_seconds <= 0:
            msg = "desloc.quorum_timeout_seconds must be positive"
            raise ValueError(msg)
        if self.betas is not None and len(self.betas) >= _MIN_BETAS_LENGTH:
            # If betas is provided, it always overrides beta1 and beta2
            # beta1 comes from the first element, beta2 from the last element
            self.beta1 = self.betas[0]
            self.beta2 = self.betas[-1]

        if isinstance(self.norm_kwargs, dict):
            self.norm_kwargs = dict(self.norm_kwargs)

        if not 0.0 <= self.galore_v1 <= 1.0:
            msg = "optimizer.galore_v1 must be in [0, 1]"
            raise ValueError(msg)
        if self.galore_rank is not None and self.galore_rank <= 0:
            msg = "optimizer.galore_rank must be positive when set."
            raise ValueError(msg)
        if self.galore_update_proj_gap <= 0:
            msg = "optimizer.galore_update_proj_gap must be positive."
            raise ValueError(msg)

    def get_betas_tuple(self) -> tuple[float, ...]:
        """Get the betas tuple, either from explicit betas or constructed from beta1/beta2.

        For AggMo optimizers, returns a tuple where:
        - All elements except the last are beta1_i for each momentum buffer
        - The last element is beta2

        Returns:
            Tuple of beta values
        """
        if self.betas is not None:
            return self.betas

        # Count non-zero vs values (number of momentum buffers)
        num_moments = sum(1 for v in self.vs if v != 0.0)

        # Construct betas: (beta1, beta1, ..., beta2) with num_moments beta1s
        return tuple([self.beta1] * num_moments + [self.beta2])


@dataclass(frozen=True)
class DesLocOuterOptimizerConfig:
    """Serializable configuration for DES-LOC's outer optimizer selection."""

    target: str | type[Optimizer] | None = None
    kwargs: dict[str, object] = field(default_factory=dict)

    def resolve_optimizer_cls(self) -> type[Optimizer]:
        """Materialize the configured optimizer class."""
        target = self.target
        if target is None:
            msg = "desloc.outer_optimizer.target must be configured before use."
            raise ValueError(msg)
        if isinstance(target, type):
            if not issubclass(target, Optimizer):
                msg = f"Configured outer optimizer class {target!r} is not an Optimizer."
                raise TypeError(msg)
            return target

        if not isinstance(target, str):
            msg = (
                "desloc.outer_optimizer.target must be a string or Optimizer subclass; "
                f"received {type(target)!r}."
            )
            raise TypeError(msg)

        module_path, _, attr = target.rpartition(".")
        if module_path:
            module = importlib.import_module(module_path)
            optimizer_cls = getattr(module, attr, None)
        else:
            optimizer_cls = getattr(torch.optim, attr, None)

        if optimizer_cls is None or not issubclass(optimizer_cls, Optimizer):
            msg = (
                f"Failed to resolve DES-LOC outer optimizer '{target}'. Ensure it refers "
                "to a torch.optim.Optimizer subclass."
            )
            raise ValueError(msg)
        return optimizer_cls
