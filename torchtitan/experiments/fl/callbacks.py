# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from torchtitan.components.metrics import BaseLogger
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config import JobConfig
    from torchtitan.distributed import ParallelDims


@dataclass(slots=True)
class CallbackSetupContext:
    """Context provided to callbacks during setup."""

    model_parts: Sequence[nn.Module]
    optimizers: OptimizersContainer
    logger: BaseLogger
    parallel_dims: ParallelDims
    job_config: JobConfig


@dataclass(slots=True)
class CallbackStepContext:
    """Context provided to callbacks at the end of a training step."""

    step: int
    model_parts: Sequence[nn.Module]
    optimizers: OptimizersContainer
    logger: BaseLogger
    mesh: DeviceMesh | None


@dataclass(slots=True)
class CallbackValidationContext:
    """Context provided to callbacks when validation logging finishes."""

    step: int
    loss: float
    logger: BaseLogger


class Callback:
    """Lightweight callback interface used by FL experiments."""

    def setup(self, context: CallbackSetupContext) -> None:
        """Called once when model parts and optimizers are available."""

    def on_step_end(self, context: CallbackStepContext) -> None:
        """Called after metrics logging for every training step."""

    def on_validation_end(self, context: CallbackValidationContext) -> None:
        """Called when validation metrics are logged."""

    def close(self) -> None:
        """Called when training finishes."""
