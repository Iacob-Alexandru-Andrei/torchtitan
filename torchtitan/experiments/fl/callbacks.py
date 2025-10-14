# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.components.metrics import (
    MetricsCallback,
    MetricsCallbackSetupContext,
    MetricsCallbackStepContext,
    MetricsCallbackValidationContext,
)


if TYPE_CHECKING:
    from torchtitan.components.metrics import BaseLogger
    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.config import JobConfig
    from torchtitan.distributed import ParallelDims


CallbackSetupContext = MetricsCallbackSetupContext
CallbackStepContext = MetricsCallbackStepContext
CallbackValidationContext = MetricsCallbackValidationContext


class Callback(MetricsCallback):
    """Lightweight callback interface used by FL experiments."""

    def setup(self, context: CallbackSetupContext) -> None:  # pragma: no cover - interface
        """Called once when model parts and optimizers are available."""

    def on_step_end(self, context: CallbackStepContext) -> None:  # pragma: no cover - interface
        """Called after metrics logging for every training step."""

    def on_validation_end(self, context: CallbackValidationContext) -> None:  # pragma: no cover - interface
        """Called when validation metrics are logged."""

    def close(self) -> None:  # pragma: no cover - interface
        """Called when training finishes."""
