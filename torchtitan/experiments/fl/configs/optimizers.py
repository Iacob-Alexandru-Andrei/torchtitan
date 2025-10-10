# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Custom optimizer hyperparameters for decoupled and quasi-hyperbolic optimizers."""

from dataclasses import dataclass

from torchtitan.config import Optimizer as BaseOptimizer


@dataclass
class MosaicOptimizerConfig(BaseOptimizer):
    """Mosaic-specific optimizer config with additional hyperparameters."""

    vs: tuple[float, ...] = (0.7,)
    """vs hyperparameters for quasi-hyperbolic optimizers (each optimizer extracts as many as needed)"""

    decouple: bool = True
    """Whether to decouple the learning rate from the weight decay"""

    param_sync_every: int = 10
    """Frequency for synchronizing model parameters in DesLoc."""

    optimizer_sync_every: list[int] = (20, 20)
    """Frequencies for synchronizing optimizer states in DesLoc."""
