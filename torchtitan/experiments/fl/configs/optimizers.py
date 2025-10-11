# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Custom optimizer hyperparameters for decoupled and quasi-hyperbolic optimizers."""

from dataclasses import dataclass

from torchtitan.config import Optimizer as BaseOptimizer

# Default values from BaseOptimizer
_MIN_BETAS_LENGTH = 2


@dataclass
class MosaicOptimizerConfig(BaseOptimizer):
    """Mosaic-specific optimizer config with additional hyperparameters."""

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

    def __post_init__(self) -> None:
        """Auto-initialize beta1 and beta2 from betas if betas is provided."""
        if self.betas is not None and len(self.betas) >= _MIN_BETAS_LENGTH:
            # If betas is provided, it always overrides beta1 and beta2
            # beta1 comes from the first element, beta2 from the last element
            self.beta1 = self.betas[0]
            self.beta2 = self.betas[-1]

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
