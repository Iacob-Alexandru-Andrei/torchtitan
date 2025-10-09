# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.config import Optimizer as BaseOptimizer


@dataclass
class MosaicOptimizerConfig(BaseOptimizer):
    """Mosaic-specific optimizer config with additional hyperparameters."""

    v1: float = 0.0
    """v1 hyperparameter for quasi-hyperbolic optimizers"""

    decouple: bool = True
    """Whether to decouple the learning rate from the weight decay"""

    report_curvature: bool = False
    """Whether to report curvature metrics"""
