# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Mosaic optimizer builder selection logic."""

from __future__ import annotations

import pytest

import torch
from torch import nn

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.configs.optimizers import MosaicOptimizerConfig
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers


class _TinyModule(nn.Module):
    """Minimal module with a single parameter for optimizer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))


def _dims() -> ParallelDims:
    return ParallelDims(
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        world_size=1,
    )


def test_default_builder_uses_core_optimizer() -> None:
    """builder='default' should delegate to the core optimizer implementation."""
    module = _TinyModule()
    config = MosaicOptimizerConfig(
        name="AdamW",
        lr=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="for-loop",
        builder="Default",  # case-insensitive
    )

    container = build_mosaic_optimizers([module], config, _dims())
    assert isinstance(container, OptimizersContainer)

    optimizer = next(iter(container))
    assert optimizer.__class__ is torch.optim.AdamW


def test_default_builder_rejects_mosaic_only_optimizer() -> None:
    """builder='default' should reject Mosaic-only optimizers."""
    module = _TinyModule()
    config = MosaicOptimizerConfig(
        name="DecoupledAdamW",
        lr=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="for-loop",
        builder="default",
    )

    with pytest.raises(ValueError, match="requires .*'mosaic'"):
        build_mosaic_optimizers([module], config, _dims())
