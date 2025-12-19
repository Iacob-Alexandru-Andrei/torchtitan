# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for optimizer checkpoint state compatibility shims."""

from __future__ import annotations

import pytest

import torch
from torch import nn

from torchtitan.components.checkpoint import _OptimizerStateLoadShim
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.fl.optimizers.galore import GaLore


class _TinyModule(nn.Module):
    """Minimal module with a single parameter for optimizer state tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))


def test_state_shim_fills_missing_galore_param_group_keys() -> None:
    """Shim should default missing GaLore keys to the current run configuration."""
    module = _TinyModule()
    optimizer_kwargs = {
        "lr": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "vs": (0.6,),
        "rank": 1,
        "update_proj_gap": 1,
        "scale": 1.0,
        "proj_type": "std",
        "dim": 2,
        "rotate_moments_on_refresh": False,
        "use_error_feedback": False,
        "qhm_outside_projection": True,
    }
    container = OptimizersContainer([module], GaLore, optimizer_kwargs)
    shim = _OptimizerStateLoadShim(
        container,
        drop_param_group_keys=("vs", "qhm_outside_projection"),
    )
    shim.state_dict()

    checkpoint_state = container.state_dict()
    if isinstance(checkpoint_state.get("param_groups"), list):
        for group in checkpoint_state["param_groups"]:
            if isinstance(group, dict):
                group.pop("vs", None)
                group.pop("qhm_outside_projection", None)
    for key in list(checkpoint_state.keys()):
        if ".vs" in key or ".qhm_outside_projection" in key:
            checkpoint_state.pop(key)

    shim.load_state_dict(checkpoint_state)

    group = container.param_groups[0]
    assert tuple(group["vs"]) == (pytest.approx(0.6),)
    assert group["qhm_outside_projection"] is True
