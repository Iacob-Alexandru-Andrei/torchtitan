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
from torchtitan.experiments.fl.optimizers.galore import GaLore
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers


class _TinyModule(nn.Module):
    """Minimal module with a single parameter for optimizer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))


class _ToyModel(nn.Module):
    """Simple module with named submodules for regex param group tests."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.ffn = nn.Linear(4, 4)
        self.other = nn.Linear(4, 4)


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


def test_qhscion_builder_exposes_betas_and_vs() -> None:
    """ScionQH param groups should expose betas and vs tuples like other QH optimizers."""
    module = _TinyModule()
    config = MosaicOptimizerConfig(
        name="ScionQH",
        lr=0.01,
        beta1=0.81,
        beta2=0.91,
        vs=(0.77,),
        eps=1e-8,
        weight_decay=0.0,
        implementation="for-loop",
    )

    container = build_mosaic_optimizers([module], config, _dims())
    optimizer = next(iter(container))
    group = optimizer.param_groups[0]

    assert group["betas"][0] == pytest.approx(0.81)
    assert group["betas"][-1] == pytest.approx(0.91)
    assert tuple(group["vs"]) == (pytest.approx(0.77),)
    assert group["v"] == pytest.approx(0.77)
    assert tuple(group["zeropower_coeffs"]) == pytest.approx((3.4445, -4.7750, 2.0315))


def test_qhscion_builder_prefers_scion_v_override() -> None:
    """Explicit optimizer.scion_v should override vector vs inputs."""
    module = _TinyModule()
    config = MosaicOptimizerConfig(
        name="ScionQH",
        lr=0.01,
        beta1=0.8,
        beta2=0.9,
        vs=(0.2, 0.3),
        scion_v=0.65,
        implementation="for-loop",
    )

    optimizer = next(iter(build_mosaic_optimizers([module], config, _dims())))
    assert tuple(optimizer.param_groups[0]["vs"]) == (pytest.approx(0.65),)


def test_scion_builder_accepts_custom_zeropower_coefficients() -> None:
    module = _TinyModule()
    config = MosaicOptimizerConfig(
        name="Scion",
        lr=0.01,
        beta1=0.9,
        beta2=0.95,
        implementation="for-loop",
        zeropower_coefficients=(1.0, 2.0, 3.0),
    )

    optimizer = next(iter(build_mosaic_optimizers([module], config, _dims())))
    coeffs = tuple(optimizer.param_groups[0]["zeropower_coeffs"])
    assert coeffs == pytest.approx((1.0, 2.0, 3.0))


def test_galore_low_rank_states_follow_projected_grad_shape() -> None:
    """GaLore moments should match the projected gradient shape, not parameter shape."""
    module = _TinyModule()
    optimizer = GaLore(module.parameters(), lr=0.01, betas=(0.9, 0.95), rank=1, update_proj_gap=1)

    module.weight.grad = torch.ones_like(module.weight)
    optimizer.step()

    state = optimizer.state[module.weight]
    projector = state["projector"]
    projected_grad = projector.project(module.weight.grad, state["step"])

    assert state["exp_avg"].shape == state["exp_avg_sq"].shape
    assert state["exp_avg"].shape == projected_grad.shape
    assert state["exp_avg"].shape != module.weight.shape


def test_galore_regex_param_groups_builds_expected_ranks() -> None:
    """Regex param groups should override GaLore rank per pattern with global defaults as fallback."""
    module = _ToyModel()
    config = MosaicOptimizerConfig(
        name="GaLore",
        lr=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        builder="mosaic",
        galore_rank=8,
        param_groups=[
            {"param_str_match": "attn", "rank": 4, "update_proj_gap": 5, "scale": 0.5, "proj_type": "left"},
            {"param_str_match": "ffn", "weight_decay": 0.0},
        ],
    )

    optimizer = next(iter(build_mosaic_optimizers([module], config, _dims())))

    def _group_params_ids(group: dict) -> set[int]:
        return {id(p) for p in group["params"]}

    attn_ids = {id(p) for p in module.attn.parameters()}
    ffn_ids = {id(p) for p in module.ffn.parameters()}
    other_ids = {id(p) for p in module.other.parameters()}

    attn_group = next(g for g in optimizer.param_groups if _group_params_ids(g) == attn_ids)
    ffn_group = next(g for g in optimizer.param_groups if _group_params_ids(g) == ffn_ids)
    other_group = next(g for g in optimizer.param_groups if _group_params_ids(g) == other_ids)

    assert attn_group["rank"] == 4
    assert attn_group["update_proj_gap"] == 5
    assert attn_group["scale"] == pytest.approx(0.5)
    assert attn_group["proj_type"] == "left"

    # ffn inherits the global rank while overriding weight decay.
    assert ffn_group["rank"] == 8
    assert ffn_group["weight_decay"] == pytest.approx(0.0)

    # Unmatched params fall back to the global defaults.
    assert other_group["rank"] == 8
    assert other_group["weight_decay"] == pytest.approx(0.1)


def test_galore_rank_regex_overrides_existing_param_groups() -> None:
    """Regex rank overrides should split existing param groups without redefining full groups."""
    module = _ToyModel()
    base_group = {
        "params": list(module.parameters()),
        "lr": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    config = MosaicOptimizerConfig(
        name="GaLore",
        lr=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        builder="mosaic",
        galore_rank=10,
        galore_param_regexes=[
            {"param_str_match": "attn", "rank": 4},
            {"param_str_match": "ffn", "rank": 6},
        ],
    )

    optimizer = next(iter(build_mosaic_optimizers([module], config, _dims(), param_groups=[base_group])))

    def _group_params_ids(group: dict) -> set[int]:
        return {id(p) for p in group["params"]}

    attn_ids = {id(p) for p in module.attn.parameters()}
    ffn_ids = {id(p) for p in module.ffn.parameters()}
    other_ids = {id(p) for p in module.other.parameters()}

    attn_group = next(g for g in optimizer.param_groups if _group_params_ids(g) == attn_ids)
    ffn_group = next(g for g in optimizer.param_groups if _group_params_ids(g) == ffn_ids)
    other_group = next(g for g in optimizer.param_groups if _group_params_ids(g) == other_ids)

    assert attn_group["rank"] == 4
    assert ffn_group["rank"] == 6
    assert other_group["rank"] == 10
