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
from torchtitan.experiments.fl.optimizers.galore import GaLore, _project as _project_galore
from torchtitan.experiments.fl.optimizers.galore_global import GaLoreGlobal, _project as _project_galore_global
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
    projected_grad = _project_galore(state, module.weight.grad, state["step"])

    assert state["exp_avg"].shape == state["exp_avg_sq"].shape
    assert state["exp_avg"].shape == projected_grad.shape
    assert state["exp_avg"].shape != module.weight.shape


def test_galore_state_dict_omits_full_rank_shape_metadata() -> None:
    """GaLore.state_dict() should drop optional projector shape metadata for compatibility."""
    module = _TinyModule()
    optimizer = GaLore(module.parameters(), lr=0.01, betas=(0.9, 0.95), rank=1, update_proj_gap=1)

    module.weight.grad = torch.ones_like(module.weight)
    optimizer.step()

    runtime_meta = optimizer.state[module.weight]["projector_meta"]
    assert "full_rank_shape" in runtime_meta
    optimizer.state[module.weight]["_bootstrap_projector"] = True

    serialized = optimizer.state_dict()
    for entry in serialized.get("state", {}).values():
        meta = entry.get("projector_meta") if isinstance(entry, dict) else None
        if meta:
            assert "full_rank_shape" not in meta
        if isinstance(entry, dict):
            assert "_bootstrap_projector" not in entry


def test_galore_global_low_rank_states_follow_projected_grad_shape() -> None:
    """GaLoreGlobal should preserve low-rank optimizer state shapes during initialization."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=1, update_proj_gap=4)

    param.grad = torch.ones_like(param)
    optimizer.step()

    state = optimizer.state[param]
    meta = state.get("projector_meta", {})
    initial_projector = meta.get("initial_projector")
    mode_code = meta.get("initial_projector_mode")
    if isinstance(mode_code, int):
        initial_projector = {0: "random", 1: "identity"}.get(mode_code)
    projected_grad = _project_galore_global(
        state,
        param.grad,
        initial_projector=initial_projector,
    )

    assert state["exp_avg"].shape == state["exp_avg_sq"].shape
    assert state["exp_avg"].shape == projected_grad.shape
    assert state["exp_avg"].shape != param.shape


def test_galore_global_identity_fallback_matches_eye() -> None:
    """GaLoreGlobal should synthesize an identity projector when requested."""
    param = nn.Parameter(torch.zeros(2, 2))
    optimizer = GaLoreGlobal(
        [param],
        lr=0.01,
        betas=(0.9, 0.95),
        rank=2,
        update_proj_gap=10,
        initial_projector="identity",
    )

    param.grad = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    optimizer.step()

    state = optimizer.state[param]
    basis = state.get("projector_basis")
    assert isinstance(basis, torch.Tensor)
    assert torch.allclose(basis, torch.eye(2, dtype=basis.dtype, device=basis.device))


def test_galore_global_identity_left_matches_expected_shape() -> None:
    """Identity projectors for LEFT projections should preserve row dimension."""
    tensor = torch.zeros(2, 4)
    basis = GaLoreGlobal._build_identity_projector(
        tensor,
        rank=2,
        resolved_proj_type="left",
        device=tensor.device,
        dtype=tensor.dtype,
    )

    assert isinstance(basis, torch.Tensor)
    assert basis.shape == (tensor.shape[0], 2)


def test_galore_global_random_fallback_builds_orthonormal_rows() -> None:
    """Random fallback projectors should contain orthonormal rows."""
    torch.manual_seed(0)
    param = nn.Parameter(torch.zeros(3, 2))
    optimizer = GaLoreGlobal(
        [param],
        lr=0.01,
        betas=(0.9, 0.95),
        rank=1,
        update_proj_gap=10,
        initial_projector="random",
    )

    param.grad = torch.randn_like(param)
    optimizer.step()

    state = optimizer.state[param]
    basis = state.get("projector_basis")
    assert isinstance(basis, torch.Tensor)
    gram = basis @ basis.T
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    assert torch.allclose(gram, eye, atol=1e-5, rtol=1e-4)


def test_galore_global_placeholder_cleanup_defers_until_finalized() -> None:
    """Placeholder projectors should remain active until explicitly finalized."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=1)

    optimizer.enable_placeholder_projectors()
    state = optimizer.state[param]
    assert "projector_basis" in state
    assert state.get("_placeholder_projector") is True

    optimizer.disable_placeholder_projectors()
    # Cleanup should be deferred, so the placeholder basis still exists.
    assert "projector_basis" in state
    assert state.get("_placeholder_projector") is True

    optimizer.finalize_placeholder_projectors()
    assert "projector_basis" not in state


def test_galore_global_finalize_preserves_real_projector() -> None:
    """Finalizing cleanup should not drop projector bases replaced by the server."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=1)
    optimizer.enable_placeholder_projectors()
    state = optimizer.state[param]

    # Simulate the server installing a real projector and clearing the placeholder flag.
    state["projector_basis"] = torch.eye(1)
    state.pop("_placeholder_projector", None)
    state.pop("_bootstrap_projector", None)

    optimizer.finalize_placeholder_projectors()
    assert "projector_basis" in state


def test_galore_global_bootstrap_identity_used_on_init() -> None:
    """_project should synthesize an identity projector during initialization."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=2)

    optimizer.enable_placeholder_projectors()
    grad = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    state = optimizer.state.setdefault(param, {})
    state.pop("projector_basis", None)
    state.pop("_placeholder_projector", None)
    state["step"] = torch.zeros((), dtype=torch.float32)

    projected = _project_galore_global(state, grad)
    assert projected.shape == grad.shape
    assert state.get("_bootstrap_projector") is True
    assert "projector_basis" in state


def test_galore_global_bootstrap_identity_handles_left_projection() -> None:
    """Bootstrap identity projectors should handle left-projection tensors."""
    param = nn.Parameter(torch.ones(2, 4))  # rows < cols -> LEFT projection
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=2)

    optimizer.enable_placeholder_projectors()
    grad = torch.randn_like(param)
    state = optimizer.state.setdefault(param, {})
    state.pop("projector_basis", None)
    state.pop("_placeholder_projector", None)
    state["step"] = torch.zeros((), dtype=torch.float32)

    projected = _project_galore_global(state, grad)
    assert projected.shape == grad.shape
    assert state.get("_bootstrap_projector") is True


def test_galore_global_requires_projector_after_init() -> None:
    """Once optimization has progressed past step 0, missing projectors should raise."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=2)

    optimizer.enable_placeholder_projectors()
    grad = torch.ones_like(param)
    state = optimizer.state.setdefault(param, {})
    state.pop("projector_basis", None)
    state.pop("_placeholder_projector", None)
    state["step"] = torch.ones((), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="requires projector bases"):
        _project_galore_global(state, grad)


def test_galore_global_state_dict_omits_full_rank_shape_metadata() -> None:
    """GaLoreGlobal.state_dict() should drop optional projector shape metadata for compatibility."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = GaLoreGlobal([param], lr=0.01, betas=(0.9, 0.95), rank=1)

    param.grad = torch.ones_like(param)
    optimizer.step()

    runtime_meta = optimizer.state[param]["projector_meta"]
    assert "full_rank_shape" in runtime_meta
    optimizer.state[param]["_bootstrap_projector"] = True

    serialized = optimizer.state_dict()
    for entry in serialized.get("state", {}).values():
        meta = entry.get("projector_meta") if isinstance(entry, dict) else None
        if meta:
            assert "full_rank_shape" not in meta
        if isinstance(entry, dict):
            assert "_bootstrap_projector" not in entry


def test_galore_global_rotate_momenta_handles_distinct_params() -> None:
    """rotate_momenta should locate param groups by identity, not value equality."""
    param_a = nn.Parameter(torch.ones(5, 3))
    param_b = nn.Parameter(torch.ones(3, 5))
    optimizer = GaLoreGlobal(
        [param_a, param_b],
        lr=0.01,
        betas=(0.9, 0.95),
        rank=1,
        update_proj_gap=1,
        rotate_moments_on_refresh=True,
    )

    for param in (param_a, param_b):
        param.grad = torch.randn_like(param)
    optimizer.step()

    state = optimizer.state[param_b]
    exp_avg_before = state["exp_avg"].clone()

    optimizer.rotate_momenta(
        param_b,
        old_basis=torch.eye(1),
        new_basis=torch.eye(1),
        proj_type="std",
    )

    assert torch.equal(state["exp_avg"], exp_avg_before)


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
    """Regex rank overrides should respect existing param groups while storing per-param ranks."""
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

    # Existing param groups should remain intact.
    assert len(optimizer.param_groups) == 1
    assert _group_params_ids(optimizer.param_groups[0]) == _group_params_ids(base_group)
    assert optimizer.param_groups[0]["rank"] == 10

    overrides = optimizer._param_rank_overrides  # type: ignore[attr-defined]

    attn_ids = {id(p) for p in module.attn.parameters()}
    ffn_ids = {id(p) for p in module.ffn.parameters()}
    other_ids = {id(p) for p in module.other.parameters()}

    for param_id in attn_ids:
        assert overrides[param_id] == 4
    for param_id in ffn_ids:
        assert overrides[param_id] == 6
    for param_id in other_ids:
        assert param_id not in overrides
