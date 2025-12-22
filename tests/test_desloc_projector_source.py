# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DES-LOC GaLore projector source selection."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ORIGINAL_FL_MODULE = sys.modules.get("torchtitan.experiments.fl")
FL_STUB = types.ModuleType("torchtitan.experiments.fl")
FL_STUB.__path__ = [str(REPO_ROOT / "torchtitan" / "experiments" / "fl")]
sys.modules["torchtitan.experiments.fl"] = FL_STUB

ORIGINAL_OPT_MODULE = sys.modules.get("torchtitan.experiments.fl.optimizers")
OPT_STUB = types.ModuleType("torchtitan.experiments.fl.optimizers")
OPT_STUB.__path__ = [
    str(REPO_ROOT / "torchtitan" / "experiments" / "fl" / "optimizers")
]
sys.modules["torchtitan.experiments.fl.optimizers"] = OPT_STUB

GALORE_SPEC = importlib.util.spec_from_file_location(
    "torchtitan.experiments.fl.optimizers.galore_global",
    REPO_ROOT / "torchtitan" / "experiments" / "fl" / "optimizers" / "galore_global.py",
)
if GALORE_SPEC is None or GALORE_SPEC.loader is None:
    msg = "Failed to load GaLore module spec"
    raise RuntimeError(msg)
galore_module = importlib.util.module_from_spec(GALORE_SPEC)
sys.modules[GALORE_SPEC.name] = galore_module
GALORE_SPEC.loader.exec_module(galore_module)

DESLOC_SPEC = importlib.util.spec_from_file_location(
    "torchtitan.experiments.fl.desloc",
    REPO_ROOT / "torchtitan" / "experiments" / "fl" / "desloc.py",
)
if DESLOC_SPEC is None or DESLOC_SPEC.loader is None:
    msg = "Failed to load DES-LOC module spec"
    raise RuntimeError(msg)
desloc_module = importlib.util.module_from_spec(DESLOC_SPEC)
sys.modules[DESLOC_SPEC.name] = desloc_module
DESLOC_SPEC.loader.exec_module(desloc_module)

GaLoreGlobal = galore_module.GaLoreGlobal
STD_PROJ = galore_module.STD_PROJ
ParameterFragmentConfig = desloc_module.ParameterFragmentConfig
_ParameterFragment = desloc_module._ParameterFragment

SIMILARITY_THRESHOLD = 0.99


def teardown_module() -> None:
    """Restore any stubbed modules after test completion."""
    if ORIGINAL_FL_MODULE is not None:
        sys.modules["torchtitan.experiments.fl"] = ORIGINAL_FL_MODULE
    else:
        sys.modules.pop("torchtitan.experiments.fl", None)
    if ORIGINAL_OPT_MODULE is not None:
        sys.modules["torchtitan.experiments.fl.optimizers"] = ORIGINAL_OPT_MODULE
    else:
        sys.modules.pop("torchtitan.experiments.fl.optimizers", None)


def _build_fragment(source: str) -> tuple[_ParameterFragment, str, torch.Tensor]:
    model = torch.nn.Linear(2, 2, bias=False)
    name, param = next(model.named_parameters())
    optimizer = GaLoreGlobal([param], rank=1, update_proj_gap=1)
    state = optimizer.state[param]
    state["projector_meta"] = {
        "rank": 1,
        "proj_type": STD_PROJ,
        "resolved_proj_type": STD_PROJ,
    }

    config = ParameterFragmentConfig(
        manager=object(),
        model=model,
        sync_every=1,
        backup_device=None,
        pin_memory=False,
        name_prefix="test",
        outer_optimizer=None,
        local_optimizer=optimizer,
        log_outer_metrics=False,
        metrics_logger=None,
        checkpoint_outer_optimizer=False,
        low_rank_server_update=True,
        outer_optimizer_low_rank=False,
        low_rank_projector_error_feedback=False,
        low_rank_projector_source=source,
    )
    fragment = _ParameterFragment(config)
    return fragment, name, param


def _basis_similarity(basis: torch.Tensor, expected: torch.Tensor) -> float:
    return torch.abs(torch.sum(basis * expected)).item()


def test_desloc_projector_uses_full_rank_gradients() -> None:
    """Full-rank gradients should drive the projector refresh when configured."""
    fragment, name, param = _build_fragment("full_rank_grad")

    avg_param = torch.ones(2, 2)
    local_snapshot = torch.zeros(2, 2)
    full_rank_grad = torch.tensor([[2.0, 0.0], [0.0, 1.0]])

    fragment._averaged_parameters = [(name, avg_param)]
    fragment._pre_sync_parameters = {name: local_snapshot}
    fragment._averaged_gradients = {name: full_rank_grad.clone()}

    fragment._update_low_rank_projectors()

    basis = fragment._local_optimizer.state[param]["projector_basis"]
    _, _, v_h = torch.linalg.svd(full_rank_grad, full_matrices=False)
    expected = v_h[:1]
    similarity = _basis_similarity(basis, expected)

    assert similarity > SIMILARITY_THRESHOLD


def test_desloc_projector_uses_pseudo_gradients_by_default() -> None:
    """Pseudo-gradients should remain the default projector refresh signal."""
    fragment, name, param = _build_fragment("pseudo_grad")

    avg_param = torch.ones(2, 2)
    local_snapshot = torch.zeros(2, 2)
    full_rank_grad = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    pseudo_grad = local_snapshot - avg_param

    fragment._averaged_parameters = [(name, avg_param)]
    fragment._pre_sync_parameters = {name: local_snapshot}
    fragment._averaged_gradients = {name: full_rank_grad.clone()}

    fragment._update_low_rank_projectors()

    basis = fragment._local_optimizer.state[param]["projector_basis"]
    _, _, v_h = torch.linalg.svd(pseudo_grad, full_matrices=False)
    expected = v_h[:1]
    similarity = _basis_similarity(basis, expected)

    assert similarity > SIMILARITY_THRESHOLD
