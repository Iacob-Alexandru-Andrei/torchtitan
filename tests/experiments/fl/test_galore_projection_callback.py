# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch import nn

import pytest

pytest.importorskip("torchmetrics")

from torchtitan.experiments.fl.callbacks import CallbackStepContext
from torchtitan.experiments.fl.metrics import (
    GaLoreMomentumProjectionCallback,
    GaLoreMomentumProjectionParams,
)
from torchtitan.experiments.fl.optimizers.galore import GaLore, RIGHT_PROJ, STD_PROJ, _orthogonal_matrix


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, float]]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.calls.append((step, metrics))


def _build_context(optimizer: GaLore, step: int) -> CallbackStepContext:
    return CallbackStepContext(
        step=step,
        model_parts=[],
        optimizers=[optimizer],
        logger=_RecordingLogger(),
        mesh=None,
    )


def test_galore_projection_projects_columns_and_updates_rank() -> None:
    param = nn.Parameter(torch.arange(6.0).reshape(3, 2))
    optimizer = GaLore([param], lr=0.1, rank=None)
    state = optimizer.state[param]
    state["exp_avg"] = torch.arange(6.0).reshape(3, 2)
    state["exp_avg_sq"] = torch.ones_like(state["exp_avg"]) * 5

    params = GaLoreMomentumProjectionParams(
        enabled=True,
        steps=(1,),
        target_ranks=(1,),
        state_keys=("exp_avg", "exp_avg_sq"),
        transform="columns",
        proj_type=STD_PROJ,
        shared_source=None,
        column_count=1,
        random_seed=None,
        random_std=1.0,
        log_metrics=True,
    )
    callback = GaLoreMomentumProjectionCallback(params)
    context = _build_context(optimizer, step=1)

    callback.on_step_end(context)

    projected_exp_avg = optimizer.state[param]["exp_avg"]
    projected_exp_avg_sq = optimizer.state[param]["exp_avg_sq"]

    assert optimizer.param_groups[0]["rank"] == 1
    assert projected_exp_avg.shape == (3, 1)
    assert projected_exp_avg_sq.shape == (3, 1)
    assert torch.allclose(projected_exp_avg.squeeze(-1), torch.tensor([0.0, 2.0, 4.0]))
    assert torch.allclose(
        projected_exp_avg_sq.squeeze(-1),
        torch.full((3,), 5.0),
    )
    assert context.logger.calls and "galore_projection/rank" in context.logger.calls[0][1]


def test_galore_projection_reuses_shared_basis() -> None:
    param = nn.Parameter(torch.tensor([[3.0, 0.0], [0.0, 1.0]]))
    optimizer = GaLore([param], lr=0.01, rank=None)
    state = optimizer.state[param]
    state["exp_avg"] = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    state["exp_avg_sq"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    params = GaLoreMomentumProjectionParams(
        enabled=True,
        steps=(2,),
        target_ranks=(1,),
        state_keys=("exp_avg", "exp_avg_sq"),
        transform="svd",
        proj_type=STD_PROJ,
        shared_source="exp_avg",
        column_count=None,
        random_seed=None,
        random_std=1.0,
        log_metrics=False,
    )
    callback = GaLoreMomentumProjectionCallback(params)
    context = _build_context(optimizer, step=2)

    expected_basis = _orthogonal_matrix(state["exp_avg"], rank=1, proj_type=RIGHT_PROJ)
    expected_exp_avg = state["exp_avg"] @ expected_basis.T
    expected_exp_avg_sq = state["exp_avg_sq"] @ expected_basis.T

    callback.on_step_end(context)

    projected_exp_avg = optimizer.state[param]["exp_avg"]
    projected_exp_avg_sq = optimizer.state[param]["exp_avg_sq"]

    assert projected_exp_avg.shape == (2, 1)
    assert projected_exp_avg_sq.shape == (2, 1)
    assert torch.allclose(projected_exp_avg, expected_exp_avg)
    assert torch.allclose(projected_exp_avg_sq, expected_exp_avg_sq)
