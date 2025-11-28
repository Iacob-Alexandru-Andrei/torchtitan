# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the AggMoMuon optimizer."""

from __future__ import annotations

import torch
from torch import nn

import pytest

# Skip when the torch build does not expose DTensor APIs (used by torchtitan imports).
try:  # pragma: no cover - environment-dependent availability
    from torch.distributed.tensor import DeviceMesh as _DeviceMesh  # noqa: F401
except Exception:  # pragma: no cover - import guard
    pytest.skip("DeviceMesh is unavailable in this torch build", allow_module_level=True)

from torchtitan.experiments.fl.optimizers.aggmo_muon import AggMoMuon


def test_aggmo_muon_metrics_do_not_mutate_state() -> None:
    """report_per_parameter_metrics should leave optimizer state untouched."""
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = AggMoMuon(
        [param],
        lr=0.01,
        betas=(0.9,),
        vs=(0.7,),
        weight_decay=0.0,
    )

    param.grad = torch.ones_like(param)
    optimizer.step()  # Initialize exp_avg

    state = optimizer.state[param]
    exp_avg_before = state["exp_avg"].clone()

    param.grad = torch.full_like(param, 2.0)
    metrics = optimizer.report_per_parameter_metrics(param, "param", {})

    torch.testing.assert_close(state["exp_avg"], exp_avg_before)
    assert "l2_norm/exp_avg/AggMoMuon/param" in metrics
