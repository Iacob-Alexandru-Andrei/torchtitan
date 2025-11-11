# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Scion optimizer variants."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchtitan.experiments.fl.optimizers.scion import QHScion, Scion


def _param() -> nn.Parameter:
    return nn.Parameter(torch.ones(2, 2))


def test_qhscion_param_group_exposes_vs_tuple() -> None:
    """`vs` and `betas` should be visible on the optimizer param group."""
    param = _param()
    optimizer = QHScion(
        [param],
        lr=0.01,
        betas=(0.8, 0.9),
        vs=(0.75,),
    )

    group = optimizer.param_groups[0]
    assert tuple(group["betas"]) == (pytest.approx(0.8), pytest.approx(0.9))
    assert tuple(group["vs"]) == (pytest.approx(0.75),)
    assert group["v"] == pytest.approx(0.75)


def test_qhscion_recovers_vs_from_legacy_v() -> None:
    """Optimizers saved without ``vs`` should still pick up ``v`` settings when stepping."""
    param = _param()
    optimizer = QHScion([param], lr=0.01, betas=(1.0,), vs=(0.5,))

    group = optimizer.param_groups[0]
    del group["vs"]
    group["v"] = 0.9

    param.grad = torch.ones_like(param)
    optimizer.step()

    assert tuple(group["vs"]) == (pytest.approx(0.9),)
    assert group["v"] == pytest.approx(0.9)


def test_scion_scale_only_group_applies_radius() -> None:
    """Groups without explicit norms should apply the configured Scion scale."""
    param = nn.Parameter(torch.zeros(1))
    optimizer = Scion(
        [
            {
                "params": [param],
                "lr": 0.1,
                "norm": None,
                "scale": 5.0,
            }
        ],
        lr=0.1,
    )

    param.grad = torch.ones_like(param)
    optimizer.step()

    assert torch.allclose(param.detach(), torch.full_like(param, -0.5), atol=1e-6)
    assert optimizer.param_groups[0]["norm_factor"] == "scale_only"


def test_scion_scale_applies_with_explicit_norm() -> None:
    param = nn.Parameter(torch.zeros(1))
    optimizer = Scion(
        [
            {
                "params": [param],
                "lr": 0.1,
                "norm": "spectral",
                "norm_kwargs": {"backend": "identity", "backend_steps": 0},
                "scale": 5.0,
            }
        ],
        lr=0.1,
    )
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert torch.allclose(param.detach(), torch.full_like(param, -0.5), atol=1e-6)


def test_scion_sign_norm_respects_normalized_flag() -> None:
    param = nn.Parameter(torch.zeros(1, 4))
    optimizer = Scion(
        [
            {
                "params": [param],
                "lr": 0.1,
                "norm": "sign",
                "norm_kwargs": {"normalized": True},
                "scale": 1.0,
            }
        ],
        lr=0.1,
    )

    param.grad = torch.ones_like(param)
    optimizer.step()

    expected = -0.1 * (1.0 / 4.0)
    assert torch.allclose(param.detach(), torch.full_like(param, expected), atol=1e-6)
