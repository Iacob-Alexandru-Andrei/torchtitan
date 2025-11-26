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
from torchtitan.experiments.fl.configs.optimizers import (
    CompositeOptimizerSpec,
    MosaicOptimizerConfig,
)
from torchtitan.experiments.fl.optimizer_builder import (
    build_mosaic_optimizers,
    CompositeOptimizersContainer,
)
from torchtitan.experiments.fl.optimizers import Muon


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


class _MuPModule(nn.Module):
    """Stub MuP-aware module exposing grouping/scaling helpers."""

    def __init__(self) -> None:
        super().__init__()
        self.decay = nn.Parameter(torch.ones(2, 2))
        self.other = nn.Parameter(torch.ones(2, 2))
        self.mup_config = type(
            "MuPConfig",
            (),
            {
                "mup_enabled": True,
                "mup_disable_hidden_lr_scaling": False,
            },
        )()

    def _iter_trainable_params(self) -> list[tuple[str, nn.Parameter]]:
        return [("decay", self.decay), ("other", self.other)]

    def _bucketize_parameters(
        self,
        _param_entries: list[tuple[str, nn.Parameter]],
    ) -> dict[str, list[nn.Parameter]]:
        return {
            "emb": [],
            "unembed": [],
            "hidden_ln": [],
            "decay_lr": [self.decay],
            "hidden_bias": [],
            "no_decay": [self.other],
        }

    def _validate_bucket_counts(
        self,
        total_params: int,
        buckets: dict[str, list[nn.Parameter]],
    ) -> None:
        bucket_total = sum(len(bucket) for bucket in buckets.values())
        assert bucket_total == total_params

    def _compute_lr_scaling(self) -> tuple[float, float]:
        return 2.0, 1.0

    def _resolve_optimizer_eps(self, eps: float, *, width_lr_scaling: float) -> float:
        return eps * width_lr_scaling


def test_composite_respects_spec_eps_overrides_with_mup_scaling() -> None:
    """Composite optimizer eps overrides should survive MuP epsilon scaling."""
    module = _MuPModule()
    config = MosaicOptimizerConfig(
        name="DecoupledAdamW",
        lr=0.01,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="for-loop",
        composite=[
            CompositeOptimizerSpec(
                name="Muon",
                labels=("decay_lr",),
                config_overrides={"eps": 1e-7},
            ),
            CompositeOptimizerSpec(name="DecoupledAdamW", default=True),
        ],
    )

    container = build_mosaic_optimizers([module], config, _dims())
    assert isinstance(container, CompositeOptimizersContainer)

    muon_opt = next(opt for opt in container if isinstance(opt, Muon))
    adamw_opt = next(opt for opt in container if not isinstance(opt, Muon))

    # MuP scales eps by a factor of 2.0 via _compute_lr_scaling; overrides should still apply.
    assert muon_opt.param_groups[0]["eps"] == pytest.approx(2e-7)
    assert adamw_opt.param_groups[0]["eps"] == pytest.approx(2e-8)
