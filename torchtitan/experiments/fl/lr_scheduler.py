# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Learning rate scheduling helpers tailored for FL MuP experiments."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchtitan.components.lr_scheduler import LRSchedulersContainer

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.experiments.fl.configs.config import FLLRSchedulerConfig


@dataclass(frozen=True)
class WarmupStableDecayParams:
    """Parameters controlling the warmup-stable-decay LR schedule."""

    warmup_steps: int
    stable_steps: int
    decay_steps: int
    lr_decay_type: str
    min_lr_factor: float


def _linear_warmup_stable_decay(
    current_step: int, *, params: WarmupStableDecayParams
) -> float:
    """Matches the default TorchTitan warmup-stable-decay schedule."""
    warmup_stable_steps = params.warmup_steps + params.stable_steps
    if current_step < params.warmup_steps:
        current_step += 1
        if params.warmup_steps == 0:
            return 1.0
        return float(current_step / params.warmup_steps)

    if current_step < warmup_stable_steps:
        return 1.0

    current_step += 1
    if params.decay_steps == 0:
        return 1.0

    progress = float(current_step - warmup_stable_steps) / params.decay_steps
    if params.lr_decay_type == "linear":
        decay_value = 1 - progress
    elif params.lr_decay_type == "sqrt":
        decay_value = 1 - math.sqrt(progress)
    elif params.lr_decay_type == "cosine":
        decay_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        msg = (
            f"Unknown decay type '{params.lr_decay_type}'. Expected linear, sqrt, or cosine."
        )
        raise ValueError(msg)

    return params.min_lr_factor + (1 - params.min_lr_factor) * decay_value


def build_fl_lr_schedulers(
    optimizers: OptimizersContainer,
    lr_scheduler_config: FLLRSchedulerConfig,
    training_steps: int,
) -> LRSchedulersContainer:
    """FL-specific LR scheduler builder with optional mid-training switch."""
    warmup_steps = int(lr_scheduler_config.warmup_steps)
    if warmup_steps > training_steps:
        logger.warning(
            "Warmup steps (%s) exceed total training steps (%s). Adjusting warmup steps to match training steps.",
            warmup_steps,
            training_steps,
        )
        warmup_steps = training_steps

    decay_ratio = lr_scheduler_config.decay_ratio
    if decay_ratio is not None:
        decay_steps = round(training_steps * decay_ratio)
        if warmup_steps + decay_steps > training_steps:
            logger.warning(
                (
                    "Warmup (%s) + decay (%s) steps exceed total training steps (%s). "
                    "Adjusting decay steps to %s."
                ),
                warmup_steps,
                decay_steps,
                training_steps,
                training_steps - warmup_steps,
            )
            decay_steps = training_steps - warmup_steps
    else:
        decay_steps = training_steps - warmup_steps

    stable_steps = training_steps + 1 - warmup_steps - decay_steps
    schedule_params = WarmupStableDecayParams(
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        lr_decay_type=lr_scheduler_config.decay_type,
        min_lr_factor=lr_scheduler_config.min_lr_factor,
    )
    base_lambda: Callable[[int], float] = functools.partial(
        _linear_warmup_stable_decay, params=schedule_params
    )

    switch_step = lr_scheduler_config.switch_step
    switch_scale = lr_scheduler_config.switch_scale if switch_step is not None else 1.0

    effective_switch_step: int | None
    if switch_step is None or math.isclose(switch_scale, 1.0):
        effective_switch_step = None
    elif switch_step <= 0:
        logger.warning(
            "Switch step (%s) must be positive. Ignoring the switch directive.",
            switch_step,
        )
        effective_switch_step = None
    else:
        zero_based = switch_step - 1
        if zero_based >= training_steps:
            logger.warning(
                "Switch step (%s) is beyond total training steps (%s). Ignoring switch.",
                switch_step,
                training_steps,
            )
            effective_switch_step = None
        else:
            effective_switch_step = zero_based

    def lr_lambda(current_step: int) -> float:
        factor = base_lambda(current_step)
        if effective_switch_step is not None and current_step >= effective_switch_step:
            factor *= switch_scale
        return factor

    return LRSchedulersContainer(optimizers, lr_lambda)
