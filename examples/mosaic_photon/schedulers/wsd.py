"""Warmup/stable/decay schedulers mirrored from Photon."""

from __future__ import annotations

from composer.core import State, Time
from composer.optim.scheduler import (
    ComposerScheduler,
    LinearScheduler,
    SqrtScheduler,
    _convert_time,
    _raise_if_cooldown_and_max_incompatible,
    _raise_if_cooldown_plus_warmup_and_max_incompatible,
    _raise_if_max_duration_exceeds_t_max,
    _raise_if_warmup_and_max_incompatible,
)
from llmfoundry.registry import schedulers

__all__ = [
    "ConstantWithLinearCooldownWithWarmupScheduler",
    "ConstantWithSqrtCooldownWithWarmupScheduler",
    "ConstantWithSqrtCooldownWithWarmupAndSwitchScheduler",
]


@schedulers.register_class("constant_with_linear_cooldown_with_warmup")
class ConstantWithLinearCooldownWithWarmupScheduler(ComposerScheduler):
    """Constant LR with optional warmup and linear cooldown."""

    def __init__(
        self,
        t_warmup: str | Time,
        t_cooldown: str | Time,
        t_max: str | Time = "1dur",
        *,
        scale_warmup: bool = False,
        scale_cooldown: bool = False,
    ) -> None:
        self.t_warmup = t_warmup
        self.t_cooldown = t_cooldown
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        self.scale_cooldown = scale_cooldown
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)
        self.cooldown_scheduler = LinearScheduler(alpha_i=1.0, alpha_f=0.0, t_max=t_cooldown)

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None
        t_warmup = _convert_time(self.t_warmup, state)
        t_cooldown = _convert_time(self.t_cooldown, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        _raise_if_warmup_and_max_incompatible(t_warmup, t_max)
        _raise_if_cooldown_and_max_incompatible(t_cooldown, t_max)
        _raise_if_cooldown_plus_warmup_and_max_incompatible(t_cooldown + t_warmup, t_max)
        _raise_if_max_duration_exceeds_t_max(t_max, state)

        if state.timestamp < t_warmup:
            return self.warmup_scheduler(state, ssr) if self.scale_warmup else self.warmup_scheduler(state)
        if state.timestamp >= t_max - t_cooldown:
            return (
                self.cooldown_scheduler(state, ssr)
                if self.scale_cooldown
                else self.cooldown_scheduler(state)
            )
        return 1.0


@schedulers.register_class("constant_with_sqrt_cooldown_with_warmup")
class ConstantWithSqrtCooldownWithWarmupScheduler(ComposerScheduler):
    """Constant LR with warmup and sqrt-style cooldown."""

    def __init__(
        self,
        t_warmup: str | Time,
        t_cooldown: str | Time,
        t_max: str | Time = "1dur",
        alpha_f: float = 1.0,
        *,
        scale_warmup: bool = False,
        scale_cooldown: bool = False,
    ) -> None:
        self.t_warmup = t_warmup
        self.t_cooldown = t_cooldown
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        self.scale_cooldown = scale_cooldown
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=alpha_f, t_max=t_warmup)
        self.cooldown_scheduler = SqrtScheduler(t_max=t_max, t_duration=t_cooldown)

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None
        t_warmup = _convert_time(self.t_warmup, state)
        t_cooldown = _convert_time(self.t_cooldown, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        _raise_if_warmup_and_max_incompatible(t_warmup, t_max)
        _raise_if_cooldown_and_max_incompatible(t_cooldown, t_max)
        _raise_if_cooldown_plus_warmup_and_max_incompatible(t_cooldown + t_warmup, t_max)
        _raise_if_max_duration_exceeds_t_max(t_max, state)

        if state.timestamp < t_warmup:
            return self.warmup_scheduler(state, ssr) if self.scale_warmup else self.warmup_scheduler(state)
        if state.timestamp >= t_max - t_cooldown:
            return (
                self.cooldown_scheduler(state, ssr)
                if self.scale_cooldown
                else self.cooldown_scheduler(state)
            )
        return 1.0


@schedulers.register_class("constant_with_sqrt_cooldown_with_warmup_and_switch")
class ConstantWithSqrtCooldownWithWarmupAndSwitchScheduler(ComposerScheduler):
    """Constant LR with warmup, sqrt cooldown, and multiplicative switch."""

    def __init__(
        self,
        t_warmup: str | Time,
        t_cooldown: str | Time,
        switch_scale: float = 1.0,
        t_max: str | Time = "1dur",
        *,
        scale_warmup: bool = False,
        scale_cooldown: bool = False,
    ) -> None:
        self.t_warmup = t_warmup
        self.t_cooldown = t_cooldown
        self.t_max = t_max
        self.scale_warmup = scale_warmup
        self.scale_cooldown = scale_cooldown
        self.switch_scale = switch_scale
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)
        self.cooldown_scheduler = SqrtScheduler(t_max=t_max, t_duration=t_cooldown)

    def __call__(self, state: State, ssr: float = 1.0) -> float:
        assert state.max_duration is not None
        t_warmup = _convert_time(self.t_warmup, state)
        t_cooldown = _convert_time(self.t_cooldown, state)
        t_max = _convert_time(self.t_max, state, ssr=ssr)

        _raise_if_warmup_and_max_incompatible(t_warmup, t_max)
        _raise_if_cooldown_and_max_incompatible(t_cooldown, t_max)
        _raise_if_cooldown_plus_warmup_and_max_incompatible(t_cooldown + t_warmup, t_max)
        _raise_if_max_duration_exceeds_t_max(t_max, state)

        if state.timestamp < t_warmup:
            return self.warmup_scheduler(state, ssr) if self.scale_warmup else self.warmup_scheduler(state)
        if state.timestamp >= t_max - t_cooldown:
            base = (
                self.cooldown_scheduler(state, ssr)
                if self.scale_cooldown
                else self.cooldown_scheduler(state)
            )
            return base * self.switch_scale
        return self.switch_scale
