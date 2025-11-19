# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration coverage for TorchFT DES-LOC configuration hooks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _DummyWork:
    def wait(self) -> None:  # pragma: no cover - simple synchronization stub
        return None


class _DummyManager:
    def __init__(self) -> None:
        self._state_dict_registry: dict[str, tuple] = {}

    def disallow_state_dict_read(self) -> None:  # pragma: no cover - stub
        return None

    def allow_state_dict_read(self) -> None:  # pragma: no cover - stub
        return None

    def start_quorum(
        self, *, allow_heal: bool, shrink_only: bool, timeout
    ) -> None:  # pragma: no cover - stub
        return None

    def report_error(self, err: Exception) -> None:  # pragma: no cover - stub
        raise err

    def should_commit(self) -> bool:
        return True

    def allreduce(self, _tensor: torch.Tensor) -> _DummyWork:
        return _DummyWork()

    def register_state_dict_fn(self, key: str, load_fn, save_fn) -> None:
        self._state_dict_registry[key] = (load_fn, save_fn)

    def current_step(self) -> int:
        return 0


@dataclass(frozen=True)
class _TestOuterOptimizerConfig:
    target: str
    kwargs: dict[str, float]

    def resolve_optimizer_cls(self):
        return getattr(torch.optim, self.target)


@dataclass
class _TestStreamingConfig:
    enabled: bool = False
    fragments: int = 1
    sync_delay: int = 0
    update_alpha: float = 0.0
    use_bucketization: bool = False
    bucket_cap_mb: float | None = None
    should_quantize: bool = False


class _TestDeslocConfig:
    def __init__(
        self,
        *,
        enabled: bool = True,
        param_sync_every: int = 1,
        optimizer_sync_every=None,
        backup_device="cpu",
        pin_memory: bool = True,
        quorum_timeout_seconds: int = 60,
        outer_optimizer: _TestOuterOptimizerConfig | None = None,
        checkpoint_outer_optimizer: bool = True,
        streaming: _TestStreamingConfig | dict[str, Any] | None = None,
        log_outer_metrics: bool = False,
    ) -> None:
        self.enabled = enabled
        self.param_sync_every = param_sync_every
        self.optimizer_sync_every = optimizer_sync_every
        self.backup_device = backup_device
        self.pin_memory = pin_memory
        self.quorum_timeout_seconds = quorum_timeout_seconds
        self.outer_optimizer = outer_optimizer
        self.checkpoint_outer_optimizer = checkpoint_outer_optimizer
        self.streaming = (
            _TestStreamingConfig(**streaming) if isinstance(streaming, dict) else streaming
        )
        self.log_outer_metrics = log_outer_metrics

    def resolved_backup_device(self) -> torch.device | None:
        return None if self.backup_device is None else torch.device(self.backup_device)

    def normalized_optimizer_sync(self):
        return self.optimizer_sync_every

    def normalized_outer_optimizer(self):
        return self.outer_optimizer

    def resolved_streaming(self):
        cfg = self.streaming
        if cfg is None or not cfg.enabled:
            return None
        return cfg


def _build_job_config(**overrides):
    desloc = overrides.get("desloc", _TestDeslocConfig())
    ft = SimpleNamespace(enable=True, semi_sync_method=None)
    ft_override = overrides.get("fault_tolerance", ft)
    optimizer = SimpleNamespace(desloc=desloc)
    optimizer_override = overrides.get("optimizer", optimizer)
    return SimpleNamespace(optimizer=optimizer_override, fault_tolerance=ft_override)


stub_optimizers = ModuleType("torchtitan.experiments.fl.configs.optimizers")
stub_optimizers.DesLocConfig = _TestDeslocConfig
stub_optimizers.DesLocOuterOptimizerConfig = _TestOuterOptimizerConfig
stub_optimizers.DesLocStreamingConfig = _TestStreamingConfig
sys.modules.setdefault("torchtitan.experiments.fl.configs.optimizers", stub_optimizers)

_DESLOC_SPEC = importlib_util.spec_from_file_location(
    "torchtitan.experiments.fl.desloc",
    _REPO_ROOT / "torchtitan" / "experiments" / "fl" / "desloc.py",
)
desloc_module = importlib_util.module_from_spec(_DESLOC_SPEC)
assert _DESLOC_SPEC.loader is not None
_DESLOC_SPEC.loader.exec_module(desloc_module)
sys.modules.setdefault("torchtitan.experiments.fl.desloc", desloc_module)

dummy_pkg = ModuleType("torchtitan.experiments.fl")
dummy_pkg.__path__ = [str((_REPO_ROOT / "torchtitan" / "experiments" / "fl").resolve())]
dummy_pkg.desloc = desloc_module
sys.modules.setdefault("torchtitan.experiments.fl", dummy_pkg)

_FT_OVERRIDE_SPEC = importlib_util.spec_from_file_location(
    "torchtitan.experiments.fl.ft_override",
    _REPO_ROOT / "torchtitan" / "experiments" / "fl" / "ft_override.py",
)
ft_override = importlib_util.module_from_spec(_FT_OVERRIDE_SPEC)
assert _FT_OVERRIDE_SPEC.loader is not None
_FT_OVERRIDE_SPEC.loader.exec_module(ft_override)
configure_desloc = ft_override.configure_desloc


def test_configure_desloc_installs_desloc_support(monkeypatch):
    monkeypatch.setattr("torchtitan.components.ft.has_torchft", True, raising=False)
    monkeypatch.setattr(ft_override, "has_torchft", True, raising=False)
    monkeypatch.setattr(
        "torchtitan.components.optimizer.has_torchft", True, raising=False
    )

    class _DummyFTOptimizer:
        def __init__(self, _manager, _container) -> None:  # pragma: no cover - stub
            return None

        def step(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            return None

        def zero_grad(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            return None

    monkeypatch.setattr(
        "torchtitan.components.optimizer.ft",
        SimpleNamespace(Optimizer=_DummyFTOptimizer),
        raising=False,
    )

    job_config = _build_job_config()
    model = nn.Linear(2, 2)

    with configure_desloc(job_config):
        assert job_config.fault_tolerance.semi_sync_method == "desloc"
        container = desloc_module.DesLocFTOptimizersContainer(
            desloc_module.DesLocFTOptimizersConfig(
                model_parts=[model],
                optimizer_cls=optim.SGD,
                optimizer_kwargs={"lr": 0.1},
                ft_manager=_DummyManager(),
                desloc_config=job_config.optimizer.desloc,
            )
        )
        assert container._desloc_controllers

        with desloc_module.desloc_semi_sync_context(_DummyManager(), container):
            pass

        assert container._desloc_controllers == []


def test_configure_desloc_requires_torchft(monkeypatch):
    monkeypatch.setattr("torchtitan.components.ft.has_torchft", False, raising=False)
    monkeypatch.setattr(ft_override, "has_torchft", False, raising=False)
    job_config = _build_job_config()

    with pytest.raises(RuntimeError, match="requires the torchft package"):
        with configure_desloc(job_config):
            pass


def test_configure_desloc_conflicting_method(monkeypatch):
    monkeypatch.setattr("torchtitan.components.ft.has_torchft", True, raising=False)
    monkeypatch.setattr(ft_override, "has_torchft", True, raising=False)
    fault_tolerance = SimpleNamespace(enable=True, semi_sync_method="diloco")
    job_config = _build_job_config(fault_tolerance=fault_tolerance)

    with pytest.raises(ValueError, match="requires fault_tolerance.semi_sync_method"):
        with configure_desloc(job_config):
            pass


def test_desloc_outer_optimizer_applies_pseudogradients(monkeypatch):
    monkeypatch.setattr(
        "torchtitan.components.optimizer.has_torchft", True, raising=False
    )

    class _DummyFTOptimizer:
        def __init__(self, _manager, _container) -> None:  # pragma: no cover - stub
            return None

        def step(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            return None

        def zero_grad(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            return None

    monkeypatch.setattr(
        "torchtitan.components.optimizer.ft",
        SimpleNamespace(Optimizer=_DummyFTOptimizer),
        raising=False,
    )

    outer_spec = _TestOuterOptimizerConfig(target="SGD", kwargs={"lr": 0.5})
    desloc_cfg = _TestDeslocConfig(outer_optimizer=outer_spec)
    dummy_manager = _DummyManager()
    model = nn.Linear(1, 1, bias=False)

    container = desloc_module.DesLocFTOptimizersContainer(
        desloc_module.DesLocFTOptimizersConfig(
            model_parts=[model],
            optimizer_cls=optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            ft_manager=dummy_manager,
            desloc_config=desloc_cfg,
            outer_optimizer=outer_spec,
        )
    )

    controller = container._desloc_controllers[0]
    fragment = controller._param_fragment
    assert isinstance(fragment, desloc_module._OuterOptimizingParameterFragment)
    assert "desloc_0_outer_optimizer" in dummy_manager._state_dict_registry

    param = next(model.parameters())
    original = param.detach().clone()
    param.data.add_(1.0)
    local_value = param.detach().clone()

    works = fragment.prepare_sync()
    for work in works:
        work.wait()
    fragment.perform_sync()
    fragment.save_state()

    expected = original + 0.5 * (local_value - original)
    assert torch.allclose(param.data, expected)
    assert param.grad is None


def test_streaming_desloc_uses_streaming_controller():
    streaming_cfg = _TestStreamingConfig(enabled=True, fragments=2)
    desloc_cfg = _TestDeslocConfig(param_sync_every=4, streaming=streaming_cfg)
    dummy_manager = _DummyManager()
    model = nn.Linear(2, 2, bias=False)
    outer_optimizer = optim.SGD(model.parameters(), lr=0.2)

    container = desloc_module.DesLocFTOptimizersContainer(
        desloc_module.DesLocFTOptimizersConfig(
            model_parts=[model],
            optimizer_cls=optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            ft_manager=dummy_manager,
            desloc_config=desloc_cfg,
            outer_optimizer=outer_optimizer,
        )
    )

    controller = container._desloc_controllers[0]
    assert isinstance(controller, desloc_module.StreamingDesLocController)
    assert controller._fragments[0]._outer_optimizer is outer_optimizer
