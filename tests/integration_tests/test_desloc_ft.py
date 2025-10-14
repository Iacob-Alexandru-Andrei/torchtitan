# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration coverage for TorchFT DES-LOC configuration hooks."""

from __future__ import annotations

import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

    def start_quorum(self, *, allow_heal: bool, shrink_only: bool, timeout) -> None:  # pragma: no cover - stub
        return None

    def report_error(self, err: Exception) -> None:  # pragma: no cover - stub
        raise err

    def should_commit(self) -> bool:
        return True

    def allreduce(self, _tensor: torch.Tensor) -> _DummyWork:
        return _DummyWork()

    def register_state_dict_fn(self, key: str, load_fn, save_fn) -> None:
        self._state_dict_registry[key] = (load_fn, save_fn)


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
    ) -> None:
        self.enabled = enabled
        self.param_sync_every = param_sync_every
        self.optimizer_sync_every = optimizer_sync_every
        self.backup_device = backup_device
        self.pin_memory = pin_memory
        self.quorum_timeout_seconds = quorum_timeout_seconds

    def resolved_backup_device(self) -> torch.device | None:
        return None if self.backup_device is None else torch.device(self.backup_device)

    def normalized_optimizer_sync(self):
        return self.optimizer_sync_every


def _build_job_config(**overrides):
    desloc = overrides.get("desloc", _TestDeslocConfig())
    ft = SimpleNamespace(enable=True, semi_sync_method=None)
    ft_override = overrides.get("fault_tolerance", ft)
    optimizer = SimpleNamespace(desloc=desloc)
    optimizer_override = overrides.get("optimizer", optimizer)
    return SimpleNamespace(optimizer=optimizer_override, fault_tolerance=ft_override)


stub_optimizers = ModuleType("torchtitan.experiments.fl.configs.optimizers")
stub_optimizers.DesLocConfig = _TestDeslocConfig
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
    monkeypatch.setattr("torchtitan.components.optimizer.has_torchft", True, raising=False)

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
            [model],
            optim.SGD,
            {"lr": 0.1},
            _DummyManager(),
            job_config.optimizer.desloc,
        )
        assert getattr(container, "_desloc_controllers")

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
