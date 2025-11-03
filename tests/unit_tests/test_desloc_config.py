from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _BaseOptimizer:
    name: str = "AdamW"
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    implementation: str = "fused"
    early_step_in_backward: bool = False


config_stub = types.ModuleType("torchtitan.config")
config_stub.Optimizer = _BaseOptimizer
sys.modules["torchtitan.config"] = config_stub

module_name = "tests._desloc_config_module"
spec = importlib.util.spec_from_file_location(
    module_name,
    _REPO_ROOT / "torchtitan" / "experiments" / "fl" / "configs" / "optimizers.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

DesLocConfig = module.DesLocConfig
DesLocOuterOptimizerConfig = module.DesLocOuterOptimizerConfig


def test_normalized_outer_optimizer_allows_empty_mapping():
    config = DesLocConfig(outer_optimizer={})
    assert config.normalized_outer_optimizer() is None


def test_normalized_outer_optimizer_requires_target_for_kwargs():
    config = DesLocConfig(outer_optimizer={"kwargs": {"lr": 0.1}})
    with pytest.raises(ValueError, match="requires a target optimizer"):
        config.normalized_outer_optimizer()


def test_normalized_outer_optimizer_dataclass_with_kwargs_requires_target():
    config = DesLocConfig(
        outer_optimizer=DesLocOuterOptimizerConfig(kwargs={"lr": 0.1})
    )
    with pytest.raises(ValueError, match="requires a target optimizer"):
        config.normalized_outer_optimizer()


def test_normalized_outer_optimizer_dataclass_returns_config():
    target = DesLocOuterOptimizerConfig(target="SGD")
    config = DesLocConfig(outer_optimizer=target)
    resolved = config.normalized_outer_optimizer()
    assert resolved is target


def test_desloc_outer_optimizer_config_resolve_requires_target():
    config = DesLocOuterOptimizerConfig()
    with pytest.raises(ValueError, match="target must be configured"):
        config.resolve_optimizer_cls()


def test_desloc_outer_optimizer_config_resolves_string_target():
    config = DesLocOuterOptimizerConfig(target="SGD")
    assert config.resolve_optimizer_cls() is optim.SGD
