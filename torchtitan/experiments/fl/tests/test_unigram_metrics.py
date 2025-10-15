"""Tests for unigram metric plumbing in FL experiments."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch

from torchtitan.components import metrics as components_metrics
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims

ROOT = Path(__file__).resolve().parents[4]
METRICS_PATH = ROOT / "torchtitan" / "experiments" / "fl" / "metrics.py"
EXPECTED_TOKEN_COUNT = 3
LOG_REL_TOL = 1e-5


class _StubMetric:
    """TorchMetrics stand-in that records states for inspection."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._states: dict[str, torch.Tensor] = {}

    def add_state(
        self,
        name: str,
        default: torch.Tensor,
        dist_reduce_fx: str | None = None,
    ) -> None:
        del dist_reduce_fx
        self._states[name] = default.clone()
        setattr(self, name, self._states[name])

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)


_torchmetrics_stub = types.ModuleType("torchmetrics")
_torchmetrics_stub.Metric = _StubMetric
sys.modules.setdefault("torchmetrics", _torchmetrics_stub)

_callbacks_stub = types.ModuleType("torchtitan.experiments.fl.callbacks")


class _StubCallback:  # pragma: no cover - simple stub
    """Placeholder callback used to satisfy imports."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None


class _StubContext:  # pragma: no cover - simple stub
    """Placeholder context object used by callback constructors."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None


_callbacks_stub.Callback = _StubCallback
_callbacks_stub.CallbackSetupContext = _StubContext
_callbacks_stub.CallbackStepContext = _StubContext
_callbacks_stub.CallbackValidationContext = _StubContext
sys.modules.setdefault("torchtitan.experiments", types.ModuleType("torchtitan.experiments"))
sys.modules.setdefault("torchtitan.experiments.fl", types.ModuleType("torchtitan.experiments.fl"))
sys.modules["torchtitan.experiments.fl.callbacks"] = _callbacks_stub

_metrics_spec = importlib.util.spec_from_file_location(
    "torchtitan_experiments_fl_metrics", METRICS_PATH
)
if _metrics_spec is None or _metrics_spec.loader is None:  # pragma: no cover - sanity guard
    message = f"Unable to load metrics module from {METRICS_PATH}"
    raise RuntimeError(message)
_metrics_module = importlib.util.module_from_spec(_metrics_spec)
sys.modules[_metrics_spec.name] = _metrics_module
_metrics_spec.loader.exec_module(_metrics_module)
FLMetricsProcessor = _metrics_module.FLMetricsProcessor
LRMonitor = _metrics_module.LRMonitor
OptimizerMonitor = _metrics_module.OptimizerMonitor
PureUnigramCrossEntropy = _metrics_module.PureUnigramCrossEntropy
UnigramMetricManager = _metrics_module.UnigramMetricManager


class _DummyDeviceMemoryMonitor:
    """Device memory monitor stub that returns zeroed statistics."""

    device_name = "cpu"

    def get_peak_stats(self) -> components_metrics.DeviceMemStats:
        return components_metrics.DeviceMemStats(0, 0, 0, 0, 0, 0)

    def reset_peak_stats(self) -> None:
        return None


components_metrics.build_device_memory_monitor = lambda: _DummyDeviceMemoryMonitor()
components_metrics.device_module = types.SimpleNamespace(
    get_device_name=lambda _device: "cpu",
    get_device_properties=lambda _device: types.SimpleNamespace(total_memory=0),
    reset_peak_memory_stats=lambda: None,
    empty_cache=lambda: None,
    memory_stats=lambda _device: {},
)
components_metrics.device_type = "cpu"


@dataclass(slots=True)
class DummyOptimizerMonitorConfig:
    """Minimal optimizer monitor settings for the tests."""

    interval: int = 10
    only_global: bool = True
    log_metrics: bool = True
    gradient_accumulation_steps: int = 1


@dataclass(slots=True)
class DummyActivationMonitorConfig:
    """Activation monitor configuration used for FL metrics tests."""

    enabled: bool = False
    interval: int = 0
    ignore_module_types: tuple[str, ...] = ()
    gradient_accumulation_steps: int = 1
    enabled_metrics: set[str] | None = None


@dataclass(slots=True)
class DummyLRMonitorConfig:
    """Learning-rate monitor configuration for the tests."""

    enabled: bool = True
    interval: int = 1


@dataclass(slots=True)
class DummyBetasMonitorConfig:
    """Beta monitor configuration for validating callback wiring."""

    enabled: bool = False
    interval: int = 0


@dataclass(slots=True)
class DummyVSMonitorConfig:
    """Velocity schedule monitor configuration for the tests."""

    enabled: bool = False
    interval: int = 0


@dataclass(slots=True)
class DummyHyperparameterSwitchConfig:
    """Hyperparameter switch configuration for callback wiring tests."""

    enabled: bool = False
    steps: tuple[int, ...] = ()
    new_vs: tuple[float, ...] | None = None
    new_betas: tuple[float, ...] | None = None
    reset_momenta: tuple[str, ...] = ()
    log_metrics: bool = True


@dataclass(slots=True)
class MetricsConfig:
    """Aggregated configuration mirroring the production metrics payload."""

    optimizer_monitor: DummyOptimizerMonitorConfig = field(
        default_factory=DummyOptimizerMonitorConfig
    )
    activation_monitor: DummyActivationMonitorConfig = field(
        default_factory=DummyActivationMonitorConfig
    )
    lr_monitor: DummyLRMonitorConfig = field(default_factory=DummyLRMonitorConfig)
    betas_monitor: DummyBetasMonitorConfig = field(
        default_factory=DummyBetasMonitorConfig
    )
    vs_monitor: DummyVSMonitorConfig = field(default_factory=DummyVSMonitorConfig)
    hyperparameter_switch: DummyHyperparameterSwitchConfig = field(
        default_factory=DummyHyperparameterSwitchConfig
    )


def test_unigram_manager_aggregation_and_reset() -> None:
    """Registered unigram metrics should aggregate and reset correctly."""
    manager = UnigramMetricManager()
    metric = PureUnigramCrossEntropy(torch.tensor([0.5, 0.5]))
    with manager.register(metric, "train"):
        labels = torch.tensor([[0, 1, 1]])
        manager.update(labels)
        total_loss, total_items = manager.collect(reset=False)
        assert total_items == EXPECTED_TOKEN_COUNT
        assert math.isclose(
            total_loss, EXPECTED_TOKEN_COUNT * math.log(2), rel_tol=LOG_REL_TOL
        )
        manager.reset()
        cleared_loss, cleared_items = manager.collect(reset=False)
        assert cleared_loss == 0.0
        assert cleared_items == 0


def test_unigram_manager_teardown_removes_metric() -> None:
    """Closing the handle should remove the metric from the manager."""
    manager = UnigramMetricManager()
    metric = PureUnigramCrossEntropy(torch.tensor([0.6, 0.4]))
    handle = manager.register(metric, "val")
    handle.close()
    manager.update(torch.tensor([[0, 1]]))
    total_loss, total_items = manager.collect()
    assert total_loss == 0.0
    assert total_items == 0


def test_fl_metrics_processor_registers_expected_callbacks() -> None:
    """FL metrics processor should build the expected callback stack."""
    job_config = JobConfig()
    parallel_dims = ParallelDims(1, -1, 1, 1, 1, 1, 1, 1)
    metrics_config = MetricsConfig()
    manager = UnigramMetricManager()

    processor = FLMetricsProcessor(
        job_config,
        parallel_dims,
        metrics_config,
        unigram_manager=manager,
    )

    callback_types = {type(callback) for callback in processor.callbacks}
    assert OptimizerMonitor in callback_types
    assert LRMonitor in callback_types
    assert processor.activation_monitor is None
