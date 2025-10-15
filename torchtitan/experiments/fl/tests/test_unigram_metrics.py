import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torchtitan.config.job_config import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.components import metrics as components_metrics

_torchmetrics_stub = types.ModuleType("torchmetrics")


class _StubMetric:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self._states: dict[str, torch.Tensor] = {}

    def add_state(
        self, name: str, default: torch.Tensor, dist_reduce_fx: str | None = None
    ) -> None:
        self._states[name] = default.clone()
        setattr(self, name, self._states[name])

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)


_torchmetrics_stub.Metric = _StubMetric
sys.modules.setdefault("torchmetrics", _torchmetrics_stub)

_callbacks_stub = types.ModuleType("torchtitan.experiments.fl.callbacks")


class _StubCallback:  # pragma: no cover - simple stub
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class _StubContext:  # pragma: no cover - simple stub
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


_callbacks_stub.Callback = _StubCallback
_callbacks_stub.CallbackSetupContext = _StubContext
_callbacks_stub.CallbackStepContext = _StubContext
_callbacks_stub.CallbackValidationContext = _StubContext
sys.modules.setdefault("torchtitan.experiments", types.ModuleType("torchtitan.experiments"))
sys.modules.setdefault("torchtitan.experiments.fl", types.ModuleType("torchtitan.experiments.fl"))
sys.modules["torchtitan.experiments.fl.callbacks"] = _callbacks_stub

METRICS_PATH = ROOT / "torchtitan" / "experiments" / "fl" / "metrics.py"
_metrics_spec = importlib.util.spec_from_file_location(
    "torchtitan_experiments_fl_metrics", METRICS_PATH
)
assert _metrics_spec and _metrics_spec.loader
_metrics_module = importlib.util.module_from_spec(_metrics_spec)
sys.modules[_metrics_spec.name] = _metrics_module
_metrics_spec.loader.exec_module(_metrics_module)
FLMetricsProcessor = _metrics_module.FLMetricsProcessor
LRMonitor = _metrics_module.LRMonitor
OptimizerMonitor = _metrics_module.OptimizerMonitor
PureUnigramCrossEntropy = _metrics_module.PureUnigramCrossEntropy
UnigramMetricManager = _metrics_module.UnigramMetricManager


class _DummyDeviceMemoryMonitor:
    device_name = "cpu"

    def get_peak_stats(self) -> components_metrics.DeviceMemStats:
        return components_metrics.DeviceMemStats(0, 0, 0, 0, 0, 0)

    def reset_peak_stats(self) -> None:
        pass


components_metrics.build_device_memory_monitor = lambda: _DummyDeviceMemoryMonitor()
components_metrics.device_module = types.SimpleNamespace(
    get_device_name=lambda device: "cpu",
    get_device_properties=lambda device: types.SimpleNamespace(total_memory=0),
    reset_peak_memory_stats=lambda: None,
    empty_cache=lambda: None,
    memory_stats=lambda device: {},
)
components_metrics.device_type = "cpu"


@dataclass
class DummyOptimizerMonitorConfig:
    interval: int = 10
    only_global: bool = True
    log_metrics: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class DummyActivationMonitorConfig:
    enabled: bool = False
    interval: int = 0
    ignore_module_types: tuple[str, ...] = ()
    gradient_accumulation_steps: int = 1
    enabled_metrics: set[str] | None = None


@dataclass
class DummyLRMonitorConfig:
    enabled: bool = True
    interval: int = 1


@dataclass
class DummyBetasMonitorConfig:
    enabled: bool = False
    interval: int = 0


@dataclass
class DummyVSMonitorConfig:
    enabled: bool = False
    interval: int = 0


@dataclass
class DummyHyperparameterSwitchConfig:
    enabled: bool = False
    steps: tuple[int, ...] = ()
    new_vs: tuple[float, ...] | None = None
    new_betas: tuple[float, ...] | None = None
    reset_momenta: tuple[str, ...] = ()
    log_metrics: bool = True


@dataclass
class MetricsConfig:
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
    manager = UnigramMetricManager()
    metric = PureUnigramCrossEntropy(torch.tensor([0.5, 0.5]))
    with manager.register(metric, "train"):
        labels = torch.tensor([[0, 1, 1]])
        manager.update(labels)
        total_loss, total_items = manager.collect(reset=False)
        assert total_items == 3
        assert math.isclose(total_loss, 3 * math.log(2), rel_tol=1e-5)
        manager.reset()
        cleared_loss, cleared_items = manager.collect(reset=False)
        assert cleared_loss == 0.0
        assert cleared_items == 0


def test_unigram_manager_teardown_removes_metric() -> None:
    manager = UnigramMetricManager()
    metric = PureUnigramCrossEntropy(torch.tensor([0.6, 0.4]))
    handle = manager.register(metric, "val")
    handle.close()
    manager.update(torch.tensor([[0, 1]]))
    total_loss, total_items = manager.collect()
    assert total_loss == 0.0
    assert total_items == 0


def test_fl_metrics_processor_registers_expected_callbacks() -> None:
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
