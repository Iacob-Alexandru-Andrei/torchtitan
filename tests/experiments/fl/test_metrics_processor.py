from __future__ import annotations

from dataclasses import replace
from importlib import util as importlib_util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torchmetrics")

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

_METRICS_SPEC = importlib_util.spec_from_file_location(
    "fl_metrics_module", ROOT / "torchtitan" / "experiments" / "fl" / "metrics.py"
)
fl_metrics = importlib_util.module_from_spec(_METRICS_SPEC)
assert _METRICS_SPEC is not None and _METRICS_SPEC.loader is not None
_METRICS_SPEC.loader.exec_module(fl_metrics)

_CONFIG_SPEC = importlib_util.spec_from_file_location(
    "fl_metrics_config_module",
    ROOT / "torchtitan" / "experiments" / "fl" / "configs" / "config.py",
)
config_module = importlib_util.module_from_spec(_CONFIG_SPEC)
assert _CONFIG_SPEC is not None and _CONFIG_SPEC.loader is not None
_CONFIG_SPEC.loader.exec_module(config_module)

ActivationMonitor = fl_metrics.ActivationMonitor
BetasMonitor = fl_metrics.BetasMonitor
FLMetricsProcessor = fl_metrics.FLMetricsProcessor
HyperparameterSwitchCallback = fl_metrics.HyperparameterSwitchCallback
LRMonitor = fl_metrics.LRMonitor
OptimizerMonitor = fl_metrics.OptimizerMonitor
VSMonitor = fl_metrics.VSMonitor

ActivationMonitorConfig = config_module.ActivationMonitorConfig
BetasMonitorConfig = config_module.BetasMonitorConfig
HyperparameterSwitchConfig = config_module.HyperparameterSwitchConfig
LRMonitorConfig = config_module.LRMonitorConfig
MetricsConfig = config_module.MetricsConfig
MosaicJobConfig = config_module.MosaicJobConfig
OptimizerMonitorConfig = config_module.OptimizerMonitorConfig
VSMonitorConfig = config_module.VSMonitorConfig


class _DummyParallelDims:
    dp_cp_enabled = False
    non_data_parallel_size = 1
    world_mesh = {"dp_cp": None}


def _build_processor(metrics_config: MetricsConfig | None = None) -> FLMetricsProcessor:
    job_config = MosaicJobConfig()
    if metrics_config is not None:
        job_config.fl_metrics = metrics_config
    return FLMetricsProcessor(job_config=job_config, parallel_dims=_DummyParallelDims())


def _callback_types(processor: FLMetricsProcessor) -> set[type]:
    return {type(callback) for callback in processor.callbacks}


def test_default_callbacks_include_optimizer_lr_and_activation() -> None:
    processor = _build_processor()
    types = _callback_types(processor)
    assert OptimizerMonitor in types
    assert LRMonitor in types
    assert ActivationMonitor in types
    assert BetasMonitor not in types
    assert VSMonitor not in types
    assert HyperparameterSwitchCallback not in types


def test_activation_monitor_disabled_with_zero_interval() -> None:
    metrics_cfg = replace(
        MosaicJobConfig().fl_metrics,
        activation_monitor=ActivationMonitorConfig(enabled=False, interval=0),
    )
    processor = _build_processor(metrics_cfg)
    assert ActivationMonitor not in _callback_types(processor)


def test_optimizer_monitor_respects_interval() -> None:
    metrics_cfg = replace(
        MosaicJobConfig().fl_metrics,
        optimizer_monitor=OptimizerMonitorConfig(interval=0),
    )
    processor = _build_processor(metrics_cfg)
    assert OptimizerMonitor not in _callback_types(processor)


def test_optional_optimizer_monitors_toggle() -> None:
    base_cfg = MosaicJobConfig().fl_metrics
    metrics_cfg = replace(
        base_cfg,
        betas_monitor=replace(
            base_cfg.betas_monitor, enabled=True, interval=4
        ),
        vs_monitor=replace(base_cfg.vs_monitor, enabled=True, interval=3),
    )
    processor = _build_processor(metrics_cfg)
    types = _callback_types(processor)
    assert BetasMonitor in types
    assert VSMonitor in types


@pytest.mark.parametrize(
    "config,expected",
    [
        (HyperparameterSwitchConfig(enabled=True, steps=()), False),
        (HyperparameterSwitchConfig(enabled=True, steps=(10,), new_betas=(0.9, 0.98)), True),
    ],
)
def test_hyperparameter_switch_requires_steps(
    config: HyperparameterSwitchConfig, expected: bool
) -> None:
    base_cfg = MosaicJobConfig().fl_metrics
    metrics_cfg = replace(base_cfg, hyperparameter_switch=config)
    processor = _build_processor(metrics_cfg)
    types = _callback_types(processor)
    assert (HyperparameterSwitchCallback in types) is expected


def test_lr_monitor_can_be_disabled() -> None:
    base_cfg = MosaicJobConfig().fl_metrics
    metrics_cfg = replace(
        base_cfg,
        lr_monitor=LRMonitorConfig(enabled=False, interval=0),
    )
    processor = _build_processor(metrics_cfg)
    assert LRMonitor not in _callback_types(processor)
