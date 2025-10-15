"""Tests covering the Mosaic config manager integration."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from torchtitan.config import ConfigManager

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ORIGINAL_FL_MODULE = sys.modules.get("torchtitan.experiments.fl")
FL_STUB = types.ModuleType("torchtitan.experiments.fl")
FL_STUB.__path__ = [str(REPO_ROOT / "torchtitan" / "experiments" / "fl")]
sys.modules["torchtitan.experiments.fl"] = FL_STUB

CONFIG_SPEC = importlib.util.spec_from_file_location(
    "mosaic_config",
    REPO_ROOT / "torchtitan" / "experiments" / "fl" / "configs" / "config.py",
)
if CONFIG_SPEC is None or CONFIG_SPEC.loader is None:
    raise RuntimeError("Failed to load Mosaic config module spec")
config_module = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = config_module
CONFIG_SPEC.loader.exec_module(config_module)

MetricsConfig = config_module.MetricsConfig
MosaicDataLoaderConfig = config_module.MosaicDataLoaderConfig
MosaicJobConfig = config_module.MosaicJobConfig
MosaicTokenizerConfig = config_module.MosaicTokenizerConfig
S3CheckpointingConfig = config_module.S3CheckpointingConfig


_OVERRIDE_INTERVAL = 42
_EXPECTED_NUM_WORKERS = 3
_EXPECTED_OPT_INTERVAL = 7

def teardown_module() -> None:
    """Restore the original FL module registration after tests complete."""

    if ORIGINAL_FL_MODULE is not None:
        sys.modules["torchtitan.experiments.fl"] = ORIGINAL_FL_MODULE
    else:
        sys.modules.pop("torchtitan.experiments.fl", None)


def test_parse_args_produces_typed_dataclasses() -> None:
    """Parsing CLI args should return a config with typed nested sections."""

    manager = ConfigManager(MosaicJobConfig)
    config = manager.parse_args([])

    assert isinstance(config, MosaicJobConfig)
    assert isinstance(config.mosaic_dataloader, MosaicDataLoaderConfig)
    assert isinstance(config.mosaic_tokenizer, MosaicTokenizerConfig)
    assert isinstance(config.fl_metrics, MetricsConfig)
    assert isinstance(config.s3_checkpoint, S3CheckpointingConfig)


def test_cli_overrides_nested_metrics_field() -> None:
    """CLI overrides should land on the nested metrics dataclass."""

    manager = ConfigManager(MosaicJobConfig)
    config = manager.parse_args(
        [f"--fl_metrics.optimizer_monitor.interval={_OVERRIDE_INTERVAL}"]
    )

    assert config.fl_metrics.optimizer_monitor.interval == _OVERRIDE_INTERVAL


def test_toml_invalid_metrics_payload_rejected(tmp_path: Path) -> None:
    """TOML payloads with unknown metrics keys should raise a ValueError."""

    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[fl_metrics]\ninvalid = 1\n", encoding="utf-8")

    manager = ConfigManager(MosaicJobConfig)
    with pytest.raises(ValueError, match="unknown key"):
        manager.parse_args(["--job.config-file", str(config_path)])


def test_manual_init_coerces_nested_sections() -> None:
    """Direct MosaicJobConfig construction should coerce nested mappings."""

    job_config = MosaicJobConfig(
        mosaic_dataloader={"dataset": {}, "num_workers": _EXPECTED_NUM_WORKERS},
        mosaic_tokenizer={"name": "meta-llama/Llama-3.1-8B"},
        fl_metrics={
            "optimizer_monitor": {"interval": _EXPECTED_OPT_INTERVAL},
            "activation_monitor": {"enabled": True},
        },
        s3_checkpoint={"enable": True, "bucket": "bucket", "prefix": "prefix"},
    )

    assert isinstance(job_config.mosaic_dataloader, MosaicDataLoaderConfig)
    assert job_config.mosaic_dataloader.num_workers == _EXPECTED_NUM_WORKERS
    assert isinstance(job_config.mosaic_tokenizer, MosaicTokenizerConfig)
    assert job_config.mosaic_tokenizer.name == "meta-llama/Llama-3.1-8B"
    assert isinstance(job_config.fl_metrics, MetricsConfig)
    assert job_config.fl_metrics.optimizer_monitor.interval == _EXPECTED_OPT_INTERVAL
    assert job_config.fl_metrics.activation_monitor.enabled is True
    assert isinstance(job_config.s3_checkpoint, S3CheckpointingConfig)
    assert job_config.s3_checkpoint.enable is True


def test_manual_init_invalid_section_type_raises() -> None:
    """Invalid nested section payloads should raise a ``TypeError``."""

    with pytest.raises(TypeError):
        MosaicJobConfig(mosaic_tokenizer="bad-tokenizer")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        MosaicJobConfig(fl_metrics="not-a-mapping")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        MosaicJobConfig(s3_checkpoint=123)  # type: ignore[arg-type]
