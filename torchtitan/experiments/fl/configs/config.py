# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration for MosaicML training jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchtitan.config import JobConfig
from torchtitan.experiments.fl.configs.optimizers import MosaicOptimizerConfig


@dataclass
class OptimizerMonitorConfig:
    """Configuration for optimizer monitoring."""

    interval: int = field(
        default=10,
        metadata={
            "help": "Interval (in steps) for the optimizer monitor to log metrics. "
            "Set to 0 to disable optimizer monitoring."
        },
    )

    only_global: bool = field(
        default=True,
        metadata={
            "help": "If True, only log global aggregated metrics. If False, log per-parameter metrics as well."
        },
    )

    log_metrics: bool = field(
        default=True,
        metadata={
            "help": "If True, log detailed optimizer metrics (moments, updates, etc.). "
            "If False, only log gradient norms."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps."},
    )

    enabled_metrics: set[str] | None = field(
        default=None,
        metadata={"help": "Set of enabled metrics. If None, all metrics are enabled."},
    )


@dataclass
class ActivationMonitorConfig:
    """Configuration for activation monitoring."""

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable logging of full-model activation statistics."},
    )

    interval: int = field(
        default=25,
        metadata={
            "help": "Training step interval for activation monitoring. Set to 0 to disable."
        },
    )

    ignore_module_types: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "help": "Optional substrings of module qualified names to skip when collecting activations."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps."},
    )

    enabled_metrics: set[str] | None = field(
        default=None,
        metadata={"help": "Set of enabled metrics. If None, all metrics are enabled."},
    )


@dataclass
class MetricsConfig:
    """Configuration for all metrics and monitoring."""

    optimizer_monitor: OptimizerMonitorConfig = field(
        default_factory=OptimizerMonitorConfig,
        metadata={"help": "Configuration for optimizer monitoring."},
    )

    activation_monitor: ActivationMonitorConfig = field(
        default_factory=ActivationMonitorConfig,
        metadata={"help": "Configuration for activation monitoring."},
    )


@dataclass
class MosaicJobConfig(JobConfig):
    """A dataclass for holding all configuration settings for a MosaicML training job.

    It inherits from the base `JobConfig` and adds Mosaic-specific sections.
    """

    # Override optimizer field to use MosaicOptimizerConfig
    optimizer: MosaicOptimizerConfig = field(  # type: ignore[assignment]
        default_factory=MosaicOptimizerConfig,
        metadata={
            "help": "Optimizer configuration with extended options for Mosaic optimizers (vs, betas)."
        },
    )

    # python-explicit-any
    mosaic_dataloader: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Configuration for the MosaicML streaming dataloader. "
            "Refer to llm-foundry for documentation on available options."
        },
    )
    # python-explicit-any
    mosaic_tokenizer: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Configuration for the MosaicML tokenizer. This should "
            "include the tokenizer name and any specific kwargs."
        },
    )

    fl_metrics: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={
            "help": "Configuration for FL-specific metrics and monitoring (optimizer and activation monitors)."
        },
    )
