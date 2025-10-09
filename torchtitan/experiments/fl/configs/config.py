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


@dataclass
class MosaicJobConfig(JobConfig):
    """A dataclass for holding all configuration settings for a MosaicML training job.

    It inherits from the base `JobConfig` and adds Mosaic-specific sections.
    """

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

    optimizer_monitor_interval: int = field(
        default=10,
        metadata={
            "help": "Interval (in steps) for the optimizer monitor to log metrics. "
            "Set to 0 to disable optimizer monitoring."
        },
    )

    optimizer_monitor_only_global: bool = field(
        default=True,
        metadata={
            "help": "If True, only log global aggregated metrics. If False, log per-parameter metrics as well."
        },
    )

    optimizer_monitor_log_metrics: bool = field(
        default=True,
        metadata={
            "help": "If True, log detailed optimizer metrics (moments, updates, etc.). "
            "If False, only log gradient norms."
        },
    )

    activation_monitor_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable logging of full-model activation statistics."
        },
    )

    activation_monitor_interval: int = field(
        default=25,
        metadata={
            "help": "Training step interval for activation monitoring. Set to 0 to disable."
        },
    )

    activation_monitor_ignore_module_types: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "help": "Optional substrings of module qualified names to skip when collecting activations."
        },
    )
