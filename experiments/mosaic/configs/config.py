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
    """
    A dataclass for holding all configuration settings for a MosaicML training job.
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