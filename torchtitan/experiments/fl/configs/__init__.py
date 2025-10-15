# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration modules for FL experiments."""

from __future__ import annotations

import sys

from typing import cast

from torchtitan.config import ConfigManager

from .config import MosaicDataLoaderConfig, MosaicJobConfig, MosaicTokenizerConfig


class MosaicConfigManager(ConfigManager):
    """ConfigManager variant that post-processes FL-specific sections."""

    def __init__(self) -> None:
        super().__init__(MosaicJobConfig)

    def parse_args(self, args: list[str] = sys.argv[1:]) -> MosaicJobConfig:
        """Return a type-stable Mosaic job config parsed from CLI arguments."""

        return cast(MosaicJobConfig, super().parse_args(args))


def load_mosaic_job_config(args: list[str] | None = None) -> MosaicJobConfig:
    """Parse CLI/TOML arguments into a fully-typed :class:`MosaicJobConfig`."""
    manager = MosaicConfigManager()
    return manager.parse_args(args or sys.argv[1:])


__all__ = [
    "MosaicConfigManager",
    "MosaicDataLoaderConfig",
    "MosaicJobConfig",
    "MosaicTokenizerConfig",
    "load_mosaic_job_config",
]
