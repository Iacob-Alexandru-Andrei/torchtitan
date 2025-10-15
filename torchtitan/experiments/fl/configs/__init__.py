# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration modules for FL experiments."""

from __future__ import annotations

import sys

from torchtitan.config import ConfigManager

from .config import (
    ensure_mosaic_job_config_types,
    FLMetricsConfigEnvelope,
    MosaicDataLoaderConfig,
    MosaicJobConfig,
    MosaicTokenizerConfig,
)


class MosaicConfigManager(ConfigManager):
    """ConfigManager variant that post-processes FL-specific sections."""

    def __init__(self) -> None:
        super().__init__(MosaicJobConfig)

    def parse_args(self, args: list[str] = sys.argv[1:]) -> MosaicJobConfig:
        """Return a type-stable Mosaic job config parsed from CLI arguments.

        Args:
            args: Command line arguments to interpret. Defaults to ``sys.argv``
                without the executable name.

        Returns:
            mosaicjobconfig: Configuration object with the dataloader and
            tokenizer sections coerced into the strongly typed dataclasses used
            by the FL stack.
        """
        config = super().parse_args(args)
        return ensure_mosaic_job_config_types(config)


def load_mosaic_job_config(args: list[str] | None = None) -> MosaicJobConfig:
    """Parse CLI/TOML arguments into a fully-typed :class:`MosaicJobConfig`."""
    manager = MosaicConfigManager()
    return manager.parse_args(args or sys.argv[1:])


__all__ = [
    "FLMetricsConfigEnvelope",
    "MosaicConfigManager",
    "MosaicDataLoaderConfig",
    "MosaicJobConfig",
    "MosaicTokenizerConfig",
    "ensure_mosaic_job_config_types",
    "load_mosaic_job_config",
]
