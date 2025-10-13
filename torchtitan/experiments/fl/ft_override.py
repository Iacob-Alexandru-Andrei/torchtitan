#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DES-LOC specific configuration helpers for TorchFT management."""

from __future__ import annotations

from torchtitan.tools.logging import logger

__all__ = ["enable_desloc_only_ft"]


def enable_desloc_only_ft(job_config) -> None:
    """Ensure DES-LOC leverages TorchFT's semi-sync infrastructure."""
    desloc_cfg = getattr(job_config.optimizer, "desloc", None)
    if not getattr(desloc_cfg, "enabled", False):
        return

    current_method = job_config.fault_tolerance.semi_sync_method
    if current_method is None:
        job_config.fault_tolerance.semi_sync_method = "desloc"
        logger.info(
            "DES-LOC enabled; defaulting fault_tolerance.semi_sync_method to 'desloc'."
        )
    elif current_method.lower() != "desloc":
        logger.warning(
            "DES-LOC is enabled but fault_tolerance.semi_sync_method is set to '%s'. "
            "Proceeding without overriding; ensure this is intended.",
            current_method,
        )
