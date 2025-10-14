#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DES-LOC configuration helpers for integrating with TorchFT."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from torchtitan.components.ft import has_torchft
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["configure_desloc"]


@contextmanager
def configure_desloc(job_config: object) -> Iterator[None]:
    """Validate and install DES-LOC integrations prior to trainer creation."""
    desloc_cfg = getattr(job_config.optimizer, "desloc", None)
    if not getattr(desloc_cfg, "enabled", False):
        yield
        return

    if desloc_cfg.param_sync_every <= 0:
        msg = "desloc.param_sync_every must be a positive integer."
        raise ValueError(msg)

    if not has_torchft:
        msg = "DES-LOC support requires the torchft package to be installed."
        raise RuntimeError(msg)

    fault_tolerance = getattr(job_config, "fault_tolerance", None)
    if fault_tolerance is None or not getattr(fault_tolerance, "enable", False):
        msg = "DES-LOC requires fault_tolerance.enable to be True."
        raise ValueError(msg)

    method = fault_tolerance.semi_sync_method
    if method is None:
        logger.info(
            "DES-LOC enabled; defaulting fault_tolerance.semi_sync_method to 'desloc'."
        )
    elif method.lower() != "desloc":
        msg = "DES-LOC requires fault_tolerance.semi_sync_method to be 'desloc'."
        raise ValueError(msg)

    fault_tolerance.semi_sync_method = "desloc"

    yield
