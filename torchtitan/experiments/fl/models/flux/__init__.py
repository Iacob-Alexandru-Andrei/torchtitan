# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Flux compatibility layer for the FL infrastructure."""

from __future__ import annotations

from dataclasses import replace

from torchtitan.experiments.fl.components import build_metrics_processor
from torchtitan.experiments.fl.dataloader.flux_builder import (
    build_fl_flux_dataloader,
    build_fl_flux_validation_dataloader,
)
from torchtitan.experiments.fl.lr_scheduler import build_fl_lr_schedulers
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.experiments.fl.validate.flux_validator import build_fl_flux_validator
from torchtitan.experiments.flux import get_train_spec as get_flux_train_spec
from torchtitan.protocols.train_spec import TrainSpec
from torchtitan.tools.logging import logger

__all__ = ["get_train_spec"]

_REGISTERED_SPEC: TrainSpec | None = None


def _build_flux_spec() -> TrainSpec:
    """Build the Flux TrainSpec wired into FL components."""
    base_spec = get_flux_train_spec()
    flux_spec = replace(
        base_spec,
        build_optimizers_fn=build_mosaic_optimizers,
        build_lr_schedulers_fn=build_fl_lr_schedulers,
        build_dataloader_fn=build_fl_flux_dataloader,
        build_validator_fn=build_fl_flux_validator,
        build_metrics_processor_fn=build_metrics_processor,
    )

    logger.info("Prepared Flux TrainSpec for FL integration: %s", flux_spec.name)

    return flux_spec


_REGISTERED_SPEC = _build_flux_spec()


def get_train_spec() -> TrainSpec:
    """Return the Flux TrainSpec configured for the FL stack."""
    return _REGISTERED_SPEC
