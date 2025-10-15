# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for adapting existing TrainSpecs to Mosaic streaming variants."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.experiments.fl.models.utils import (
    build_mosaic_spec,
    MosaicSpecOverrides,
)
from torchtitan.protocols.train_spec import (
    get_train_spec,
    register_train_spec,
    TrainSpec,
    update_train_spec,
)
from torchtitan.tools.logging import logger


@dataclass(slots=True)
class MosaicTrainSpecAdapter:
    """Adapter that derives and registers Mosaic-enabled TrainSpecs on demand."""

    base_spec_name: str
    """Name of the TrainSpec to wrap with Mosaic components."""

    spec_name: str | None = None
    """Optional explicit name for the derived Mosaic TrainSpec."""

    overrides: MosaicSpecOverrides | None = None
    """Optional override hooks applied while constructing the Mosaic spec."""

    _cached_spec: TrainSpec | None = None

    def build(self) -> TrainSpec:
        """Construct (but do not register) the Mosaic-enabled TrainSpec."""
        base_spec = get_train_spec(self.base_spec_name)
        spec_name = self.spec_name or f"mosaic_{base_spec.name}"
        mosaic_spec = build_mosaic_spec(
            base_spec,
            spec_name=spec_name,
            overrides=self.overrides,
        )
        self._cached_spec = mosaic_spec
        return mosaic_spec

    def register(self) -> TrainSpec:
        """Register the Mosaic TrainSpec, updating an existing entry if needed."""
        spec = self._cached_spec or self.build()

        try:
            existing_spec = get_train_spec(spec.name)
        except ValueError:
            register_train_spec(spec)
            logger.info("Registered new TrainSpec: %s", spec.name)
        else:
            if existing_spec != spec:
                try:
                    update_train_spec(spec)
                except ValueError:
                    register_train_spec(spec)
                    logger.info(
                        "Registered new TrainSpec after import-time registration: %s",
                        spec.name,
                    )
                else:
                    logger.info("Updated TrainSpec: %s", spec.name)
            else:
                logger.info("TrainSpec %s already registered, reusing it", spec.name)

        return get_train_spec(spec.name)


__all__ = ["MosaicTrainSpecAdapter"]
