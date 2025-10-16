# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mosaic-enabled MPT MuP model definitions for federated learning experiments."""

from torchtitan.experiments.fl.models.mosaic_adapter import MosaicTrainSpecAdapter
from torchtitan.experiments.fl.models.utils import MosaicSpecOverrides
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.experiments.fl.validate import build_mosaic_validator
from torchtitan.protocols.train_spec import TrainSpec


_ADAPTER = MosaicTrainSpecAdapter(
    "mpt_mup",
    spec_name="mosaic_mpt_mup",
    overrides=MosaicSpecOverrides(
        optimizers=build_mosaic_optimizers,
        validator=build_mosaic_validator,
    ),
)

_REGISTERED_SPEC: TrainSpec = _ADAPTER.register()


def get_train_spec() -> TrainSpec:
    """Return the Mosaic-enabled TrainSpec for the MPT MuP model."""
    return _REGISTERED_SPEC


__all__ = ["get_train_spec"]
