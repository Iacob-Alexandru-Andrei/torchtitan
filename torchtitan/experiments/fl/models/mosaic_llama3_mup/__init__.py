# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mosaic Llama3 MuP model with Mosaic streaming support."""

from dataclasses import replace

from torchtitan.experiments.fl.models.constants import MOSAIC_LLAMA_VOCAB_SIZE
from torchtitan.experiments.fl.models.mosaic_adapter import MosaicTrainSpecAdapter
from torchtitan.experiments.fl.models.utils import MosaicSpecOverrides
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.experiments.fl.validate import build_mosaic_validator
from torchtitan.protocols.train_spec import TrainSpec


def _update_vocab_sizes(base_spec: TrainSpec, mosaic_spec: TrainSpec) -> TrainSpec:
    model_args = {
        name: replace(config, vocab_size=MOSAIC_LLAMA_VOCAB_SIZE)
        for name, config in base_spec.model_args.items()
    }
    return replace(mosaic_spec, model_args=model_args)


_ADAPTER = MosaicTrainSpecAdapter(
    "llama3_mup",
    spec_name="mosaic_llama3_mup",
    overrides=MosaicSpecOverrides(
        optimizers=build_mosaic_optimizers,
        validator=build_mosaic_validator,
        post_transform=_update_vocab_sizes,
    ),
)

_REGISTERED_SPEC: TrainSpec = _ADAPTER.register()


def get_train_spec() -> TrainSpec:
    """Get the training specification for Llama3 MuP with Mosaic streaming support."""
    return _REGISTERED_SPEC


__all__ = ["build_mosaic_mup_optimizers", "get_train_spec"]
