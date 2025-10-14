# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mosaic Llama3 model with Mosaic streaming support."""

from dataclasses import replace

from torchtitan.experiments.fl.models.constants import MOSAIC_LLAMA_VOCAB_SIZE
from torchtitan.experiments.fl.models.utils import ensure_mosaic_spec
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.experiments.fl.validate import build_mosaic_validator
from torchtitan.protocols.train_spec import (
    TrainSpec,
    get_train_spec as get_registered_train_spec,
)


def _update_vocab_sizes(base_spec: TrainSpec, mosaic_spec: TrainSpec) -> TrainSpec:
    model_args = {
        name: replace(config, vocab_size=MOSAIC_LLAMA_VOCAB_SIZE)
        for name, config in base_spec.model_args.items()
    }
    return replace(mosaic_spec, model_args=model_args)


def get_train_spec() -> TrainSpec:
    """Get the training specification for Llama3 with Mosaic streaming support."""

    spec_name = ensure_mosaic_spec(
        "llama3",
        spec_name="mosaic_llama3",
        optimizers_fn=build_mosaic_optimizers,
        validator_fn=build_mosaic_validator,
        post_transform=_update_vocab_sizes,
    )
    return get_registered_train_spec(spec_name)
