# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mosaic Llama3 model with Mosaic streaming support."""

from dataclasses import replace
from typing import cast

from torchtitan.experiments.fl.components import build_metrics_processor

from torchtitan.experiments.fl.dataloader.dataloader import build_mosaic_dataloader
from torchtitan.experiments.fl.dataloader.tokenizer import build_mosaic_tokenizer
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.experiments.fl.validate import build_mosaic_validator

from torchtitan.models.llama3 import get_train_spec as get_base_llama3_spec
from torchtitan.protocols.train_spec import TokenizerBuilder, TrainSpec


def get_train_spec() -> TrainSpec:
    """Get the training specification for Llama3 with Mosaic streaming support.

    This function wraps the base Llama3 TrainSpec to make it compatible with
    Mosaic streaming. It also adds a new model configuration with a larger
    vocab size (50368) to match the Mosaic tokenizer.
    """
    # Get the base Llama3 spec
    base_spec = get_base_llama3_spec()

    # Update all model configurations with larger vocab size for Mosaic tokenizer
    model_args = {
        name: replace(config, vocab_size=50368)
        for name, config in base_spec.model_args.items()
    }

    # Return a new spec with the Mosaic components and updated vocab sizes
    return replace(
        base_spec,
        name="mosaic_llama3",
        model_args=model_args,
        build_dataloader_fn=build_mosaic_dataloader,
        build_tokenizer_fn=cast("TokenizerBuilder", build_mosaic_tokenizer),
        build_metrics_processor_fn=build_metrics_processor,
        build_optimizers_fn=build_mosaic_optimizers,
        build_validator_fn=build_mosaic_validator,
    )
