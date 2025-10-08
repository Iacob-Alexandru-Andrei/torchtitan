# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from typing import cast

from torchtitan.protocols.train_spec import TokenizerBuilder, TrainSpec

from ...dataloader.dataloader import build_mosaic_dataloader
from ...dataloader.tokenizer import build_mosaic_tokenizer
from ..llama3_mup.train_configs import get_train_spec as get_llama3_mup_train_spec


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for Llama3 MuP with Mosaic streaming support.

    This function wraps the base Llama3 MuP TrainSpec to make it compatible with
    Mosaic streaming by replacing the dataloader and tokenizer builders.
    """
    # Get the base Llama3 MuP spec
    base_spec = get_llama3_mup_train_spec()

    # Update all model configurations with larger vocab size for Mosaic tokenizer
    model_args = {
        name: replace(config, vocab_size=50368)
        for name, config in base_spec.model_args.items()
    }

    # Return a new spec with Mosaic components and updated vocab sizes
    return replace(
        base_spec,
        name="mosaic_llama3_mup",
        model_args=model_args,
        build_dataloader_fn=build_mosaic_dataloader,
        build_tokenizer_fn=cast(TokenizerBuilder, build_mosaic_tokenizer),
    )
