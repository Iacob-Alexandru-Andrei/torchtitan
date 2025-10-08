# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from typing import cast

from torchtitan.models.llama3 import get_train_spec as get_base_llama3_spec
from torchtitan.protocols.train_spec import register_train_spec, TokenizerBuilder

from ..dataloader.dataloader import build_mosaic_dataloader
from ..dataloader.tokenizer import build_mosaic_tokenizer
from .llama3_mup.train_configs import get_train_spec as get_llama3_mup_train_spec


def _get_mosaic_llama3_spec():
    """
    This function wraps the base Llama3 TrainSpec to make it compatible with
    Mosaic streaming. It also adds a new model configuration with a larger
    vocab size.
    """
    # Get the base Llama3 spec
    base_spec = get_base_llama3_spec()

    # Add a new model configuration
    new_configs = {
        "debugmodel_smoll": replace(
            base_spec.model_args["debugmodel"],
            vocab_size=50368,
        )
    }
    model_args = {**base_spec.model_args, **new_configs}

    # Return a new spec with the Mosaic components and the new model configs
    return replace(
        base_spec,
        name="mosaic_llama3",
        model_args=model_args,
        build_dataloader_fn=build_mosaic_dataloader,
        build_tokenizer_fn=cast(TokenizerBuilder, build_mosaic_tokenizer),
    )


register_train_spec(_get_mosaic_llama3_spec())


def _get_llama3_mup_spec():
    spec = get_llama3_mup_train_spec()
    return replace(spec, name="llama3_mup")


register_train_spec(_get_llama3_mup_spec())