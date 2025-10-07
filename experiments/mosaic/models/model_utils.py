# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model utilities for applying Mosaic-specific customizations."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from experiments.mosaic.dataloader.dataloader import build_mosaic_dataloader
from experiments.mosaic.dataloader.tokenizer import build_mosaic_tokenizer
from torchtitan.protocols.train_spec import TokenizerBuilder, TrainSpec


def get_mosaic_train_spec(base_spec: TrainSpec) -> TrainSpec:
    """
    Takes a base TrainSpec and replaces its dataloader and tokenizer
    builders with Mosaic-specific implementations.

    This allows any model to be trained with Mosaic streaming by simply
    wrapping its existing TrainSpec with this function.

    Args:
        base_spec: The base TrainSpec to modify.

    Returns:
        A new TrainSpec configured for Mosaic streaming.
    """
    return replace(
        base_spec,
        build_dataloader_fn=build_mosaic_dataloader,
        build_tokenizer_fn=cast(TokenizerBuilder, build_mosaic_tokenizer),
    )


__all__ = ["get_mosaic_train_spec"]