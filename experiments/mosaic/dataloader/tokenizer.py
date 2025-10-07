# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Builds a Mosaic tokenizer from a config."""

from __future__ import annotations

import os
from typing import Any

from experiments.mosaic.configs.config import MosaicJobConfig

try:
    from llmfoundry import registry
    from llmfoundry.utils.registry_utils import construct_from_registry
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
    )
except ImportError as exc:
    raise RuntimeError(
        "llm-foundry is required to build Mosaic tokenizers. "
        "Please install llm-foundry to enable this integration."
    ) from exc


def build_mosaic_tokenizer(
    job_config: MosaicJobConfig,
) -> PreTrainedTokenizerBase | PreTrainedTokenizerFast:
    """Build a Mosaic tokenizer using the provided job configuration.

    This function supports building tokenizers from either the llm-foundry
    registry or directly from HuggingFace, based on the configuration in
    `job_config.mosaic_tokenizer`.

    Args:
        job_config: The Mosaic job configuration object.

    Returns:
        An instance of a HuggingFace tokenizer.

    Raises:
        ValueError: If the tokenizer configuration is missing or if the
                    resulting tokenizer does not have an EOS token.
    """
    if not job_config.mosaic_tokenizer:
        raise ValueError("mosaic_tokenizer config must be set.")

    tokenizer_name: str = job_config.mosaic_tokenizer["name"]
    tokenizer_kwargs: dict[str, Any] = job_config.mosaic_tokenizer.get("kwargs", {})

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if tokenizer_name in registry.tokenizers:
        tokenizer = construct_from_registry(
            name=tokenizer_name,
            registry=registry.tokenizers,
            partial_function=True,
            pre_validation_function=PreTrainedTokenizerBase,
            post_validation_function=None,
            kwargs=tokenizer_kwargs,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            **tokenizer_kwargs,
        )

        # HuggingFace does not respect the model_max_length kwarg, and overrides it with
        # min(kwargs['model_max_length'], original_config['model_max_length']), so we
        # explicitly set it here
        if "model_max_length" in tokenizer_kwargs:
            tokenizer.model_max_length = tokenizer_kwargs["model_max_length"]

    if not hasattr(tokenizer, "eos_token") or tokenizer.eos_token is None:
        raise ValueError(
            f"The tokenizer {tokenizer_name} must have an eos_token.",
        )

    return tokenizer


__all__ = ["build_mosaic_tokenizer"]