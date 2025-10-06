# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from typing import Any

from llmfoundry import registry
from llmfoundry.utils.registry_utils import construct_from_registry
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from torchtitan.config import JobConfig


def build_mosaic_tokenizer(
    job_config: JobConfig,
) -> PreTrainedTokenizerBase | PreTrainedTokenizerFast:

    tokenizer_name: str = job_config.training.mosaic.mosaic_tokenizer["name"]
    tokenizer_kwargs: dict[str, Any] = job_config.training.mosaic.mosaic_tokenizer.get(
        "kwargs", {}
    )

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
        tokenizer.model_max_length = tokenizer_kwargs.get(
            "model_max_length",
            int(1e30),
        )

    if not hasattr(tokenizer, "eos_token") or tokenizer.eos_token is None:
        raise ValueError(
            f"The tokenizer {tokenizer_name} must have an eos_token.",
        )

    return tokenizer


__all__ = ["build_mosaic_tokenizer"]
