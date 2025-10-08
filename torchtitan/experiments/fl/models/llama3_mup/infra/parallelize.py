# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization for Llama3 MuP model with weight tying support.

This module extends the base Llama3 parallelization with proper handling
of weight tying, following the Qwen3 pattern where weights are tied AFTER
all parallelization strategies have been applied.
"""

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.infra.parallelize import parallelize_llama


def parallelize_llama_mup(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """Parallelize Llama3 MuP model with weight tying support.

    This function applies the base Llama3 parallelization and then
    handles weight tying by sharing the tok_embeddings.weight with
    output.weight AFTER parallelization is complete.

    This ensures both weights are parallelized consistently before
    they are tied, avoiding DTensor/Tensor mixing errors.

    Args:
        model: The Llama3 MuP Transformer model to parallelize
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration

    Returns:
        The parallelized model with weight tying applied if enabled
    """
    # Apply standard Llama3 parallelization (TP, FSDP, compilation, etc.)
    model = parallelize_llama(model, parallel_dims, job_config)

    # Apply weight tying AFTER parallelization (following Qwen3 pattern)
    # At this point, both tok_embeddings.weight and output.weight are DTensors
    # with consistent sharding, so tying them is safe
    if (
        model.model_args.tie_word_embeddings  # pyright: ignore[reportAttributeAccessIssue]
    ):
        model.output.weight = (  # pyright: ignore[reportAttributeAccessIssue]
            model.tok_embeddings.weight  # pyright: ignore[reportAttributeAccessIssue]
        )

    return model
