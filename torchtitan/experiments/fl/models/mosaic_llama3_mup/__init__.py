# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mosaic Llama3 MuP model with Mosaic streaming support."""

from dataclasses import replace
from typing import Any, cast, TYPE_CHECKING

from torch import nn

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.components import build_metrics_processor
from torchtitan.experiments.fl.configs.optimizers import MosaicOptimizerConfig
from torchtitan.experiments.fl.dataloader.dataloader import build_mosaic_dataloader
from torchtitan.experiments.fl.dataloader.tokenizer import build_mosaic_tokenizer
from torchtitan.experiments.fl.validate import build_mosaic_validator
from torchtitan.experiments.fl.models.llama3_mup.train_configs import (
    get_train_spec as get_llama3_mup_train_spec,
)
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.protocols.train_spec import TokenizerBuilder, TrainSpec

if TYPE_CHECKING:
    from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import Transformer


def build_mosaic_mup_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig | dict[str, Any],
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Builder function for MuP that works with Mosaic optimizers.

    This function extracts MuP parameter groups from the model and passes them to
    the Mosaic optimizer builder which supports optimizers like DecoupledAdamW.

    Args:
        model_parts: List of model parts to optimize.
        optimizer_config: Optimizer configuration (or dict to be converted).
        parallel_dims: Parallel dimensions for distributed training.
        ft_manager: Optional fault tolerance manager.

    Returns:
        OptimizersContainer: Container with optimizers for each model part.
    """
    # Convert dict to MosaicOptimizerConfig if needed
    if isinstance(optimizer_config, dict):
        optimizer_config = MosaicOptimizerConfig(**optimizer_config)

    # Cast to Transformer to access MuP-specific methods
    model = cast("Transformer", model_parts[0])

    # Construct the initial kwargs dict from the config object.
    initial_optimizer_kwargs: dict[str, Any] = {
        "lr": optimizer_config.lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
    }

    # MuP requires custom parameter groups for different learning rates.
    param_groups_or_iter, final_optimizer_kwargs = model.get_optimizer_param_groups(initial_optimizer_kwargs)

    # Convert Iterator to None for build_optimizers (it will use model.parameters())
    param_groups_list = param_groups_or_iter if isinstance(param_groups_or_iter, list) else None

    # Update the config with MuP-adjusted values
    optimizer_config.eps = final_optimizer_kwargs.get("eps", optimizer_config.eps)

    # Use the Mosaic optimizer builder which supports DecoupledAdamW, etc.
    # Cast is safe because MosaicOptimizerConfig extends OptimizerConfig with optional fields
    return build_mosaic_optimizers(
        model_parts,
        cast("MosaicOptimizerConfig", optimizer_config),
        parallel_dims,
        ft_manager,
        param_groups=param_groups_list,
    )


def get_train_spec() -> TrainSpec:
    """Get the training specification for Llama3 MuP with Mosaic streaming support.

    This function wraps the base Llama3 MuP TrainSpec to make it compatible with
    Mosaic streaming by replacing the dataloader, tokenizer, and optimizer builders
    to support Mosaic-specific optimizers like DecoupledAdamW.
    """
    # Get the base Llama3 MuP spec
    base_spec = get_llama3_mup_train_spec()

    # Update all model configurations with larger vocab size for Mosaic tokenizer
    model_args = {name: replace(config, vocab_size=50368) for name, config in base_spec.model_args.items()}

    # Return a new spec with Mosaic components and updated vocab sizes
    return replace(
        base_spec,
        name="mosaic_llama3_mup",
        model_args=model_args,
        build_dataloader_fn=build_mosaic_dataloader,
        build_tokenizer_fn=cast("TokenizerBuilder", build_mosaic_tokenizer),
        build_optimizers_fn=build_mosaic_mup_optimizers,
        build_metrics_processor_fn=build_metrics_processor,
        build_validator_fn=build_mosaic_validator,
    )
