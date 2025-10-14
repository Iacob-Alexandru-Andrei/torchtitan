# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Mosaic Llama3 MuP model with Mosaic streaming support."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from torch import nn

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.configs.optimizers import MosaicOptimizerConfig
from torchtitan.experiments.fl.models.constants import MOSAIC_LLAMA_VOCAB_SIZE
from torchtitan.experiments.fl.models.utils import ensure_mosaic_spec
from torchtitan.experiments.fl.validate import build_mosaic_validator
from torchtitan.experiments.fl.optimizer_builder import build_mosaic_optimizers
from torchtitan.protocols.train_spec import (
    TrainSpec,
    get_train_spec as get_registered_train_spec,
)

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


def _update_vocab_sizes(base_spec: TrainSpec, mosaic_spec: TrainSpec) -> TrainSpec:
    model_args = {
        name: replace(config, vocab_size=MOSAIC_LLAMA_VOCAB_SIZE)
        for name, config in base_spec.model_args.items()
    }
    return replace(mosaic_spec, model_args=model_args)


def get_train_spec() -> TrainSpec:
    """Get the training specification for Llama3 MuP with Mosaic streaming support."""

    spec_name = ensure_mosaic_spec(
        "llama3_mup",
        spec_name="mosaic_llama3_mup",
        optimizers_fn=build_mosaic_mup_optimizers,
        validator_fn=build_mosaic_validator,
        post_transform=_update_vocab_sizes,
    )
    return get_registered_train_spec(spec_name)
