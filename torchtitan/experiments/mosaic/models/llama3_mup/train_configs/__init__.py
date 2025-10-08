# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.optimizer import build_optimizers
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.components.ft import FTManager

from torchtitan.experiments.mosaic.models.llama3_mup.model.mup_model import Transformer
from torchtitan.experiments.mosaic.models.llama3_mup.model.mup_args import TransformerModelArgs
from torchtitan.experiments.mosaic.models.llama3_mup.model.state_dict_adapter import (
    Llama3MuPStateDictAdapter,
)
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.protocols.train_spec import TrainSpec


def build_mup_optimizers(
    model_parts,
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
):
    """
    Builder function for MuP that extracts parameter groups from the model
    and passes them to the core optimizer builder.
    """
    model = model_parts[0]

    # Construct the initial kwargs dict from the config object.
    # This will be passed to the model to be potentially modified (e.g. for eps scaling).
    initial_optimizer_kwargs = {
        "lr": optimizer_config.lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
    }

    # MuP requires custom parameter groups for different learning rates.
    # The model returns the parameter groups and potentially updated optimizer kwargs.
    param_groups, final_optimizer_kwargs = model.get_optimizer_param_groups(
        initial_optimizer_kwargs
    )

    # The model might have updated some optimizer kwargs (e.g., eps for MuP scaling).
    # We update the original config object so that the core builder uses the correct values.
    optimizer_config.eps = final_optimizer_kwargs.get("eps", optimizer_config.eps)

    # Use the core optimizer builder with the custom param groups.
    return build_optimizers(
        model_parts,
        optimizer_config,
        parallel_dims,
        ft_manager,
        param_groups=param_groups,
    )


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for the Llama-3 MuP model.
    """
    return TrainSpec(
        model_class=Transformer,
        model_args_class=TransformerModelArgs,
        state_dict_adapter_class=Llama3MuPStateDictAdapter,
        parallelize_fn=parallelize_llama,
        build_optimizers_fn=build_mup_optimizers,
    )