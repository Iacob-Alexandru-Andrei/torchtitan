# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import JobConfig
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3_mup.model.mup_model import Transformer
from torchtitan.models.llama3_mup.model.mup_args import TransformerModelArgs
from torchtitan.models.llama3_mup.model.state_dict_adapter import (
    Llama3MuPStateDictAdapter,
)
from torchtitan.protocols.train_spec import TrainSpec
from torchtitan.components.optimizer import build_optimizers


def build_mup_optimizers(
    model_parts, optimizer_config, parallel_dims, ft_manager=None
):
    model = model_parts[0]
    param_groups, optimizer_kwargs = model.get_optimizer_param_groups(
        optimizer_config.optimizer_kwargs
    )
    optimizer_config.optimizer_kwargs = optimizer_kwargs
    return build_optimizers(
        model_parts, optimizer_config, parallel_dims, ft_manager, param_groups
    )


def get_train_spec(job_config: JobConfig) -> TrainSpec:
    """
    Get the training specification for the Llama-3 8B model.
    """
    return TrainSpec(
        model_class=Transformer,
        model_args_class=TransformerModelArgs,
        state_dict_adapter_class=Llama3MuPStateDictAdapter,
        parallelize_fn=parallelize_llama,
        build_optimizers_fn=build_mup_optimizers,
    )