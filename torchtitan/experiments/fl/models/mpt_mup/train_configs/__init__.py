# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training configuration for MuP-enabled MPT models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast, TYPE_CHECKING

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.experiments.fl.lr_scheduler import build_fl_lr_schedulers

from torchtitan.experiments.fl.models.mpt_mup.infra.parallelize import (
    parallelize_mpt_mup,
)
from torchtitan.experiments.fl.models.mpt_mup.model.mup_args import TransformerModelArgs
from torchtitan.experiments.fl.models.mpt_mup.model.mup_model import (
    MuPOptimizerOverride,
    Transformer,
)
from torchtitan.protocols.train_spec import TrainSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

    from torchtitan.components.ft import FTManager
    from torchtitan.config import Optimizer as OptimizerConfig
    from torchtitan.distributed import ParallelDims


def _maybe_override_optimizer_config(
    overrides: MuPOptimizerOverride | None,
    optimizer_config: OptimizerConfig,
) -> OptimizerConfig:
    if overrides is None or not overrides.config_updates:
        return optimizer_config
    return replace(optimizer_config, **overrides.config_updates)


def build_mup_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    model = cast("Transformer", model_parts[0])
    overrides = model.build_mup_optimizer_overrides(
        lr=optimizer_config.lr,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )
    param_groups = overrides.param_groups if overrides else None
    updated_config = _maybe_override_optimizer_config(overrides, optimizer_config)

    return build_optimizers(
        model_parts,
        updated_config,
        parallel_dims,
        ft_manager,
        param_groups=param_groups,
    )


mpt_mup_configs: Mapping[str, TransformerModelArgs] = {
    "16M": TransformerModelArgs(
        d_model=256,
        n_layers=4,
        n_heads=4,
        expansion_ratio=4.0,
        vocab_size=50368,
        max_seq_len=2048,
        learned_pos_emb=False,
        no_bias=True,
        use_cache=True,
        attn_config={
            "attn_impl": "torch",
            "rope": True,
            "alibi": False,
            "rope_theta": 10_000,
            "rope_impl": "hf",
        },
        ffn_config={
            "ffn_act_fn": {"name": "silu"},
            "ffn_hidden_size": 1024,
        },
        use_embedding_norm=True,
        use_peri_norm=True,
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 8.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 3.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": True,
            "mup_scale_expert_sel": True,
        },
        init_config={
            "name": "baseline_",
            "init_std": 0.02,
            "emb_init_std": 0.02,
        },
        fc_type={"name": "torch"},
    ),
    "125M": TransformerModelArgs(
        d_model=768,
        n_layers=12,
        n_heads=12,
        expansion_ratio=4.0,
        vocab_size=50368,
        max_seq_len=2048,
        learned_pos_emb=True,
        attn_config=None,
        ffn_config=None,
        tie_word_embeddings=True,
        use_embedding_norm=True,
        use_peri_norm=True,
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 1.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 1.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": True,
        },
        init_config={
            "name": "kaiming_normal_",
            "init_std": 0.02,
            "emb_init_std": 0.02,
        },
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        name="mpt_mup",
        model_cls=Transformer,
        model_args=mpt_mup_configs,
        parallelize_fn=parallelize_mpt_mup,
        pipelining_fn=None,
        build_optimizers_fn=build_mup_optimizers,
        build_lr_schedulers_fn=build_fl_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=None,
    )


__all__ = ["build_mup_optimizers", "get_train_spec", "mpt_mup_configs"]
