# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Training configuration for Llama-3 MuP models."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

from torch import nn

from torchtitan.components.ft import FTManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.lr_scheduler import build_fl_lr_schedulers
from torchtitan.experiments.fl.models.llama3_mup.infra.parallelize import (
    parallelize_llama_mup,
)
from torchtitan.experiments.fl.models.llama3_mup_disco.model.mup_args import (
    TransformerModelArgs,
)
from torchtitan.experiments.fl.models.llama3_mup_disco.model.mup_model import (
    Transformer,
)
from torchtitan.experiments.fl.models.llama3_mup.model.state_dict_adapter import (
    Llama3MuPStateDictAdapter,
)
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
from torchtitan.protocols.train_spec import TrainSpec


def build_mup_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Builder function for MuP that extracts parameter groups from the model.

    This function extracts parameter groups from the model and passes them to
    the core optimizer builder.

    Args:
        model_parts: List of model parts to optimize.
        optimizer_config: Optimizer configuration.
        parallel_dims: Parallel dimensions for distributed training.
        ft_manager: Optional fault tolerance manager.

    Returns:
        OptimizersContainer: Container with optimizers for each model part.
    """
    # Cast to Transformer to access MuP-specific methods
    model = cast("Transformer", model_parts[0])

    # Construct the initial kwargs dict from the config object.
    # This will be passed to the model to be potentially modified (e.g. for eps scaling).
    initial_optimizer_kwargs: dict[str, Any] = {
        "lr": optimizer_config.lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
    }

    overrides = model.build_mup_optimizer_overrides(
        lr=initial_optimizer_kwargs["lr"],
        eps=initial_optimizer_kwargs["eps"],
        weight_decay=initial_optimizer_kwargs["weight_decay"],
        scion_hidden_scale=getattr(optimizer_config, "scion_hidden_scale", None),
        scion_output_scale=getattr(optimizer_config, "scion_output_scale", None),
        scion_hidden_norm=getattr(optimizer_config, "scion_hidden_norm", None),
        scion_output_norm=getattr(optimizer_config, "scion_output_norm", None),
        scion_hidden_norm_kwargs=deepcopy(getattr(optimizer_config, "scion_hidden_norm_kwargs", None)),
        scion_output_norm_kwargs=deepcopy(getattr(optimizer_config, "scion_output_norm_kwargs", None)),
    )

    param_groups_list = overrides.param_groups if overrides else None

    updated_config = (
        replace(optimizer_config, **overrides.config_updates)
        if overrides and overrides.config_updates
        else optimizer_config
    )

    return build_optimizers(
        model_parts,
        updated_config,
        parallel_dims,
        ft_manager,
        param_groups=param_groups_list,
    )


llama3_mup_configs: dict[str, TransformerModelArgs] = {
    "16M": TransformerModelArgs(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=True,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=True,
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=True,
        use_flex_attn=True,
        attn_mask_type="block_causal",
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
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "125M": TransformerModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=True,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=True,
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=True,
        use_flex_attn=True,
        attn_mask_type="block_causal",
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 3.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 3.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "125M_scion": TransformerModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=False,
        layernorm_impl="rms",
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=False,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=False,
        use_torch_qk_layernorm=False,
        qk_layernorm_impl="rms",
        use_flex_attn=True,
        attn_mask_type="block_causal",
        use_disco=False,
        use_scion=True,
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 3.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 3.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "360M": TransformerModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=16,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=True,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=True,
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=True,
        use_flex_attn=True,
        attn_mask_type="block_causal",
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 4.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 6.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "360M_scion": TransformerModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=16,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=False,
        layernorm_impl="rms",
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=False,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=False,
        use_torch_qk_layernorm=False,
        qk_layernorm_impl="rms",
        use_flex_attn=True,
        attn_mask_type="block_causal",
        use_disco=False,
        use_scion=True,
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 4.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 6.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "720M": TransformerModelArgs(
        dim=2048,
        n_layers=12,
        n_heads=16,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=True,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=True,
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=True,
        use_flex_attn=True,
        attn_mask_type="block_causal",
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
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "720M_scion": TransformerModelArgs(
        dim=2048,
        n_layers=12,
        n_heads=16,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=False,
        layernorm_impl="rms",
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=False,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=False,
        use_torch_qk_layernorm=False,
        qk_layernorm_impl="rms",
        use_flex_attn=True,
        attn_mask_type="block_causal",
        use_disco=False,
        use_scion=True,
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
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=True,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=True,
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=True,
        use_flex_attn=True,
        attn_mask_type="block_causal",
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 8.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 6.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
    "1B_scion": TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=50368,
        rope_theta=10_000,
        ffn_dim_multiplier=None,
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        use_torch_layernorm=False,
        layernorm_impl="rms",
        torch_layernorm_bias=False,
        torch_layernorm_elementwise_affine=False,
        use_simple_silu_ffn=False,
        qk_norm=True,
        qk_norm_bias=False,
        qk_norm_elementwise_affine=False,
        use_torch_qk_layernorm=False,
        qk_layernorm_impl="rms",
        use_flex_attn=True,
        attn_mask_type="block_causal",
        use_disco=False,
        use_scion=True,
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,
            "mup_disable_hidden_lr_scaling": False,
            "mup_width_multiplier": 8.0,
            "mup_input_alpha": 1.0,
            "mup_output_alpha": 1.0,
            "completep_depth_alpha_enabled": True,
            "completep_depth_multiplier": 6.0,
            "completep_depth_alpha_exp": 1.0,
            "completep_eps_scaling_enabled": False,
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,
            "output_mult": None,
        },
    ),
}


# Register parameter-less RMSNorm variants (BF16-friendly) for every base config.
def _add_rms_variants() -> None:
    rms_suffix = "_RMS"
    base_items = list(llama3_mup_configs.items())
    for name, args in base_items:
        rms_args = replace(
            args,
            use_torch_layernorm=False,
            layernorm_impl="rms",
            torch_layernorm_elementwise_affine=False,
            torch_layernorm_bias=False,
            use_torch_qk_layernorm=False,
            qk_layernorm_impl="rms",
            qk_norm_elementwise_affine=False,
            qk_norm_bias=False,
            force_rmsnorm_bf16=True,
        )
        llama3_mup_configs[f"{name}{rms_suffix}"] = rms_args


_add_rms_variants()


def get_train_spec() -> TrainSpec:
    """Get the training specification for the Llama-3 MuP model."""
    return TrainSpec(
        name="llama3_mup_disco",
        model_cls=Transformer,
        model_args=llama3_mup_configs,
        parallelize_fn=parallelize_llama_mup,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_mup_optimizers,
        build_lr_schedulers_fn=build_fl_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3MuPStateDictAdapter,
    )
