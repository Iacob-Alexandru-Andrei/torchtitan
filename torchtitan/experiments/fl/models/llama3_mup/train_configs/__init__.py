# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.ft import FTManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.fl.models.llama3_mup.model.mup_args import (
    TransformerModelArgs,
)

from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import Transformer
from torchtitan.experiments.fl.models.llama3_mup.model.state_dict_adapter import (
    Llama3MuPStateDictAdapter,
)
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3.infra.pipeline import pipeline_llama
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


llama3_mup_configs = {
    "16M": TransformerModelArgs(
        dim=256,
        n_layers=4,
        n_heads=4,
        vocab_size=2048,
        rope_theta=10_000,
        # FFN expansion ratio of 4x (4 * 256 = 1024)
        ffn_dim_multiplier=4.0,
        multiple_of=256,
        # MuP features
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        # Flex attention
        use_flex_attn=True,
        attn_mask_type="block_causal",
        # No μP/CompleteP for baseline 16M model
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
            "init_std": 0.02,
            "emb_init_std": 0.02,  # Will default to init_std
            "output_mult": None,  # No special output scaling
        },
    ),
    "125M": TransformerModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        vocab_size=2048,
        rope_theta=10_000,
        # FFN expansion ratio of 4x (4 * 768 = 3072)
        ffn_dim_multiplier=4.0,
        multiple_of=256,
        # MuP features
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        # Flex attention
        use_flex_attn=True,
        attn_mask_type="block_causal",
        # μP and CompleteP configuration
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,  # Keep attention scaling standard
            "mup_disable_hidden_lr_scaling": False,  # Apply hidden layer LR scaling
            "mup_width_multiplier": 3.0,  # Scale from base width
            "mup_input_alpha": 1.0,  # Standard input scaling
            "mup_output_alpha": 1.0,  # Standard output scaling
            "completep_depth_alpha_enabled": True,  # Enable depth scaling
            "completep_depth_multiplier": 3.0,  # Scale from base depth
            "completep_depth_alpha_exp": 1.0,  # Linear depth scaling
            "completep_eps_scaling_enabled": True,  # Scale optimizer eps
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,  # Will use mup-adjusted value
            "output_mult": None,  # Will use mup-adjusted value
        },
    ),
    "720M": TransformerModelArgs(
        dim=2048,
        n_layers=12,
        n_heads=16,
        vocab_size=2048,
        rope_theta=10_000,
        # FFN expansion ratio of 4x (4 * 2048 = 8192)
        ffn_dim_multiplier=4.0,
        multiple_of=256,
        # MuP features
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        # Flex attention
        use_flex_attn=True,
        attn_mask_type="block_causal",
        # μP and CompleteP configuration
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,  # Keep attention scaling standard
            "mup_disable_hidden_lr_scaling": False,  # Apply hidden layer LR scaling
            "mup_width_multiplier": 8.0,  # Scale from base width
            "mup_input_alpha": 1.0,  # Standard input scaling
            "mup_output_alpha": 1.0,  # Standard output scaling
            "completep_depth_alpha_enabled": True,  # Enable depth scaling
            "completep_depth_multiplier": 3.0,  # Scale from base depth
            "completep_depth_alpha_exp": 1.0,  # Linear depth scaling
            "completep_eps_scaling_enabled": True,  # Scale optimizer eps
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,  # Will use mup-adjusted value
            "output_mult": None,  # Will use mup-adjusted value
        },
    ),
    "1.3B": TransformerModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=2048,
        rope_theta=10_000,
        # FFN expansion ratio of 4x (4 * 2048 = 8192)
        ffn_dim_multiplier=4.0,
        multiple_of=256,
        # MuP features
        use_embedding_norm=True,
        use_peri_norm=True,
        tie_word_embeddings=True,
        # Flex attention
        use_flex_attn=True,
        attn_mask_type="block_causal",
        # μP and CompleteP configuration
        mup_config={
            "mup_enabled": True,
            "mup_disable_attention_scaling": True,  # Keep attention scaling standard
            "mup_disable_hidden_lr_scaling": False,  # Apply hidden layer LR scaling
            "mup_width_multiplier": 8.0,  # Scale from base width (same as 720M)
            "mup_input_alpha": 1.0,  # Standard input scaling
            "mup_output_alpha": 1.0,  # Standard output scaling
            "completep_depth_alpha_enabled": True,  # Enable depth scaling
            "completep_depth_multiplier": 6.0,  # Scale from base depth (2x deeper)
            "completep_depth_alpha_exp": 1.0,  # Linear depth scaling
            "completep_eps_scaling_enabled": True,  # Scale optimizer eps
        },
        init_config={
            "init_std": 0.02,
            "emb_init_std": 0.02,  # Will use mup-adjusted value
            "output_mult": None,  # Will use mup-adjusted value
        },
    ),
}


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for the Llama-3 MuP model.
    """
    return TrainSpec(
        name="llama3_mup",
        model_cls=Transformer,
        model_args=llama3_mup_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_mup_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3MuPStateDictAdapter,
    )
