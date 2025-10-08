# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

# Import reusable components from the base llama3 model
from torchtitan.models.llama3.model.model import (
    Attention as BaseAttention,
    FeedForward as BaseFeedForward,
    precompute_freqs_cis,
)

from torchtitan.protocols.train_spec import ModelProtocol

from .mup_args import TransformerModelArgs


class Attention(BaseAttention):
    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)


class FeedForward(BaseFeedForward):
    def init_weights(self, init_std: float):
        for linear in (self.w1, self.w2, self.w3):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.dim = model_args.dim
        # Use the MuP-specific Attention class
        self.attention = Attention(model_args)
        # Use the MuP-specific FeedForward class
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.use_peri_norm = model_args.use_peri_norm
        if self.use_peri_norm:
            self.post_attn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.mup_config = model_args.mup_config_obj
        if self.mup_config.mup_enabled:
            self.residual_scaling = (
                1.0
                / (
                    self.mup_config.completep_depth_multiplier
                    ** self.mup_config.completep_depth_alpha_exp
                )
                if self.mup_config.completep_depth_alpha_enabled
                else 1.0
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        attn_out = self.attention(self.attention_norm(x), freqs_cis)
        if self.use_peri_norm:
            attn_out = self.post_attn_norm(attn_out)

        h = x + attn_out * (
            self.residual_scaling if self.mup_config.mup_enabled else 1.0
        )

        ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.use_peri_norm:
            ffn_out = self.post_ffn_norm(ffn_out)

        out = h + ffn_out * (
            self.residual_scaling if self.mup_config.mup_enabled else 1.0
        )
        return out

    def init_weights(self):
        mup_enabled = self.mup_config.mup_enabled
        width_mult = self.mup_config.mup_width_multiplier
        init_std = self.model_args.init_config_obj.init_std

        if mup_enabled:
            std_to_use = init_std / (width_mult**0.5)
        else:
            std_to_use = init_std

        self.attention.init_weights(std_to_use)
        self.feed_forward.init_weights(std_to_use)

        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        if self.use_peri_norm:
            for norm in (self.post_attn_norm, self.post_ffn_norm):
                norm.reset_parameters()


class Transformer(nn.Module, ModelProtocol):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.mup_config = model_args.mup_config_obj
        self.init_config = model_args.init_config_obj

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Embedding normalization and scaling
        self.embedding_norm = (
            nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
            if model_args.use_embedding_norm
            else None
        )
        self.embedding_scale = (
            self.mup_config.mup_input_alpha if self.mup_config.mup_enabled else 1.0
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Only create output layer if not tying weights
        # When tying, we'll use F.linear with tok_embeddings.weight directly
        if not model_args.tie_word_embeddings:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        else:
            self.output = None

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        init_std = self.init_config.init_std
        emb_init_std = self.init_config.emb_init_std or init_std

        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=emb_init_std)

        if self.embedding_norm is not None:
            self.embedding_norm.reset_parameters()

        for layer_module in self.layers.values():
            if layer_module is not None:
                layer = cast(TransformerBlock, layer_module)
                layer.init_weights()

        self.norm.reset_parameters()

        # Initialize output layer weights (only if not tying)
        if self.output is not None:
            final_out_std = (self.model_args.dim**-0.5) * (
                self.init_config.output_mult or 1.0
            )
            nn.init.normal_(self.output.weight, mean=0.0, std=final_out_std)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def get_optimizer_param_groups(self, optimizer_config: dict):
        if not (
            self.mup_config.mup_enabled
            and not self.mup_config.mup_disable_hidden_lr_scaling
        ):
            return (
                self.parameters(),
                optimizer_config,
            )

        emb_params = []
        hidden_ln_params = []
        decay_lr_depth_width_scaling = []
        hidden_bias_params = []
        no_decay_params = []

        # Add embedding weight
        emb_params.append(self.tok_embeddings.weight)

        # Add output weight if it exists (not tied)
        if self.output is not None:
            emb_params.append(self.output.weight)

        if self.embedding_norm is not None:
            no_decay_params.extend(p for p in self.embedding_norm.parameters())

        no_decay_params.extend(p for p in self.norm.parameters())

        for block_module in self.layers.values():
            # Cast to TransformerBlock to help type checker
            block = cast(TransformerBlock, block_module)

            hidden_ln_params.extend(p for p in block.attention_norm.parameters())
            hidden_ln_params.extend(p for p in block.ffn_norm.parameters())
            if block.use_peri_norm:
                hidden_ln_params.extend(p for p in block.post_attn_norm.parameters())
                hidden_ln_params.extend(p for p in block.post_ffn_norm.parameters())

            # Cast attention and feed_forward to their correct types
            attention = cast(Attention, block.attention)
            feed_forward = cast(FeedForward, block.feed_forward)

            decay_lr_depth_width_scaling.append(attention.wq.weight)
            decay_lr_depth_width_scaling.append(attention.wk.weight)
            decay_lr_depth_width_scaling.append(attention.wv.weight)
            decay_lr_depth_width_scaling.append(attention.wo.weight)
            decay_lr_depth_width_scaling.append(feed_forward.w1.weight)
            decay_lr_depth_width_scaling.append(feed_forward.w2.weight)
            decay_lr_depth_width_scaling.append(feed_forward.w3.weight)

        base_lr = optimizer_config["lr"]
        weight_decay = optimizer_config.get("weight_decay", 0.0)

        width_lr_scaling = 1.0 / self.mup_config.mup_width_multiplier
        depth_lr_scaling = 1.0
        if self.mup_config.completep_depth_alpha_enabled:
            depth_lr_scaling = self.mup_config.completep_depth_multiplier ** (
                self.mup_config.completep_depth_alpha_exp - 1.0
            )

        if self.mup_config.completep_eps_scaling_enabled:
            depth_eps_scaling = self.mup_config.completep_depth_multiplier ** (
                -1.0 * self.mup_config.completep_depth_alpha_exp
            )
            optimizer_config["eps"] = (
                optimizer_config.get("eps", 1e-8) * width_lr_scaling * depth_eps_scaling
            )

        param_groups = [
            {"params": emb_params, "weight_decay": weight_decay, "lr": base_lr},
            {
                "params": hidden_ln_params,
                "weight_decay": 0.0,
                "lr": base_lr * depth_lr_scaling,
            },
            {
                "params": decay_lr_depth_width_scaling,
                "weight_decay": weight_decay / width_lr_scaling,
                "lr": base_lr * width_lr_scaling * depth_lr_scaling,
            },
            {
                "params": hidden_bias_params,
                "weight_decay": 0.0,
                "lr": base_lr * depth_lr_scaling,
            },
            {"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr},
        ]

        param_groups = [pg for pg in param_groups if pg["params"]]

        return param_groups, optimizer_config

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens)

        # Apply embedding normalization and scaling
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)
        h = h * self.embedding_scale

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h)

        # Use output layer or F.linear with embedding weights if tied
        if self.output is None:
            output = F.linear(h, self.tok_embeddings.weight)
        else:
            output = self.output(h)

        if self.mup_config.mup_enabled:
            output = output * (
                self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier
            )
        return output
