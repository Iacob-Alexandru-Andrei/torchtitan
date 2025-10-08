# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.train_spec import ModelProtocol
# Import reusable components from the base llama3 model
from torchtitan.models.llama3.model.model import (
    Attention as BaseAttention,
    FeedForward as BaseFeedForward,
    precompute_freqs_cis,
)

from .mup_args import TransformerModelArgs


class MuPSharedEmbedding(nn.Embedding):
    """SharedEmbedding that scales its forward output for muP."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        scale: float = 1.0,
        use_embedding_norm: bool = False,
        norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(vocab_size, d_model, **kwargs)
        self.scale = scale
        self.norm = (
            nn.RMSNorm(d_model, eps=norm_eps) if use_embedding_norm else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.norm:
            out = self.norm(out)
        out = out * self.scale
        return out


class Attention(BaseAttention):
    def init_weights(self, std: float):
        for linear in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(linear.weight, mean=0.0, std=std)


class FeedForward(BaseFeedForward):
    def init_weights(self, std: float):
        for linear in (self.w1, self.w2, self.w3):
            nn.init.normal_(linear.weight, mean=0.0, std=std)


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
                / (self.mup_config.completep_depth_multiplier ** self.mup_config.completep_depth_alpha_exp)
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

        h = x + attn_out * (self.residual_scaling if self.mup_config.mup_enabled else 1.0)

        ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.use_peri_norm:
            ffn_out = self.post_ffn_norm(ffn_out)

        out = h + ffn_out * (self.residual_scaling if self.mup_config.mup_enabled else 1.0)
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

        if self.mup_config.mup_enabled:
            self.tok_embeddings = MuPSharedEmbedding(
                model_args.vocab_size,
                model_args.dim,
                scale=self.mup_config.mup_input_alpha,
                use_embedding_norm=model_args.use_embedding_norm,
                norm_eps=model_args.norm_eps,
            )
        else:
            self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        init_std = self.init_config.init_std
        emb_init_std = self.init_config.emb_init_std or init_std

        if self.mup_config.mup_enabled:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=emb_init_std)
        else:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        self.norm.reset_parameters()

        final_out_std = (self.model_args.dim**-0.5) * (self.init_config.output_mult or 1.0)
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

        emb_params.append(self.tok_embeddings.weight)
        if self.tok_embeddings.norm is not None:
            no_decay_params.extend(p for p in self.tok_embeddings.norm.parameters())

        no_decay_params.extend(p for p in self.norm.parameters())

        for block in self.layers.values():
            hidden_ln_params.extend(p for p in block.attention_norm.parameters())
            hidden_ln_params.extend(p for p in block.ffn_norm.parameters())
            if block.use_peri_norm:
                hidden_ln_params.extend(p for p in block.post_attn_norm.parameters())
                hidden_ln_params.extend(p for p in block.post_ffn_norm.parameters())

            decay_lr_depth_width_scaling.append(block.attention.wq.weight)
            decay_lr_depth_width_scaling.append(block.attention.wk.weight)
            decay_lr_depth_width_scaling.append(block.attention.wv.weight)
            decay_lr_depth_width_scaling.append(block.attention.wo.weight)
            decay_lr_depth_width_scaling.append(block.feed_forward.w1.weight)
            decay_lr_depth_width_scaling.append(block.feed_forward.w2.weight)
            decay_lr_depth_width_scaling.append(block.feed_forward.w3.weight)

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
                optimizer_config.get("eps", 1e-8)
                * width_lr_scaling
                * depth_eps_scaling
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

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h)
        output = self.output(h)

        if self.mup_config.mup_enabled:
            output = output * (
                self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier
            )
        return output