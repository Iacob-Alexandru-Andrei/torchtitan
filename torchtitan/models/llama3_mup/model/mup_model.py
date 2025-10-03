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

from torchtitan.models.attention import build_attention
from torchtitan.protocols.train_spec import ModelProtocol

from .mup_args import TransformerModelArgs


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


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


class Attention(nn.Module):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float, mup_enabled: bool, width_mult: float):
        base_std = 0.02
        mup_std = base_std / (width_mult**0.5) if mup_enabled else init_std

        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=mup_std)
        nn.init.normal_(self.wo.weight, mean=0.0, std=mup_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = self.sdpa(xq, keys, values)

        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def init_weights(self, init_std: float, mup_enabled: bool, width_mult: float):
        base_std = 0.02
        mup_std = base_std / (width_mult**0.5) if mup_enabled else init_std

        nn.init.normal_(self.w1.weight, mean=0.0, std=mup_std)
        nn.init.normal_(self.w3.weight, mean=0.0, std=mup_std)
        nn.init.normal_(self.w2.weight, mean=0.0, std=mup_std)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.dim = model_args.dim
        self.attention = Attention(model_args)
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
        base_std = 0.02
        init_std = base_std / (2 * self.model_args.n_layers) ** 0.5

        self.attention.init_weights(init_std, mup_enabled, width_mult)
        self.feed_forward.init_weights(init_std, mup_enabled, width_mult)

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

        init_std = self.init_config.init_std if self.init_config.init_std > 0 else 0.02
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