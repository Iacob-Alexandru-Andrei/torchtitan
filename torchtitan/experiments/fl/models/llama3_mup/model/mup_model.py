# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
"""Model components for Llama-3 MuP."""

import logging
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast, Protocol, runtime_checkable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

# Import reusable components from the base llama3 model
from torchtitan.experiments.qwen3.model.model import (
    Attention as QwenAttention,
    apply_rotary_emb,
    precompute_rope_cache,
    repeat_kv,
)
from torchtitan.models.llama3.model.model import (
    FeedForward as BaseFeedForward,
    Transformer as BaseTransformer,
    TransformerBlock as BaseTransformerBlock,
)

from .disco_init import init_linear_weight, initialize_tensor
from .mup_args import TransformerModelArgs as TransformerModelArgsMuP


logger = logging.getLogger(__name__)
_debug_env = os.getenv("TORCHTITAN_DEBUG_DISCO_NORMS")
if _debug_env is None:
    _debug_env = os.getenv("TORCHTITAN_DEBUG_SCION_NORMS", "")
_DISCO_NORM_DEBUG_ENABLED = _debug_env.lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SCION_SCALE_DEBUG_ENABLED = os.getenv("TORCHTITAN_DEBUG_SCION_SCALES", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _cast_if_autocast_enabled(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Cast tensors to the current autocast dtype when autocast is active."""
    if tensor is None:
        return None
    if not torch.is_autocast_enabled():
        return tensor
    if tensor.device.type == "cuda":
        dtype = torch.get_autocast_gpu_dtype()
    elif tensor.device.type == "cpu":
        dtype = torch.get_autocast_cpu_dtype()
    else:
        msg = f"Unsupported device for autocast: {tensor.device.type}"
        raise NotImplementedError(msg)
    return tensor.to(dtype=dtype)


class LPLayerNorm(torch.nn.LayerNorm):
    """LayerNorm variant that evaluates in the autocast downcast dtype."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight)
        downcast_bias = _cast_if_autocast_enabled(self.bias)
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )


class TitanRMSNorm(nn.Module):
    """RMSNorm variant inspired by Gemma with optional BF16 execution."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        add_unit_offset: bool = True,
        force_bf16: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, Sequence):
            self.normalized_shape = tuple(normalized_shape)
        else:
            self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.add_unit_offset = add_unit_offset
        self.force_bf16 = force_bf16
        self._norm_axes = tuple(range(-len(self.normalized_shape), 0))  # normalize last N dims

        if elementwise_affine:
            init = torch.zeros(self.normalized_shape)
            self.weight = nn.Parameter(init)
        else:
            self.register_parameter("weight", None)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.constant_(self.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            if self.force_bf16 and x.dtype != torch.bfloat16:
                return F.rms_norm(
                    x.to(torch.bfloat16),
                    self.normalized_shape,
                    None,
                    self.eps,
                ).to(x.dtype)
            return F.rms_norm(x, self.normalized_shape, None, self.eps)

        compute_dtype = torch.bfloat16 if self.force_bf16 else torch.float32
        hidden_states = x.to(compute_dtype)
        variance = hidden_states.pow(2).mean(dim=self._norm_axes, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        scale = 1 + self.weight if self.add_unit_offset else self.weight
        hidden_states = hidden_states * scale.to(compute_dtype)

        return hidden_states.to(x.dtype)


def _build_norm_module(
    normalized_shape: int | Sequence[int],
    *,
    eps: float,
    model_args: TransformerModelArgsMuP,
    prefer_torch: bool,
    elementwise_affine: bool = True,
    bias: bool = False,
) -> nn.Module:
    if prefer_torch:
        return LPLayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )
    return TitanRMSNorm(
        normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        add_unit_offset=elementwise_affine,
        force_bf16=model_args.force_rmsnorm_bf16,
    )


@dataclass(frozen=True)
class MuPOptimizerOverride:
    """MuP-specific optimizer adjustments returned by compatible models."""

    param_groups: list[dict[str, Any]] | None
    """Optional custom parameter groups to hand to the optimizer constructor."""

    config_updates: dict[str, Any]
    """Keyword overrides to apply when building the optimizer configuration."""


@runtime_checkable
class SupportsMuPOptimizerOverrides(Protocol):
    """Protocol for models exposing MuP optimizer override information."""

    def build_mup_optimizer_overrides(
        self,
        *,
        lr: float,
        eps: float,
        weight_decay: float,
        scion_hidden_scale: float | None = None,
        scion_output_scale: float | None = None,
        scion_hidden_norm: str | None = None,
        scion_output_norm: str | None = None,
        scion_hidden_norm_kwargs: dict[str, Any] | None = None,
        scion_output_norm_kwargs: dict[str, Any] | None = None,
    ) -> MuPOptimizerOverride | None:
        """Return MuP-aware optimizer overrides, if any."""


class _MuPScaledAttention(nn.Module):
    """Wrapper that injects MuP attention scaling into SDPA kernels."""

    def __init__(self, inner: nn.Module, scale: float) -> None:
        super().__init__()
        self.inner = inner
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Ignore the incoming scale and force the MuP-specific one.
        return self.inner(q, k, v, scale=self.scale, **kwargs)


class Attention(QwenAttention):
    """Multi-head attention layer with MuP-specific weight initialization."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(model_args)
        self.model_args = model_args
        self.mup_config = model_args.mup_config_obj
        self._disco_eps = model_args.init_config_obj.scion_init_eps
        self._hidden_init_type = model_args.init_config_obj.resolved_hidden_init(model_args.use_disco)
        self._trunc_normal_cutoff = model_args.init_config_obj.trunc_normal_cutoff
        self.v_norm: nn.Module | None = None
        self.o_norm: nn.Module | None = None
        if model_args.qk_norm and self.q_norm is not None and self.k_norm is not None:
            self.q_norm = self._build_head_norm(model_args)
            self.k_norm = self._build_head_norm(model_args)
            logger.info(
                "MuP QK head normalization enabled: head_dim=%d, norm_type=%s",
                self.head_dim,
                self.q_norm.__class__.__name__,
            )
        if model_args.use_attention_value_norm:
            self.v_norm = self._build_head_norm(model_args, use_torch=model_args.use_torch_layernorm)
        if model_args.use_attention_output_norm:
            self.o_norm = self._build_output_norm(model_args)
        apply_mup_attention_scaling = (
            self.mup_config.mup_enabled
            and not self.mup_config.mup_disable_attention_scaling
            and not model_args.use_scion
        )
        if apply_mup_attention_scaling:
            scale = 1.0 / float(self.head_dim)
            self.sdpa = _MuPScaledAttention(self.sdpa, scale)
            logger.info(
                "MuP attention scaling enabled: head_dim=%d, scale=%.6f",
                self.head_dim,
                scale,
            )
        else:
            reason = "disabled"
            if not self.mup_config.mup_enabled:
                reason = "mup_disabled"
            elif self.mup_config.mup_disable_attention_scaling:
                reason = "config_opt_out"
            logger.info(
                "MuP attention scaling skipped (%s): head_dim=%d",
                reason,
                self.head_dim,
            )

    def init_weights(self, init_std: float) -> None:
        """Initialize weights with MuP-specific scaling.

        Args:
            init_std (float): Standard deviation for weight initialization.
        """
        layer_id = getattr(self, "mup_layer_id", "unknown")
        logger.info(
            "Initializing MuP Attention (layer=%s) with std=%.6f for weights [wq, wk, wv, wo]",
            layer_id,
            init_std,
        )
        for linear in (self.wq, self.wk, self.wv, self.wo):
            init_linear_weight(
                linear,
                init_std=init_std,
                init_type=self._hidden_init_type,
                scion_eps=self._disco_eps,
                trunc_normal_cutoff=self._trunc_normal_cutoff,
            )
        for norm in (self.q_norm, self.k_norm, self.v_norm, self.o_norm):
            if norm is not None:
                norm.reset_parameters()

    def _build_head_norm(
        self,
        model_args: TransformerModelArgsMuP,
        *,
        use_torch: bool | None = None,
    ) -> nn.Module:
        flag = model_args.use_torch_qk_layernorm if use_torch is None else use_torch
        return _build_norm_module(
            self.head_dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=flag,
            elementwise_affine=model_args.qk_norm_elementwise_affine,
            bias=model_args.qk_norm_bias,
        )

    def _build_output_norm(self, model_args: TransformerModelArgsMuP) -> nn.Module:
        return _build_norm_module(
            model_args.dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=model_args.use_torch_layernorm,
            elementwise_affine=model_args.torch_layernorm_elementwise_affine,
            bias=model_args.torch_layernorm_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)
        if self.v_norm is not None:
            xv = self.v_norm(xv)

        xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        output = self.sdpa(xq, xk, xv, scale=self.scaling)

        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        if self.o_norm is not None:
            output = self.o_norm(output)
        return self.wo(output)


class FeedForward(BaseFeedForward):
    """Feed-forward network with MuP-specific weight initialization."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        self.model_args = model_args
        self._disco_eps = model_args.init_config_obj.scion_init_eps
        self._hidden_init_type = model_args.init_config_obj.resolved_hidden_init(model_args.use_disco)
        self._trunc_normal_cutoff = model_args.init_config_obj.trunc_normal_cutoff
        self._activation_scale = math.sqrt(2.0) if model_args.use_disco else 1.0
        hidden_dim = 4 * model_args.dim
        if model_args.use_simple_silu_ffn:
            # Base Llama FFN scales the provided hidden dim by 2/3 to support the gated branch.
            # Pre-scale by 3/2 so the resulting two-layer FFN width stays at 4 * d_model.
            hidden_dim = 6 * model_args.dim
        super().__init__(
            dim=model_args.dim,
            hidden_dim=hidden_dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        effective_hidden_dim = self.w1.out_features

        self.use_simple_silu_ffn = model_args.use_simple_silu_ffn
        if self.use_simple_silu_ffn:
            # Drop the unused gated branch when configured for a simple MLP.
            self.w3 = None
        if model_args.use_mlp_mid_norm and not self.use_simple_silu_ffn:
            elementwise_affine = model_args.torch_layernorm_elementwise_affine
            bias = model_args.torch_layernorm_bias if model_args.use_torch_layernorm else False
            self.mid_norm = _build_norm_module(
                effective_hidden_dim,
                eps=model_args.norm_eps,
                model_args=model_args,
                prefer_torch=model_args.use_torch_layernorm,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )
        else:
            self.mid_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply either gated or simple SiLU feed-forward projection."""
        up = F.silu(self.w1(x))
        if self._activation_scale != 1.0:
            up = up * self._activation_scale
        if self.mid_norm is not None:
            up = self.mid_norm(up)
        if self.use_simple_silu_ffn or self.w3 is None:
            return self.w2(up)
        gate = self.w3(x)
        return self.w2(up * gate)

    def init_weights(self, init_std: float) -> None:
        """Initialize weights with MuP-specific scaling."""
        layer_id = getattr(self, "mup_layer_id", "unknown")

        if self.use_simple_silu_ffn or self.w3 is None:
            weight_labels = "[w1, w2]"
            linears = (self.w1, self.w2)
        else:
            weight_labels = "[w1, w2, w3]"
            linears = (self.w1, self.w2, self.w3)

        logger.info(
            "Initializing MuP FeedForward (layer=%s) with std=%.6f for weights %s",
            layer_id,
            init_std,
            weight_labels,
        )
        for linear in linears:
            init_linear_weight(
                linear,
                init_std=init_std,
                init_type=self._hidden_init_type,
                scion_eps=self._disco_eps,
                trunc_normal_cutoff=self._trunc_normal_cutoff,
            )
        if self.mid_norm is not None:
            self.mid_norm.reset_parameters()


class TransformerBlock(BaseTransformerBlock):
    """Transformer block with attention and feed-forward layers with MuP configurations.

    Args:
        layer_id: Identifier for the layer (reserved for future use).
        model_args: Model configuration arguments.
    """

    def __init__(self, layer_id: int, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(layer_id, model_args)
        self.layer_id = layer_id
        self.model_args = model_args
        self.mup_config = model_args.mup_config_obj
        self.init_config = model_args.init_config_obj
        self._hidden_init_type = self.init_config.resolved_hidden_init(model_args.use_disco)
        # Override attention/feed-forward with MuP-aware variants
        self.attention = Attention(model_args)
        self.attention.mup_layer_id = layer_id
        self.feed_forward = FeedForward(model_args)
        self.feed_forward.mup_layer_id = layer_id

        elementwise_affine = model_args.torch_layernorm_elementwise_affine
        bias = model_args.torch_layernorm_bias if model_args.use_torch_layernorm else False

        self.attention_norm = _build_norm_module(
            model_args.dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=model_args.use_torch_layernorm,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )
        self.ffn_norm = _build_norm_module(
            model_args.dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=model_args.use_torch_layernorm,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

        self.use_peri_norm = model_args.use_peri_norm
        self.post_attn_norm: nn.Module | None = None
        self.post_ffn_norm: nn.Module | None = None
        if self.use_peri_norm:
            self.post_attn_norm = _build_norm_module(
                model_args.dim,
                eps=model_args.norm_eps,
                model_args=model_args,
                prefer_torch=model_args.use_torch_layernorm,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )
            self.post_ffn_norm = _build_norm_module(
                model_args.dim,
                eps=model_args.norm_eps,
                model_args=model_args,
                prefer_torch=model_args.use_torch_layernorm,
                elementwise_affine=elementwise_affine,
                bias=bias,
            )

        self.residual_scaling = 1.0
        if self.mup_config.completep_depth_alpha_enabled:
            self.residual_scaling = 1.0 / (
                self.mup_config.completep_depth_multiplier**self.mup_config.completep_depth_alpha_exp
            )
        logger.info(
            "Initialized MuP TransformerBlock(layer=%s) "
            "residual_scaling=%.6f, use_peri_norm=%s, completep_depth_alpha_enabled=%s, "
            "norm_type=%s, simple_silu_ffn=%s",
            layer_id,
            self.residual_scaling,
            self.use_peri_norm,
            self.mup_config.completep_depth_alpha_enabled,
            "LPLayerNorm" if model_args.use_torch_layernorm else "TitanRMSNorm",
            self.feed_forward.use_simple_silu_ffn,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor.
            freqs_cis: Precomputed frequency tensor for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor after attention and feed-forward layers.
        """
        attn_out = self.attention(self.attention_norm(x), freqs_cis)
        if self.post_attn_norm:
            attn_out = self.post_attn_norm(attn_out)

        scaling = self.residual_scaling if self.mup_config.mup_enabled else 1.0
        h = x + attn_out * scaling

        ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.post_ffn_norm:
            ffn_out = self.post_ffn_norm(ffn_out)

        return h + ffn_out * scaling

    def init_weights(self) -> None:
        """Initialize weights for the Transformer block."""
        super().init_weights()

        base_std = self.init_config.init_std or self.weight_init_std
        init_std = base_std
        if self.mup_config.mup_enabled and not self.model_args.use_scion:
            init_std = init_std / (self.mup_config.mup_width_multiplier**0.5)

        logger.info(
            "MuP TransformerBlock(layer=%s) weight init std=%.6f (base=%.6f, width_multiplier=%.6f, init_type=%s)",
            self.layer_id,
            init_std,
            base_std,
            self.mup_config.mup_width_multiplier,
            self._hidden_init_type,
        )

        self.attention.init_weights(init_std)
        self.feed_forward.init_weights(init_std)

        for norm in (self.post_attn_norm, self.post_ffn_norm):
            if norm is not None:
                norm.reset_parameters()


class Transformer(BaseTransformer):
    """Transformer model with Maximal Update Parametrization (MuP) support.

    This model implements the Transformer architecture with optional MuP scaling
    for improved training dynamics across different model widths.

    Args:
        model_args: Model configuration arguments.
    """

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(model_args)
        self.mup_config = model_args.mup_config_obj
        self.init_config = model_args.init_config_obj
        self._hidden_init_type = self.init_config.resolved_hidden_init(model_args.use_disco)
        self._embed_init_type = self.init_config.resolved_embed_init(model_args.use_disco)
        self._output_init_type = self.init_config.resolved_output_init(model_args.use_disco)
        self._trunc_normal_cutoff = self.init_config.trunc_normal_cutoff
        self._logged_bucket_assignments = False
        self._last_bucket_assignments: dict[str, str] = {}

        logger.info(
            "MuP Transformer configuration: enabled=%s, width_multiplier=%.6f, "
            "input_alpha=%.6f, output_alpha=%.6f, "
            "completep_depth_alpha_enabled=%s, depth_multiplier=%.6f, depth_alpha_exp=%.6f, "
            "eps_scaling_enabled=%s, disable_attention_scaling=%s, disable_hidden_lr_scaling=%s",
            self.mup_config.mup_enabled,
            self.mup_config.mup_width_multiplier,
            self.mup_config.mup_input_alpha,
            self.mup_config.mup_output_alpha,
            self.mup_config.completep_depth_alpha_enabled,
            self.mup_config.completep_depth_multiplier,
            self.mup_config.completep_depth_alpha_exp,
            self.mup_config.completep_eps_scaling_enabled,
            self.mup_config.mup_disable_attention_scaling,
            self.mup_config.mup_disable_hidden_lr_scaling,
        )
        logger.info(
            "MuP init configuration: init_std=%.6f, emb_init_std=%s, output_mult=%s, "
            "use_embedding_norm=%s, use_peri_norm=%s, tie_word_embeddings=%s",
            self.init_config.init_std,
            self.init_config.emb_init_std,
            self.init_config.output_mult,
            model_args.use_embedding_norm,
            model_args.use_peri_norm,
            model_args.tie_word_embeddings,
        )
        logger.info(
            "MuP Transformer architecture options: use_torch_layernorm=%s, use_simple_silu_ffn=%s",
            model_args.use_torch_layernorm,
            model_args.use_simple_silu_ffn,
        )
        logger.info(
            "MuP init type selection: hidden=%s, embed=%s, output=%s, trunc_cutoff=%.2e",
            self._hidden_init_type,
            self._embed_init_type,
            self._output_init_type,
            self._trunc_normal_cutoff,
        )

        # Embedding normalization and scaling
        if model_args.use_embedding_norm:
            emb_elementwise = model_args.torch_layernorm_elementwise_affine
            emb_bias = model_args.torch_layernorm_bias if model_args.use_torch_layernorm else False
            self.embedding_norm = _build_norm_module(
                model_args.dim,
                eps=model_args.norm_eps,
                model_args=model_args,
                prefer_torch=model_args.use_torch_layernorm,
                elementwise_affine=emb_elementwise,
                bias=emb_bias,
            )
        else:
            self.embedding_norm = None

        self.layers = nn.ModuleDict(
            {str(layer_id): TransformerBlock(layer_id, model_args) for layer_id in range(model_args.n_layers)}
        )
        self.norm = _build_norm_module(
            model_args.dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=model_args.use_torch_layernorm,
            elementwise_affine=model_args.torch_layernorm_elementwise_affine,
            bias=(model_args.torch_layernorm_bias if model_args.use_torch_layernorm else False),
        )
        if model_args.tie_word_embeddings:
            # Share embedding weights with the output projection when requested.
            self.output.weight = self.tok_embeddings.weight
        self.model_args = cast("TransformerModelArgsMuP", model_args)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Device to place buffers on. Defaults to freqs_cis device.
        """
        super().init_weights(buffer_device)

        init_std = self.init_config.init_std
        emb_init_std = self.init_config.emb_init_std or init_std

        logger.info(
            "MuP Transformer init_weights: buffer_device=%s, init_std=%.6f, emb_init_std=%.6f, tie_word_embeddings=%s",
            buffer_device,
            init_std,
            emb_init_std,
            self.model_args.tie_word_embeddings,
        )

        disco_eps = self.init_config.scion_init_eps
        trunc_cutoff = self._trunc_normal_cutoff
        embed_init_type = self._embed_init_type
        output_init_type = self._output_init_type

        if self.tok_embeddings is not None:
            initialize_tensor(
                self.tok_embeddings.weight,
                init_type=embed_init_type,
                init_std=emb_init_std,
                scion_eps=disco_eps,
                trunc_normal_cutoff=trunc_cutoff,
            )

        if self.embedding_norm is not None:
            self.embedding_norm.reset_parameters()

        self.norm.reset_parameters()

        if not self.model_args.tie_word_embeddings:
            initialize_tensor(
                self.output.weight,
                init_type=output_init_type,
                init_std=emb_init_std,
                scion_eps=disco_eps,
                trunc_normal_cutoff=trunc_cutoff,
            )
        else:
            if embed_init_type != output_init_type:
                logger.warning(
                    "tie_word_embeddings enabled, but embed_init=%s differs from output_init=%s; "
                    "using the embedding initialization for both.",
                    embed_init_type,
                    output_init_type,
                )
            self.output.weight = self.tok_embeddings.weight

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precompute rotary embeddings using the Qwen-style cache layout."""
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def _iter_trainable_params(self) -> list[tuple[str, Parameter]]:
        """Return trainable parameters with their qualified names."""
        return [(name, param) for name, param in self.named_parameters(remove_duplicate=True) if param.requires_grad]

    def _bucketize_parameters(self, param_entries: list[tuple[str, Parameter]]) -> dict[str, list[Parameter]]:
        """Group parameters according to MuP-specific update rules."""
        buckets: dict[str, list[Parameter]] = {
            "emb": [],
            "unembed": [],
            "hidden_ln": [],
            "decay_lr": [],
            "hidden_bias": [],
            "no_decay": [],
        }
        assignment_summary: dict[str, list[str]] = {bucket: [] for bucket in buckets}
        bucket_assignments: dict[str, str] = {}

        embed_suffixes = ["tok_embeddings.weight"]
        unembed_suffixes: list[str] = []
        if not self.model_args.tie_word_embeddings:
            unembed_suffixes.append("output.weight")

        hidden_ln_suffixes: list[str] = []

        def _extend_norm_suffixes(target: list[str], names: list[str]) -> None:
            for base in names:
                target.append(f"{base}.weight")
                target.append(f"{base}.bias")

        _extend_norm_suffixes(hidden_ln_suffixes, ["attention_norm", "ffn_norm"])
        if self.model_args.use_peri_norm:
            _extend_norm_suffixes(hidden_ln_suffixes, ["post_attn_norm", "post_ffn_norm"])
        if self.model_args.use_attention_value_norm:
            _extend_norm_suffixes(hidden_ln_suffixes, ["attention.v_norm"])
        if self.model_args.use_attention_output_norm:
            _extend_norm_suffixes(hidden_ln_suffixes, ["attention.o_norm"])
        if self.model_args.use_mlp_mid_norm:
            _extend_norm_suffixes(hidden_ln_suffixes, ["feed_forward.mid_norm"])

        no_decay_suffixes: list[str] = []
        _extend_norm_suffixes(no_decay_suffixes, ["embedding_norm", "norm"])
        decay_weight_suffixes = [
            "wq.weight",
            "wk.weight",
            "wv.weight",
            "wo.weight",
            "w1.weight",
            "w2.weight",
            "w3.weight",
        ]

        for name, param in param_entries:
            bucket_key = self._resolve_bucket_name(
                name,
                embed_suffixes,
                unembed_suffixes,
                hidden_ln_suffixes,
                no_decay_suffixes,
                decay_weight_suffixes,
            )
            if bucket_key is None:
                target_bucket = "decay_lr" if name.endswith(".weight") else "no_decay"
                buckets[target_bucket].append(param)
                assignment_summary[target_bucket].append(name)
                bucket_assignments[name] = target_bucket
            else:
                buckets[bucket_key].append(param)
                assignment_summary[bucket_key].append(name)
                bucket_assignments[name] = bucket_key

        self._last_bucket_assignments = bucket_assignments
        if not self._logged_bucket_assignments:
            for bucket, names in assignment_summary.items():
                if names:
                    logger.info(
                        "MuP bucket '%s' assigned %d parameter(s): %s",
                        bucket,
                        len(names),
                        ", ".join(sorted(names)),
                    )
            self._logged_bucket_assignments = True

        return buckets

    def _resolve_bucket_name(
        self,
        name: str,
        embed_suffixes: list[str],
        unembed_suffixes: list[str],
        hidden_ln_suffixes: list[str],
        no_decay_suffixes: list[str],
        decay_weight_suffixes: list[str],
    ) -> str | None:
        """Return the MuP bucket identifier for a parameter name."""
        if any(name.endswith(suffix) for suffix in embed_suffixes):
            return "emb"
        if any(name.endswith(suffix) for suffix in unembed_suffixes):
            return "unembed"
        if any(name.endswith(suffix) for suffix in hidden_ln_suffixes):
            return "hidden_ln"
        if name.endswith(".bias"):
            return "hidden_bias"
        if any(name.endswith(suffix) for suffix in no_decay_suffixes):
            return "no_decay"
        if any(name.endswith(suffix) for suffix in decay_weight_suffixes):
            return "decay_lr"
        return None

    def _validate_bucket_counts(self, total_params: int, buckets: dict[str, list[Parameter]]) -> None:
        """Ensure all trainable parameters are accounted for in MuP buckets."""
        total_bucketed = sum(len(values) for values in buckets.values())
        if total_bucketed != total_params:
            msg = (
                "MuP optimizer grouping failed to account for all parameters. "
                f"Expected {total_params}, got {total_bucketed}."
            )
            raise RuntimeError(msg)

    def _compute_lr_scaling(self) -> tuple[float, float]:
        """Return width and depth scaling factors for MuP updates."""
        if self.model_args.use_scion:
            width_lr_scaling = 1.0
        else:
            width_lr_scaling = 1.0 / self.mup_config.mup_width_multiplier
        depth_lr_scaling = 1.0
        if self.mup_config.completep_depth_alpha_enabled:
            depth_lr_scaling = self.mup_config.completep_depth_multiplier ** (
                self.mup_config.completep_depth_alpha_exp - 1.0
            )
        logger.info(
            "MuP LR scaling computed: width_lr_scaling=%.6f, depth_lr_scaling=%.6f",
            width_lr_scaling,
            depth_lr_scaling,
        )
        return width_lr_scaling, depth_lr_scaling

    def _resolve_optimizer_eps(
        self,
        eps: float,
        *,
        width_lr_scaling: float,
    ) -> float:
        """Return MuP-adjusted epsilon when CompleteP scaling is enabled."""
        if not self.mup_config.completep_eps_scaling_enabled:
            logger.info("MuP epsilon scaling disabled; using base eps=%.6f", eps)
            return eps

        depth_eps_scaling = self.mup_config.completep_depth_multiplier ** (
            -1.0 * self.mup_config.completep_depth_alpha_exp
        )
        adjusted_eps = eps * width_lr_scaling * depth_eps_scaling
        logger.info(
            "MuP epsilon scaling applied: base_eps=%.6f, width_lr_scaling=%.6f, "
            "depth_eps_scaling=%.6f, adjusted_eps=%.6f",
            eps,
            width_lr_scaling,
            depth_eps_scaling,
            adjusted_eps,
        )
        return adjusted_eps

    def _build_param_groups(
        self,
        buckets: dict[str, list[Parameter]],
        *,
        base_lr: float,
        weight_decay: float,
        width_lr_scaling: float,
        depth_lr_scaling: float,
        scion_hidden_scale: float | None = None,
        scion_output_scale: float | None = None,
        scion_hidden_norm: str | None = None,
        scion_output_norm: str | None = None,
        scion_hidden_norm_kwargs: dict[str, Any] | None = None,
        scion_output_norm_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Construct optimizer parameter groups based on MuP buckets."""
        group_specs: list[tuple[str, dict[str, Any]]] = [
            (
                "emb",
                {"params": buckets["emb"], "weight_decay": weight_decay, "lr": base_lr},
            ),
            (
                "unembed",
                {"params": buckets["unembed"], "weight_decay": weight_decay, "lr": base_lr},
            ),
            (
                "hidden_ln",
                {
                    "params": buckets["hidden_ln"],
                    "weight_decay": 0.0,
                    "lr": base_lr * depth_lr_scaling,
                },
            ),
            (
                "decay_lr",
                {
                    "params": buckets["decay_lr"],
                    "weight_decay": weight_decay / width_lr_scaling,
                    "lr": base_lr * width_lr_scaling * depth_lr_scaling,
                },
            ),
            (
                "hidden_bias",
                {
                    "params": buckets["hidden_bias"],
                    "weight_decay": 0.0,
                    "lr": base_lr * depth_lr_scaling,
                },
            ),
            (
                "no_decay",
                {"params": buckets["no_decay"], "weight_decay": 0.0, "lr": base_lr},
            ),
        ]

        filtered_groups: list[dict[str, Any]] = []
        filtered_labels: list[str] = []
        for label, group in group_specs:
            if group["params"]:
                filtered_groups.append(group)
                filtered_labels.append(label)

        param_occurrences: dict[int, list[str]] = {}
        for label, group in zip(filtered_labels, filtered_groups, strict=True):
            for param in group["params"]:
                bucket_hits = param_occurrences.setdefault(id(param), [])
                bucket_hits.append(label)

        duplicate_conflicts = [
            "/".join(sorted(set(labels))) if len(set(labels)) > 1 else labels[0]
            for labels in param_occurrences.values()
            if len(labels) > 1
        ]
        if duplicate_conflicts:
            conflicts = ", ".join(duplicate_conflicts)
            msg = (
                "MuP optimizer grouping assigned at least one parameter to multiple param groups. "
                f"Conflicts detected for bucket(s): {conflicts}."
            )
            raise ValueError(msg)

        assigned_params = {p for group in filtered_groups for p in group["params"]}
        unassigned = {param for bucket in buckets.values() for param in bucket if param not in assigned_params}
        if unassigned:
            msg = (
                f"MuP optimizer grouping left {len(unassigned)} parameters without a param group. "
                "This indicates a mismatch between bucket definitions and grouping logic."
            )
            raise ValueError(msg)

        if self.model_args.use_scion:
            if self.model_args.use_disco:
                self._apply_disco_norm_overrides(filtered_labels, filtered_groups)
            else:
                self._apply_scion_scales(
                    filtered_labels,
                    filtered_groups,
                    hidden_scale=scion_hidden_scale,
                    output_scale=scion_output_scale,
                    hidden_norm=scion_hidden_norm,
                    output_norm=scion_output_norm,
                    hidden_norm_kwargs=scion_hidden_norm_kwargs,
                    output_norm_kwargs=scion_output_norm_kwargs,
                )

        for label, group in zip(filtered_labels, filtered_groups, strict=True):
            param_count = sum(param.numel() for param in group["params"])
            bucket_param_names = sorted(
                name for name, bucket in self._last_bucket_assignments.items() if bucket == label
            )
            logger.info(
                "MuP optimizer param group '%s': %d tensors, %d parameters, lr=%.6f, weight_decay=%.6f, params=[%s]",
                label,
                len(group["params"]),
                param_count,
                group["lr"],
                group["weight_decay"],
                ", ".join(bucket_param_names),
            )

        return filtered_groups

    def _apply_scion_scales(
        self,
        labels: Sequence[str],
        groups: Sequence[dict[str, Any]],
        *,
        hidden_scale: float | None,
        output_scale: float | None,
        hidden_norm: str | None,
        output_norm: str | None,
        hidden_norm_kwargs: dict[str, Any] | None,
        output_norm_kwargs: dict[str, Any] | None,
    ) -> None:
        """Attach classic Scion radii to embedding vs hidden parameter groups."""
        resolved_hidden_scale = (
            float(hidden_scale) if hidden_scale is not None else float(self.model_args.scion_hidden_scale)
        )
        resolved_output_scale = (
            float(output_scale) if output_scale is not None else float(self.model_args.scion_output_scale)
        )
        resolved_hidden_norm = hidden_norm or "spectral"
        resolved_output_norm = output_norm or "sign"
        base_hidden_kwargs = dict(hidden_norm_kwargs or {})
        base_output_kwargs = dict(output_norm_kwargs or {})
        if resolved_hidden_norm.lower() == "spectral":
            base_hidden_kwargs.setdefault("backend", "newtonschulz5")
            base_hidden_kwargs.setdefault("backend_steps", 5)
            base_hidden_kwargs.setdefault("normalized", True)
        if resolved_output_norm.lower() == "sign":
            base_output_kwargs.setdefault("normalized", True)
        default_output_norm_per_bucket = {
            "emb": resolved_output_norm,
            "unembed": resolved_output_norm,
        }

        for label, group in zip(labels, groups, strict=True):
            is_embed = label in {"emb", "unembed"}
            scale = resolved_output_scale if is_embed else resolved_hidden_scale
            if is_embed:
                norm_name = default_output_norm_per_bucket.get(label, resolved_output_norm)
                norm_kwargs_source = base_output_kwargs
            else:
                norm_name = resolved_hidden_norm
                norm_kwargs_source = base_hidden_kwargs

            group.setdefault("scale", scale)
            if "norm" not in group or group["norm"] is None:
                group["norm"] = norm_name
                group["norm_kwargs"] = dict(norm_kwargs_source)
            elif "norm_kwargs" not in group or group["norm_kwargs"] is None:
                group["norm_kwargs"] = dict(norm_kwargs_source)
            if _SCION_SCALE_DEBUG_ENABLED:
                params = group.get("params", [])
                param_count = sum(param.numel() for param in params) if params else 0
                bucket_param_names = sorted(
                    name for name, bucket in self._last_bucket_assignments.items() if bucket == label
                )[:5]
                logger.info(
                    "Scion scale debug: bucket=%s scale=%.6f norm=%s tensors=%d params=%d sample_params=%s",
                    label,
                    scale,
                    group.get("norm", "<unset>"),
                    len(params),
                    param_count,
                    bucket_param_names or "n/a",
                )

    def _apply_disco_norm_overrides(
        self,
        labels: Sequence[str],
        groups: Sequence[dict[str, Any]],
    ) -> None:
        """Attach per-bucket Disco norms for embeddings vs hidden layers."""
        embed_norm = "embed_linear"
        unembed_override: tuple[str, dict[str, Any]] | None = None
        if not self.model_args.tie_word_embeddings:
            embed_norm = "embed_sqrt"
            unembed_override = ("unembed_sqrt", {"backend": "identity", "backend_steps": 0})

        norm_overrides: dict[str, tuple[str, dict[str, Any]]] = {
            "emb": (embed_norm, {"backend": "identity", "backend_steps": 0}),
            "decay_lr": ("spectral", {"backend": "newtonschulz5", "backend_steps": 5}),
        }
        if unembed_override is not None:
            norm_overrides["unembed"] = unembed_override
        for label, group in zip(labels, groups, strict=True):
            override = norm_overrides.get(label)
            if override is None:
                continue
            norm_name, norm_kwargs = override
            group["norm"] = norm_name
            group["norm_kwargs"] = dict(norm_kwargs)
        if _DISCO_NORM_DEBUG_ENABLED:
            self._log_disco_norm_assignments(labels, groups)

    def _log_disco_norm_assignments(
        self,
        labels: Sequence[str],
        groups: Sequence[dict[str, Any]],
    ) -> None:
        """Emit debug info describing Disco norm choices for each bucket."""
        for label, group in zip(labels, groups, strict=True):
            norm_name = group.get("norm", "<unset>")
            params = group.get("params", [])
            param_count = sum(param.numel() for param in params) if params else 0
            bucket_param_names = sorted(
                name for name, bucket in self._last_bucket_assignments.items() if bucket == label
            )[:5]
            logger.info(
                "Disco norm debug: bucket=%s norm=%s tensors=%d params=%d sample_params=%s",
                label,
                norm_name,
                len(params),
                param_count,
                bucket_param_names or "n/a",
            )

    def build_mup_optimizer_overrides(
        self,
        *,
        lr: float,
        eps: float,
        weight_decay: float,
        scion_hidden_scale: float | None = None,
        scion_output_scale: float | None = None,
        scion_hidden_norm: str | None = None,
        scion_output_norm: str | None = None,
        scion_hidden_norm_kwargs: dict[str, Any] | None = None,
        scion_output_norm_kwargs: dict[str, Any] | None = None,
    ) -> MuPOptimizerOverride | None:
        """Compute MuP optimizer overrides without mutating caller state."""
        if not (self.mup_config.mup_enabled and not self.mup_config.mup_disable_hidden_lr_scaling):
            logger.info(
                "MuP optimizer overrides skipped: enabled=%s, disable_hidden_lr_scaling=%s",
                self.mup_config.mup_enabled,
                self.mup_config.mup_disable_hidden_lr_scaling,
            )
            return None

        param_entries = self._iter_trainable_params()
        buckets = self._bucketize_parameters(param_entries)
        self._validate_bucket_counts(len(param_entries), buckets)

        width_lr_scaling, depth_lr_scaling = self._compute_lr_scaling()
        adjusted_eps = self._resolve_optimizer_eps(
            eps,
            width_lr_scaling=width_lr_scaling,
        )

        param_groups = self._build_param_groups(
            buckets,
            base_lr=lr,
            weight_decay=weight_decay,
            width_lr_scaling=width_lr_scaling,
            depth_lr_scaling=depth_lr_scaling,
            scion_hidden_scale=scion_hidden_scale,
            scion_output_scale=scion_output_scale,
            scion_hidden_norm=scion_hidden_norm,
            scion_output_norm=scion_output_norm,
            scion_hidden_norm_kwargs=scion_hidden_norm_kwargs,
            scion_output_norm_kwargs=scion_output_norm_kwargs,
        )

        config_updates: dict[str, Any] = {}
        if adjusted_eps != eps:
            config_updates["eps"] = adjusted_eps

        logger.info(
            "MuP optimizer overrides prepared: %d buckets with params, config_updates=%s",
            sum(1 for bucket in buckets.values() if bucket),
            config_updates,
        )

        return MuPOptimizerOverride(
            param_groups=param_groups or None,
            config_updates=config_updates,
        )

    def get_optimizer_param_groups(
        self, optimizer_config: dict[str, Any]
    ) -> tuple[Iterator[Parameter] | list[dict[str, Any]], dict[str, Any]]:
        """Get optimizer parameter groups with MuP-specific learning rates."""
        overrides = self.build_mup_optimizer_overrides(
            lr=optimizer_config["lr"],
            eps=optimizer_config.get("eps", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
            scion_hidden_scale=optimizer_config.get("scion_hidden_scale"),
            scion_output_scale=optimizer_config.get("scion_output_scale"),
            scion_hidden_norm=optimizer_config.get("scion_hidden_norm"),
            scion_output_norm=optimizer_config.get("scion_output_norm"),
            scion_hidden_norm_kwargs=optimizer_config.get("scion_hidden_norm_kwargs"),
            scion_output_norm_kwargs=optimizer_config.get("scion_output_norm_kwargs"),
        )

        if overrides is None:
            return self.parameters(), optimizer_config

        updated_config = dict(optimizer_config)
        updated_config.update(overrides.config_updates)

        if overrides.param_groups is None:
            return self.parameters(), updated_config

        return overrides.param_groups, updated_config

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass through the Transformer model.

        Args:
            tokens: Input token indices.
            input_batch: Optional input batch for document masking (unused in this implementation).

        Returns:
            torch.Tensor: Output logits.
        """
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Apply embedding normalization and scaling
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)
        apply_mup_scaling = self.mup_config.mup_enabled and not self.model_args.use_scion
        if apply_mup_scaling:
            h = h * self.mup_config.mup_input_alpha

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h

        if apply_mup_scaling:
            h = h * (self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier)

        logits = self.output(h)
        if self.init_config.output_mult is not None:
            logits = logits * self.init_config.output_mult

        # Always use self.output (nn.Linear) for DTensor compatibility.
        # When weight tying is enabled, output.weight is the same object as tok_embeddings.weight.
        return logits
