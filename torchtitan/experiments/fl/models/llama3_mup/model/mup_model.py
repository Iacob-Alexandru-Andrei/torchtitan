# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
"""Model components for Llama-3 MuP."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

# Import reusable components from the base llama3 model
from torchtitan.experiments.qwen3.model.model import (
    Attention as QwenAttention,
    precompute_rope_cache,
)
from torchtitan.experiments.fl.models.llama3_mup.model.rms_norm import build_norm_module
from torchtitan.models.llama3.model.model import (
    FeedForward as BaseFeedForward,
    Transformer as BaseTransformer,
    TransformerBlock as BaseTransformerBlock,
)

from .mup_args import TransformerModelArgs as TransformerModelArgsMuP


logger = logging.getLogger(__name__)


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


def _prefer_torch_layernorm(model_args: "TransformerModelArgsMuP") -> bool:
    """Resolve whether to prefer torch LayerNorm-style layers."""
    if model_args.layernorm_impl is not None:
        return model_args.layernorm_impl == "torch"
    return model_args.use_torch_layernorm


def _prefer_torch_qk_layernorm(model_args: "TransformerModelArgsMuP") -> bool:
    """Resolve whether to prefer torch LayerNorm for QK norms."""
    if model_args.qk_layernorm_impl is not None:
        return model_args.qk_layernorm_impl == "torch"
    if model_args.layernorm_impl is not None:
        return model_args.layernorm_impl == "torch"
    return model_args.use_torch_layernorm


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
        self.mup_config = model_args.mup_config_obj
        if model_args.qk_norm and self.q_norm is not None and self.k_norm is not None:
            self.q_norm = self._build_head_norm(model_args)
            self.k_norm = self._build_head_norm(model_args)
            logger.info(
                "MuP QK head normalization enabled: head_dim=%d, norm_type=%s",
                self.head_dim,
                self.q_norm.__class__.__name__,
            )
        if (
            self.mup_config.mup_enabled
            and not self.mup_config.mup_disable_attention_scaling
        ):
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
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.k_norm):
            if norm is not None:
                norm.reset_parameters()

    def _build_head_norm(self, model_args: TransformerModelArgsMuP) -> nn.Module:
        prefer_torch = _prefer_torch_qk_layernorm(model_args)
        return build_norm_module(
            self.head_dim,
            eps=model_args.norm_eps,
            prefer_torch=prefer_torch,
            elementwise_affine=model_args.qk_norm_elementwise_affine,
            bias=model_args.qk_norm_bias if prefer_torch else False,
            torch_norm_cls=LPLayerNorm,
            add_unit_offset=model_args.qk_norm_elementwise_affine,
            force_bf16=model_args.force_rmsnorm_bf16,
        )


class FeedForward(BaseFeedForward):
    """Feed-forward network with MuP-specific weight initialization."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
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
        self.use_simple_silu_ffn = model_args.use_simple_silu_ffn
        if self.use_simple_silu_ffn:
            # Drop the unused gated branch when configured for a simple MLP.
            self.w3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply either gated or simple SiLU feed-forward projection."""
        up = F.silu(self.w1(x))
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
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)


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
        prefer_torch_norm = _prefer_torch_layernorm(model_args)

        # Override attention/feed-forward with MuP-aware variants
        self.attention = Attention(model_args)
        self.attention.mup_layer_id = layer_id
        self.feed_forward = FeedForward(model_args)
        self.feed_forward.mup_layer_id = layer_id

        def _build_block_norm() -> nn.Module:
            return build_norm_module(
                model_args.dim,
                eps=model_args.norm_eps,
                prefer_torch=prefer_torch_norm,
                elementwise_affine=model_args.torch_layernorm_elementwise_affine,
                bias=model_args.torch_layernorm_bias if prefer_torch_norm else False,
                torch_norm_cls=LPLayerNorm,
                add_unit_offset=model_args.torch_layernorm_elementwise_affine,
                force_bf16=model_args.force_rmsnorm_bf16,
            )

        self.attention_norm = _build_block_norm()
        self.ffn_norm = _build_block_norm()

        self.use_peri_norm = model_args.use_peri_norm
        self.post_attn_norm: nn.Module | None = None
        self.post_ffn_norm: nn.Module | None = None
        if self.use_peri_norm:
            self.post_attn_norm = _build_block_norm()
            self.post_ffn_norm = _build_block_norm()

        self.residual_scaling = 1.0
        if self.mup_config.completep_depth_alpha_enabled:
            self.residual_scaling = 1.0 / (
                self.mup_config.completep_depth_multiplier
                ** self.mup_config.completep_depth_alpha_exp
            )
        logger.info(
            "Initialized MuP TransformerBlock(layer=%s) "
            "residual_scaling=%.6f, use_peri_norm=%s, completep_depth_alpha_enabled=%s, "
            "norm_type=%s, simple_silu_ffn=%s",
            layer_id,
            self.residual_scaling,
            self.use_peri_norm,
            self.mup_config.completep_depth_alpha_enabled,
            type(self.attention_norm).__name__,
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
        if self.mup_config.mup_enabled:
            init_std = init_std / (self.mup_config.mup_width_multiplier**0.5)

        logger.info(
            "MuP TransformerBlock(layer=%s) weight init std=%.6f (base=%.6f, width_multiplier=%.6f)",
            self.layer_id,
            init_std,
            base_std,
            self.mup_config.mup_width_multiplier,
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
        self._logged_bucket_assignments = False
        self._last_bucket_assignments: dict[str, str] = {}
        resolved_norm_impl = "torch" if prefer_torch_norm else "rms"

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
            "MuP Transformer architecture options: use_torch_layernorm=%s, norm_impl=%s, use_simple_silu_ffn=%s",
            model_args.use_torch_layernorm,
            resolved_norm_impl,
            model_args.use_simple_silu_ffn,
        )

        prefer_torch_norm = _prefer_torch_layernorm(model_args)

        # Embedding normalization and scaling
        if model_args.use_embedding_norm:
            self.embedding_norm = build_norm_module(
                model_args.dim,
                eps=model_args.norm_eps,
                prefer_torch=prefer_torch_norm,
                elementwise_affine=model_args.torch_layernorm_elementwise_affine,
                bias=model_args.torch_layernorm_bias if prefer_torch_norm else False,
                torch_norm_cls=LPLayerNorm,
                add_unit_offset=model_args.torch_layernorm_elementwise_affine,
                force_bf16=model_args.force_rmsnorm_bf16,
            )
        else:
            self.embedding_norm = None

        self.layers = nn.ModuleDict(
            {
                str(layer_id): TransformerBlock(layer_id, model_args)
                for layer_id in range(model_args.n_layers)
            }
        )
        self.norm = build_norm_module(
            model_args.dim,
            eps=model_args.norm_eps,
            prefer_torch=prefer_torch_norm,
            elementwise_affine=model_args.torch_layernorm_elementwise_affine,
            bias=model_args.torch_layernorm_bias if prefer_torch_norm else False,
            torch_norm_cls=LPLayerNorm,
            add_unit_offset=model_args.torch_layernorm_elementwise_affine,
            force_bf16=model_args.force_rmsnorm_bf16,
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

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=emb_init_std)

        if self.embedding_norm is not None:
            self.embedding_norm.reset_parameters()

        self.norm.reset_parameters()

        if not self.model_args.tie_word_embeddings:
            nn.init.normal_(self.output.weight, mean=0.0, std=emb_init_std)
        else:
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
        return [
            (name, param)
            for name, param in self.named_parameters(remove_duplicate=True)
            if param.requires_grad
        ]

    def _bucketize_parameters(
        self, param_entries: list[tuple[str, Parameter]]
    ) -> dict[str, list[Parameter]]:
        """Group parameters according to MuP-specific update rules."""
        buckets: dict[str, list[Parameter]] = {
            "emb": [],
            "hidden_ln": [],
            "decay_lr": [],
            "hidden_bias": [],
            "no_decay": [],
        }
        assignment_summary: dict[str, list[str]] = {bucket: [] for bucket in buckets}
        bucket_assignments: dict[str, str] = {}

        embed_suffixes = ["tok_embeddings.weight"]
        if not self.model_args.tie_word_embeddings:
            embed_suffixes.append("output.weight")

        hidden_ln_suffixes = ["attention_norm.weight", "ffn_norm.weight"]
        if self.model_args.use_peri_norm:
            hidden_ln_suffixes.extend(["post_attn_norm.weight", "post_ffn_norm.weight"])

        no_decay_suffixes = ["embedding_norm.weight", "norm.weight"]
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
        hidden_ln_suffixes: list[str],
        no_decay_suffixes: list[str],
        decay_weight_suffixes: list[str],
    ) -> str | None:
        """Return the MuP bucket identifier for a parameter name."""
        if any(name.endswith(suffix) for suffix in embed_suffixes):
            return "emb"
        if any(name.endswith(suffix) for suffix in hidden_ln_suffixes):
            return "hidden_ln"
        if name.endswith(".bias"):
            return "hidden_bias"
        if any(name.endswith(suffix) for suffix in no_decay_suffixes):
            return "no_decay"
        if any(name.endswith(suffix) for suffix in decay_weight_suffixes):
            return "decay_lr"
        return None

    def _validate_bucket_counts(
        self, total_params: int, buckets: dict[str, list[Parameter]]
    ) -> None:
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
    ) -> list[dict[str, Any]]:
        """Construct optimizer parameter groups based on MuP buckets."""
        param_groups = [
            {"params": buckets["emb"], "weight_decay": weight_decay, "lr": base_lr},
            {
                "params": buckets["hidden_ln"],
                "weight_decay": 0.0,
                "lr": base_lr * depth_lr_scaling,
            },
            {
                "params": buckets["decay_lr"],
                "weight_decay": weight_decay / width_lr_scaling,
                "lr": base_lr * width_lr_scaling * depth_lr_scaling,
            },
            {
                "params": buckets["hidden_bias"],
                "weight_decay": 0.0,
                "lr": base_lr * depth_lr_scaling,
            },
            {"params": buckets["no_decay"], "weight_decay": 0.0, "lr": base_lr},
        ]

        group_labels = ["emb", "hidden_ln", "decay_lr", "hidden_bias", "no_decay"]

        filtered_groups: list[dict[str, Any]] = []
        filtered_labels: list[str] = []
        for group, label in zip(param_groups, group_labels, strict=True):
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
        unassigned = {
            param
            for bucket in buckets.values()
            for param in bucket
            if param not in assigned_params
        }
        if unassigned:
            msg = (
                f"MuP optimizer grouping left {len(unassigned)} parameters without a param group. "
                "This indicates a mismatch between bucket definitions and grouping logic."
            )
            raise ValueError(msg)

        for label, group in zip(filtered_labels, filtered_groups, strict=True):
            param_count = sum(param.numel() for param in group["params"])
            bucket_param_names = sorted(
                name
                for name, bucket in self._last_bucket_assignments.items()
                if bucket == label
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

    def build_mup_optimizer_overrides(
        self,
        *,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> MuPOptimizerOverride | None:
        """Compute MuP optimizer overrides without mutating caller state."""
        if not (
            self.mup_config.mup_enabled
            and not self.mup_config.mup_disable_hidden_lr_scaling
        ):
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
        if self.mup_config.mup_enabled:
            h = h * self.mup_config.mup_input_alpha

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h

        if self.mup_config.mup_enabled:
            h = h * (
                self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier
            )

        # Always use self.output (nn.Linear) for DTensor compatibility
        # When weight tying is enabled, output.weight is the same object as tok_embeddings.weight
        return self.output(h)
