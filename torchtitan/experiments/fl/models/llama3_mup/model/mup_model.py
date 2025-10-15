# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
"""Model components for Llama-3 MuP."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast, Protocol, runtime_checkable

import torch
from torch import nn
from torch.nn.parameter import Parameter

# Import reusable components from the base llama3 model
from torchtitan.models.llama3.model.model import (
    Attention as BaseAttention,
    FeedForward as BaseFeedForward,
    Transformer as BaseTransformer,
    TransformerBlock as BaseTransformerBlock,
)

from .mup_args import TransformerModelArgs as TransformerModelArgsMuP


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
    ) -> torch.Tensor:
        return self.inner(q, k, v, scale=self.scale)


class Attention(BaseAttention):
    """Multi-head attention layer with MuP-specific weight initialization."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(model_args)
        self.mup_config = model_args.mup_config_obj
        if (
            self.mup_config.mup_enabled
            and not self.mup_config.mup_disable_attention_scaling
        ):
            scale = 1.0 / float(self.head_dim)
            self.sdpa = _MuPScaledAttention(self.sdpa, scale)

    def init_weights(self, init_std: float) -> None:
        """Initialize weights with MuP-specific scaling.

        Args:
            init_std (float): Standard deviation for weight initialization.
        """
        for linear in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)


class FeedForward(BaseFeedForward):
    """Feed-forward network with MuP-specific weight initialization."""

    def init_weights(self, init_std: float) -> None:
        """Initialize weights with MuP-specific scaling.

        Args:
            init_std (float): Standard deviation for weight initialization.

        """
        for linear in (self.w1, self.w2, self.w3):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(BaseTransformerBlock):
    """Transformer block with attention and feed-forward layers with MuP configurations.

    Args:
        layer_id: Identifier for the layer (reserved for future use).
        model_args: Model configuration arguments.
    """

    def __init__(self, layer_id: int, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(layer_id, model_args)
        self.model_args = model_args
        self.mup_config = model_args.mup_config_obj
        self.init_config = model_args.init_config_obj

        # Override attention/feed-forward with MuP-aware variants
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )

        self.use_peri_norm = model_args.use_peri_norm
        self.post_attn_norm: nn.RMSNorm | None = None
        self.post_ffn_norm: nn.RMSNorm | None = None
        if self.use_peri_norm:
            self.post_attn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.residual_scaling = 1.0
        if self.mup_config.completep_depth_alpha_enabled:
            self.residual_scaling = 1.0 / (
                self.mup_config.completep_depth_multiplier
                ** self.mup_config.completep_depth_alpha_exp
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

        init_std = self.init_config.init_std or self.weight_init_std
        if self.mup_config.mup_enabled:
            init_std = init_std / (self.mup_config.mup_width_multiplier**0.5)

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

        # Embedding normalization and scaling
        self.embedding_norm = (
            nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
            if model_args.use_embedding_norm
            else None
        )
        if self.embedding_norm is not None and self.tok_embeddings is not None:
            # Expose embedding norm via the embedding for compatibility with tests/utilities.
            self.tok_embeddings.norm = self.embedding_norm  # type: ignore[attr-defined]
        self.layers = nn.ModuleDict(
            {
                str(layer_id): TransformerBlock(layer_id, model_args)
                for layer_id in range(model_args.n_layers)
            }
        )
        if (
            model_args.tie_word_embeddings
            and self.output is not None
            and self.tok_embeddings is not None
        ):
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

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=emb_init_std)

        if self.embedding_norm is not None:
            self.embedding_norm.reset_parameters()

        if not self.model_args.tie_word_embeddings:
            final_out_std = (self.model_args.dim**-0.5) * (
                self.init_config.output_mult or 1.0
            )
            nn.init.normal_(self.output.weight, mean=0.0, std=final_out_std)

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
            else:
                buckets[bucket_key].append(param)

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
        return width_lr_scaling, depth_lr_scaling

    def _resolve_optimizer_eps(
        self,
        eps: float,
        *,
        width_lr_scaling: float,
    ) -> float:
        """Return MuP-adjusted epsilon when CompleteP scaling is enabled."""
        if not self.mup_config.completep_eps_scaling_enabled:
            return eps

        depth_eps_scaling = self.mup_config.completep_depth_multiplier ** (
            -1.0 * self.mup_config.completep_depth_alpha_exp
        )
        return eps * width_lr_scaling * depth_eps_scaling

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

        filtered_groups = [group for group in param_groups if group["params"]]

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

        # Always use self.output (nn.Linear) for DTensor compatibility
        # When weight tying is enabled, output.weight is the same object as tok_embeddings.weight
        output = self.output(h) if self.output else h

        if self.mup_config.mup_enabled:
            output = output * (
                self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier
            )
        return output
