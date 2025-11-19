# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
"""Model components for Llama-3 MuP with optional Disco extensions."""

import logging
import math
import os
from collections.abc import Iterator, Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import (
    Attention as MuPAttention,
    FeedForward as MuPFeedForward,
    LPLayerNorm,
    MuPOptimizerOverride,
    SupportsMuPOptimizerOverrides,
    Transformer as MuPTransformer,
    TransformerBlock as MuPTransformerBlock,
)

from .disco_init import init_linear_weight, initialize_tensor
from .mup_args import TransformerModelArgs as TransformerModelArgsMuP

logger = logging.getLogger(__name__)
_debug_env = os.getenv("TORCHTITAN_DEBUG_DISCO_NORMS")
if _debug_env is None:
    _debug_env = os.getenv("TORCHTITAN_DEBUG_SCION_NORMS", "")
_DISCO_NORM_DEBUG_ENABLED = _debug_env.lower() in {"1", "true", "yes", "on"}
_SCION_SCALE_DEBUG_ENABLED = os.getenv("TORCHTITAN_DEBUG_SCION_SCALES", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class TitanRMSNorm(nn.Module):
    """RMSNorm variant that can optionally train an additive offset."""

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
        self._norm_axes = tuple(range(-len(self.normalized_shape), 0))

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


class Attention(MuPAttention):
    """MuP attention layer with optional Disco initialization and norms."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(model_args)
        init_config = model_args.init_config_obj
        self._disco_eps = init_config.scion_init_eps
        self._hidden_init_type = init_config.resolved_hidden_init(model_args.use_disco)
        self._trunc_normal_cutoff = init_config.trunc_normal_cutoff
        self._use_custom_init = model_args.use_disco or init_config.hidden_init is not None

    def _build_head_norm(self, model_args: TransformerModelArgsMuP) -> nn.Module:
        prefer_torch = model_args.use_torch_qk_layernorm
        return _build_norm_module(
            self.head_dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=prefer_torch,
            elementwise_affine=model_args.qk_norm_elementwise_affine,
            bias=model_args.qk_norm_bias,
        )

    def init_weights(self, init_std: float) -> None:
        if not self._use_custom_init:
            super().init_weights(init_std)
            return

        layer_id = getattr(self, "mup_layer_id", "unknown")
        logger.info(
            "Initializing Disco Attention (layer=%s) with std=%.6f for weights [wq, wk, wv, wo]",
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
        for norm in (self.q_norm, self.k_norm):
            if norm is not None:
                norm.reset_parameters()


class FeedForward(MuPFeedForward):
    """Feed-forward block with optional Disco activation scaling/init."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        self._activation_scale = math.sqrt(2.0) if model_args.use_disco else 1.0
        init_config = model_args.init_config_obj
        self._disco_eps = init_config.scion_init_eps
        self._hidden_init_type = init_config.resolved_hidden_init(model_args.use_disco)
        self._trunc_normal_cutoff = init_config.trunc_normal_cutoff
        self._use_custom_init = model_args.use_disco or init_config.hidden_init is not None
        super().__init__(model_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.silu(self.w1(x))
        if self._activation_scale != 1.0:
            up = up * self._activation_scale
        if self.use_simple_silu_ffn or self.w3 is None:
            return self.w2(up)
        gate = self.w3(x)
        return self.w2(up * gate)

    def init_weights(self, init_std: float) -> None:
        if not self._use_custom_init:
            super().init_weights(init_std)
            return

        layer_id = getattr(self, "mup_layer_id", "unknown")
        if self.use_simple_silu_ffn or self.w3 is None:
            weight_labels = "[w1, w2]"
            linears = (self.w1, self.w2)
        else:
            weight_labels = "[w1, w2, w3]"
            linears = (self.w1, self.w2, self.w3)
        logger.info(
            "Initializing Disco FeedForward (layer=%s) with std=%.6f for weights %s",
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


class TransformerBlock(MuPTransformerBlock):
    """Transformer block that swaps in Disco attention/FFN/norms."""

    def __init__(self, layer_id: int, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(layer_id, model_args)
        self.attention = Attention(model_args)
        self.attention.mup_layer_id = layer_id
        self.feed_forward = FeedForward(model_args)
        self.feed_forward.mup_layer_id = layer_id
        self._refresh_norms(model_args)

    def _refresh_norms(self, model_args: TransformerModelArgsMuP) -> None:
        elementwise_affine = model_args.torch_layernorm_elementwise_affine
        bias = model_args.torch_layernorm_bias if model_args.use_torch_layernorm else False
        build_norm = lambda: _build_norm_module(  # noqa: E731
            model_args.dim,
            eps=model_args.norm_eps,
            model_args=model_args,
            prefer_torch=model_args.use_torch_layernorm,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )
        self.attention_norm = build_norm()
        self.ffn_norm = build_norm()
        if self.use_peri_norm:
            self.post_attn_norm = build_norm()
            self.post_ffn_norm = build_norm()
        else:
            self.post_attn_norm = None
            self.post_ffn_norm = None


class Transformer(MuPTransformer):
    """Transformer model that reuses MuP core with Disco options."""

    def __init__(self, model_args: TransformerModelArgsMuP) -> None:
        super().__init__(model_args)
        self.model_args = cast("TransformerModelArgsMuP", model_args)
        self.init_config = model_args.init_config_obj
        self._hidden_init_type = self.init_config.resolved_hidden_init(model_args.use_disco)
        self._embed_init_type = self.init_config.resolved_embed_init(model_args.use_disco)
        self._output_init_type = self.init_config.resolved_output_init(model_args.use_disco)
        self._trunc_normal_cutoff = self.init_config.trunc_normal_cutoff
        self._use_custom_init = model_args.use_disco or any(
            init_type not in {None, "normal"}
            for init_type in (
                self.init_config.hidden_init,
                self.init_config.embed_init,
                self.init_config.output_init,
            )
        )

        emb_elementwise = model_args.torch_layernorm_elementwise_affine
        emb_bias = model_args.torch_layernorm_bias if model_args.use_torch_layernorm else False
        if model_args.use_embedding_norm:
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
        self._warned_missing_unembed_bucket = False

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        super().init_weights(buffer_device)
        if not self._use_custom_init:
            return

        init_std = self.init_config.init_std
        emb_init_std = self.init_config.emb_init_std or init_std
        disco_eps = self.init_config.scion_init_eps
        trunc_cutoff = self._trunc_normal_cutoff

        if self.tok_embeddings is not None:
            initialize_tensor(
                self.tok_embeddings.weight,
                init_type=self._embed_init_type,
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
                init_type=self._output_init_type,
                init_std=emb_init_std,
                scion_eps=disco_eps,
                trunc_normal_cutoff=trunc_cutoff,
            )
        else:
            if self._embed_init_type != self._output_init_type:
                logger.warning(
                    "tie_word_embeddings enabled but embed_init=%s differs from output_init=%s; "
                    "using embedding initialization for both.",
                    self._embed_init_type,
                    self._output_init_type,
                )
            self.output.weight = self.tok_embeddings.weight

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
        def _get_bucket(bucket_name: str) -> list[Parameter]:
            return buckets.get(bucket_name, [])

        group_specs: list[tuple[str, dict[str, Any]]] = [
            (
                "emb",
                {"params": _get_bucket("emb"), "weight_decay": weight_decay, "lr": base_lr},
            ),
        ]
        unembed_params = buckets.get("unembed")
        if unembed_params is None:
            if not self._warned_missing_unembed_bucket:
                reason = (
                    "tied embeddings share the output projection"
                    if self.model_args.tie_word_embeddings
                    else "no parameters were assigned to the unembed bucket"
                )
                logger.info(
                    "MuP optimizer overrides: skipping dedicated 'unembed' param group because %s.",
                    reason,
                )
                self._warned_missing_unembed_bucket = True
        else:
            group_specs.append(
                (
                    "unembed",
                    {"params": unembed_params, "weight_decay": weight_decay, "lr": base_lr},
                )
            )
        group_specs.extend(
            [
                (
                    "hidden_ln",
                    {
                        "params": _get_bucket("hidden_ln"),
                        "weight_decay": 0.0,
                        "lr": base_lr * depth_lr_scaling,
                    },
                ),
                (
                    "decay_lr",
                    {
                        "params": _get_bucket("decay_lr"),
                        "weight_decay": weight_decay / width_lr_scaling,
                        "lr": base_lr * width_lr_scaling * depth_lr_scaling,
                    },
                ),
                (
                    "hidden_bias",
                    {
                        "params": _get_bucket("hidden_bias"),
                        "weight_decay": 0.0,
                        "lr": base_lr * depth_lr_scaling,
                    },
                ),
                (
                    "no_decay",
                    {"params": _get_bucket("no_decay"), "weight_decay": 0.0, "lr": base_lr},
                ),
            ]
        )

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
                    name
                    for name, bucket in self._last_bucket_assignments.items()
                    if bucket == label
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
        for label, group in zip(labels, groups, strict=True):
            norm_name = group.get("norm", "<unset>")
            params = group.get("params", [])
            param_count = sum(param.numel() for param in params) if params else 0
            bucket_param_names = sorted(
                name
                for name, bucket in self._last_bucket_assignments.items()
                if bucket == label
            )[:5]
            logger.info(
                "Disco norm debug: bucket=%s norm=%s tensors=%d params=%d sample_params=%s",
                label,
                norm_name,
                len(params),
                param_count,
                bucket_param_names or "n/a",
            )

    def get_optimizer_param_groups(
        self, optimizer_config: dict[str, Any]
    ) -> tuple[Iterator[Parameter] | list[dict[str, Any]], dict[str, Any]]:
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
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        if self.embedding_norm is not None:
            h = self.embedding_norm(h)
        apply_mup_scaling = self.mup_config.mup_enabled and not self.model_args.use_scion
        if apply_mup_scaling:
            h = h * self.mup_config.mup_input_alpha

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h

        if apply_mup_scaling:
            h = h * (
                self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier
            )

        logits = self.output(h)
        if self.init_config.output_mult is not None:
            logits = logits * self.init_config.output_mult
        return logits
