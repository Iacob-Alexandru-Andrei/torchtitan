# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Dataclasses defining MuP-specific configuration for LLaMA models."""

from dataclasses import dataclass, field
from typing import Any, Literal

from torchtitan.models.llama3.model.args import (
    TransformerModelArgs as BaseTransformerModelArgs,
)

from .disco_init import ALLOWED_INIT_TYPES, canonicalize_init_type


@dataclass
class MuPConfig:
    """Options controlling MuP/CompleteP behaviour."""

    mup_enabled: bool = True
    mup_disable_attention_scaling: bool = True
    mup_disable_hidden_lr_scaling: bool = False
    mup_width_multiplier: float = 1.0
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    completep_depth_alpha_enabled: bool = True
    completep_depth_multiplier: float = 1.0
    completep_depth_alpha_exp: float = 1.0
    completep_eps_scaling_enabled: bool = True


@dataclass
class ModelInitConfig:
    """Initialization overrides for MuP-tuned models."""

    init_std: float = 0.02
    emb_init_std: float | None = 0.02
    output_mult: float | None = None
    scion_init_eps: float = 1e-12
    hidden_init: str | None = None
    embed_init: str | None = None
    output_init: str | None = None
    trunc_normal_cutoff: float = 3.0

    def __post_init__(self) -> None:
        """Normalize initializer strings and validate numeric inputs."""
        for attr in ("hidden_init", "embed_init", "output_init"):
            value = getattr(self, attr)
            if value is None:
                continue
            lowered = value.lower()
            canonical = canonicalize_init_type(lowered)
            setattr(self, attr, canonical)
        if self.trunc_normal_cutoff <= 0:
            msg = "trunc_normal_cutoff must be positive."
            raise ValueError(msg)

    def resolved_hidden_init(self, use_disco: bool) -> str:
        if self.hidden_init is not None:
            return self.hidden_init
        return "disco_normal" if use_disco else "normal"

    def resolved_embed_init(self, use_disco: bool) -> str:
        if self.embed_init is not None:
            return self.embed_init
        if use_disco:
            return "disco_normal_input"
        return self.resolved_hidden_init(use_disco)

    def resolved_output_init(self, use_disco: bool) -> str:
        if self.output_init is not None:
            return self.output_init
        if use_disco:
            return "disco_normal_output"
        return self.resolved_hidden_init(use_disco)


@dataclass
class TransformerModelArgs(BaseTransformerModelArgs):
    """Extended transformer arguments adding MuP-specific sections."""

    # muP / CompleteP
    use_embedding_norm: bool = True
    use_peri_norm: bool = True
    tie_word_embeddings: bool = True
    use_torch_layernorm: bool = True
    layernorm_impl: Literal["torch", "rms"] | None = None
    force_rmsnorm_bf16: bool = False
    use_simple_silu_ffn: bool = False
    head_dim: int | None = None
    qk_norm: bool = True
    use_torch_qk_layernorm: bool | None = None
    qk_layernorm_impl: Literal["torch", "rms"] | None = None
    use_attention_value_norm: bool = False
    use_attention_output_norm: bool = False
    use_mlp_mid_norm: bool = False
    torch_layernorm_elementwise_affine: bool = True
    qk_norm_elementwise_affine: bool = True
    torch_layernorm_bias: bool = False
    qk_norm_bias: bool = False
    use_scion: bool = False
    use_disco: bool = False
    scion_hidden_scale: float = 50.0
    scion_output_scale: float = 3000.0
    mup_config: dict[str, Any] = field(default_factory=dict)
    init_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Instantiate strongly typed helpers from the raw configuration maps."""
        if self.use_disco and not self.use_scion:
            self.use_scion = True
        self.mup_config_obj = MuPConfig(**self.mup_config)
        self.init_config_obj = ModelInitConfig(**self.init_config)
        if self.scion_hidden_scale <= 0 or self.scion_output_scale <= 0:
            msg = "scion_hidden_scale and scion_output_scale must be positive."
            raise ValueError(msg)
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
        if self.layernorm_impl is not None:
            impl = self.layernorm_impl.lower()
            if impl not in {"torch", "rms"}:
                msg = "layernorm_impl must be 'torch' or 'rms'"
                raise ValueError(msg)
            self.use_torch_layernorm = impl == "torch"

        resolved_qk_impl: bool
        if self.qk_layernorm_impl is not None:
            impl = self.qk_layernorm_impl.lower()
            if impl not in {"torch", "rms"}:
                msg = "qk_layernorm_impl must be 'torch' or 'rms'"
                raise ValueError(msg)
            resolved_qk_impl = impl == "torch"
        elif self.use_torch_qk_layernorm is not None:
            resolved_qk_impl = bool(self.use_torch_qk_layernorm)
        else:
            resolved_qk_impl = self.use_torch_layernorm
        self.use_torch_qk_layernorm = resolved_qk_impl
