# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration helpers for the MuP-enabled MPT model."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

from llmfoundry.models.mpt.configuration_mpt import MPTConfig

from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torchtitan.config import JobConfig


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
    mup_scale_expert_sel: bool = False  # retained for compatibility with SigmaMoE


@dataclass
class ModelInitConfig:
    """Initialization overrides for MuP-tuned models."""

    init_std: float | None = 0.02
    emb_init_std: float | None = None
    output_mult: float | None = None


class MPTMuPConfig(MPTConfig):
    """Extends :class:`MPTConfig` with MuP-specific settings."""

    def __init__(
        self,
        *args: Any,
        n_non_expert_layers: int = 0,
        mup_config: Mapping[str, Any] | None = None,
        use_peri_norm: bool = True,
        use_embedding_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        obj_mup_config = MuPConfig(**(mup_config or {}))

        self.mup_config = asdict(obj_mup_config)
        self.n_non_expert_layers = n_non_expert_layers
        self.use_peri_norm = use_peri_norm
        self.use_embedding_norm = use_embedding_norm

        super().__init__(*args, **kwargs)

        if self.n_non_expert_layers < 0:
            msg = f"n_non_expert_layers ({self.n_non_expert_layers}) must be >= 0."
            raise ValueError(msg)
        if self.n_non_expert_layers > self.n_layers:
            msg = (
                f"n_non_expert_layers ({self.n_non_expert_layers}) must be <= n_layers "
                f"({self.n_layers})."
            )
            raise ValueError(msg)
        if self.n_non_expert_layers > 0:
            logger.warning(
                "n_non_expert_layers (%s) is > 0, dense blocks will be used for early layers.",
                self.n_non_expert_layers,
            )

        if (
            self.mup_config["mup_enabled"]
            and not self.mup_config["mup_disable_attention_scaling"]
        ):
            head_dim = self.d_model // self.n_heads
            attn_config = dict(self.attn_config)
            if attn_config.get("softmax_scale") is None:
                attn_config["softmax_scale"] = 1.0 / float(head_dim)
            self.attn_config = attn_config

        if self.logit_scale is not None and obj_mup_config.mup_enabled:
            msg = (
                "logit_scale is not supported with muP enabled. Set logit_scale to None "
                "and adjust mup_output_alpha instead."
            )
            raise ValueError(msg)


@dataclass
class TransformerModelArgs(BaseModelArgs):
    """User-facing configuration for the MuP-enabled MPT model."""

    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    head_dim: int | None = None
    expansion_ratio: float = 4.0
    vocab_size: int = 50368
    max_seq_len: int = 2048
    resid_pdrop: float = 0.0
    emb_pdrop: float = 0.0
    learned_pos_emb: bool = False
    attn_config: Mapping[str, Any] | None = None
    ffn_config: Mapping[str, Any] | None = None
    init_device: str = "meta"
    logit_scale: float | None = None
    no_bias: bool = False
    attention_bias: bool | None = None
    embedding_fraction: float = 1.0
    norm_type: str = "low_precision_layernorm"
    norm_eps: float = 1e-5
    use_cache: bool = False
    init_config: Mapping[str, Any] = field(
        default_factory=lambda: {"name": "kaiming_normal_", "init_std": 0.02}
    )
    fc_type: Mapping[str, Any] | None = None
    tie_word_embeddings: bool = True
    use_pad_tok_in_ffn: bool = True
    block_overrides: Mapping[str, Any] | None = None
    final_logit_softcapping: float | None = None

    # MuP extensions
    use_peri_norm: bool = True
    use_embedding_norm: bool = True
    n_non_expert_layers: int = 0
    mup_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mup_config_obj = MuPConfig(**self.mup_config)
        self.init_config_obj = ModelInitConfig(
            **{
                k: self.init_config.get(k)
                for k in ("init_std", "emb_init_std", "output_mult")
            }
        )

    @property
    def dim(self) -> int:
        return self.d_model

    @property
    def enable_weight_tying(self) -> bool:
        return self.tie_word_embeddings

    def update_from_config(self, job_config: JobConfig, **kwargs: Any) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                "Sequence length %s exceeds configured max_seq_len %s; overriding.",
                seq_len,
                self.max_seq_len,
            )
        self.max_seq_len = seq_len

    def get_nparams_and_flops(self, model, seq_len: int) -> tuple[int, float]:
        return get_dense_model_nparams_and_flops(self, model, seq_len)

    def to_config(self) -> MPTMuPConfig:
        attn_config = (
            copy.deepcopy(self.attn_config) if self.attn_config is not None else None
        )
        ffn_config = (
            copy.deepcopy(self.ffn_config) if self.ffn_config is not None else None
        )
        fc_type = copy.deepcopy(self.fc_type) if self.fc_type is not None else None
        block_overrides = (
            copy.deepcopy(self.block_overrides)
            if self.block_overrides is not None
            else None
        )
        init_config = dict(self.init_config)
        init_config.setdefault("name", "kaiming_normal_")

        return MPTMuPConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            head_dim=self.head_dim,
            expansion_ratio=self.expansion_ratio,
            max_seq_len=self.max_seq_len,
            vocab_size=self.vocab_size,
            resid_pdrop=self.resid_pdrop,
            emb_pdrop=self.emb_pdrop,
            learned_pos_emb=self.learned_pos_emb,
            attn_config=attn_config,
            ffn_config=ffn_config,
            init_device=self.init_device,
            logit_scale=self.logit_scale,
            no_bias=self.no_bias,
            attention_bias=self.attention_bias,
            embedding_fraction=self.embedding_fraction,
            norm_type=self.norm_type,
            norm_eps=self.norm_eps,
            use_cache=self.use_cache,
            init_config=init_config,
            fc_type=fc_type,
            tie_word_embeddings=self.tie_word_embeddings,
            use_pad_tok_in_ffn=self.use_pad_tok_in_ffn,
            block_overrides=block_overrides,
            final_logit_softcapping=self.final_logit_softcapping,
            n_non_expert_layers=self.n_non_expert_layers,
            mup_config=self.mup_config,
            use_peri_norm=self.use_peri_norm,
            use_embedding_norm=self.use_embedding_norm,
        )


__all__ = [
    "MPTMuPConfig",
    "ModelInitConfig",
    "MuPConfig",
    "TransformerModelArgs",
]
