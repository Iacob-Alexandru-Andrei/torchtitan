# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field
from typing import Any

from torchtitan.models.llama3.model.args import (
    TransformerModelArgs as BaseTransformerModelArgs,
)


@dataclass
class MuPConfig:
    mup_enabled: bool = False
    mup_disable_attention_scaling: bool = True
    mup_disable_hidden_lr_scaling: bool = False
    mup_width_multiplier: float = 1.0
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    completep_depth_alpha_enabled: bool = False
    completep_depth_multiplier: float = 1.0
    completep_depth_alpha_exp: float = 1.0
    completep_eps_scaling_enabled: bool = True


@dataclass
class ModelInitConfig:
    init_std: float = 0.02
    emb_init_std: float | None = None
    output_mult: float | None = None


@dataclass
class TransformerModelArgs(BaseTransformerModelArgs):
    # muP / CompleteP
    use_embedding_norm: bool = False
    use_peri_norm: bool = False
    tie_word_embeddings: bool = False
    mup_config: dict[str, Any] = field(default_factory=dict)
    init_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.mup_config_obj = MuPConfig(**self.mup_config)
        self.init_config_obj = ModelInitConfig(**self.init_config)
