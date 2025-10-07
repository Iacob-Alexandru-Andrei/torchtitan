# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.config import JobConfig
from .mup_args import TransformerModelArgs
from torchtitan.protocols import StateDictAdapter

# to be used with the checkpoint converter
pattern_mapping = {
    r"tok_embeddings.weight": "tok_embeddings.weight",
    r"tok_embeddings.norm.weight": "tok_embeddings.norm.weight",
    r"layers\.(\d+)\.attention\.wq\.weight": "layers.{}.attention.wq.weight",
    r"layers\.(\d+)\.attention\.wk\.weight": "layers.{}.attention.wk.weight",
    r"layers\.(\d+)\.attention\.wv\.weight": "layers.{}.attention.wv.weight",
    r"layers\.(\d+)\.attention\.wo\.weight": "layers.{}.attention.wo.weight",
    r"layers\.(\d+)\.feed_forward\.w1\.weight": "layers.{}.feed_forward.w1.weight",
    r"layers\.(\d+)\.feed_forward\.w2\.weight": "layers.{}.feed_forward.w2.weight",
    r"layers\.(\d+)\.feed_forward\.w3\.weight": "layers.{}.feed_forward.w3.weight",
    r"layers\.(\d+)\.attention_norm\.weight": "layers.{}.attention_norm.weight",
    r"layers\.(\d+)\.ffn_norm\.weight": "layers.{}.ffn_norm.weight",
    r"layers\.(\d+)\.post_attn_norm\.weight": "layers.{}.post_attn_norm.weight",
    r"layers\.(\d+)\.post_ffn_norm\.weight": "layers.{}.post_ffn_norm.weight",
    r"norm\.weight": "norm.weight",
    r"output\.weight": "output.weight",
}


class Llama3MuPStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_args: TransformerModelArgs,
        job_config: JobConfig,
    ):
        super().__init__(model_args, job_config)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {}
        for titan_fqn, tensor in state_dict.items():
            for hf_pattern, hf_fqn_template in pattern_mapping.items():
                m = re.fullmatch(hf_pattern, titan_fqn)
                if m:
                    if m.re.groups > 0:
                        hf_state_dict[hf_fqn_template.format(m.group(1))] = tensor
                    else:
                        hf_state_dict[hf_fqn_template] = tensor
                    break
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        titan_state_dict = {}
        for hf_fqn, tensor in hf_state_dict.items():
            for titan_pattern, titan_fqn_template in pattern_mapping.items():
                m = re.fullmatch(titan_pattern, hf_fqn)
                if m:
                    if m.re.groups > 0:
                        titan_state_dict[titan_fqn_template.format(m.group(1))] = tensor
                    else:
                        titan_state_dict[titan_fqn_template] = tensor
                    break
        return titan_state_dict