# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter
from .mup_args import TransformerModelArgs


class Llama3MuPStateDictAdapter(Llama3StateDictAdapter):
    """State dict adapter for Llama3 MuP model.

    Inherits from the standard Llama3StateDictAdapter and extends the mapping
    to handle MuP-specific features:

    1. embedding_norm: Optional RMSNorm layer applied to embeddings
    2. post_attn_norm/post_ffn_norm: Peri-normalization layers (standard in μP)
    3. output.weight: Optional - not present when tie_word_embeddings=True

    Weight Tying Compatibility:
    ---------------------------
    When tie_word_embeddings=True:
        - TorchTitan state_dict contains: tok_embeddings.weight (no output.weight)
        - HuggingFace format uses: model.embed_tokens.weight (no lm_head.weight)
        - The base adapter already handles this correctly via from_hf_map

    When tie_word_embeddings=False:
        - TorchTitan state_dict contains: tok_embeddings.weight AND output.weight
        - HuggingFace format has: model.embed_tokens.weight AND lm_head.weight
        - Both weights are converted using the standard mapping

    Why This Works:
    ---------------
    The base Llama3StateDictAdapter iterates through keys in the state_dict
    and converts them. If output.weight doesn't exist (weight tying case),
    it simply won't be in the state_dict and won't be converted. This is
    correct behavior - no special handling needed.

    MuP Extensions:
    ---------------
    We extend from_hf_map to add mappings for:
    - embedding_norm.weight (applied to embeddings after lookup)
    - post_attn_norm.weight (peri-normalization after attention)
    - post_ffn_norm.weight (peri-normalization after FFN)

    These are ignored during HF->TorchTitan conversion (mapped to None)
    since they don't exist in standard HuggingFace Llama3 checkpoints.
    """

    def __init__(
        self,
        model_args: TransformerModelArgs,
        hf_assets_path: str | None,
    ) -> None:
        # Initialize base adapter with standard Llama3 mappings
        super().__init__(model_args, hf_assets_path)

        # Extend the mapping with MuP-specific layers
        # These are TorchTitan-only features, so we map them to None for HF conversion
        self.from_hf_map.update(
            {
                # MuP extension: optional embedding normalization
                "embedding_norm.weight": None,  # Not in HF format
                # MuP extension: peri-normalization (per-layer)
                "model.layers.{}.post_attn_norm.weight": None,  # Not in HF format
                "model.layers.{}.post_ffn_norm.weight": None,  # Not in HF format
            }
        )
