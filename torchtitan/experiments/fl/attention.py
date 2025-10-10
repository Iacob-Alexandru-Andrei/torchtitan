# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexAttention utilities tailored for the FL experiments."""

from __future__ import annotations

from typing import Callable, ClassVar

import torch
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)

from torchtitan.models.attention import ScaledDotProductAttention


class FlexAttention(torch.nn.Module):
    """FlexAttention wrapper with sequence id aware masking support.

    This wrapper mirrors the behaviour of :class:`torch.nn.attention.flex_attention`
    while extending it with a ``sequence_id_causal`` mask type. This mask ensures
    that attention is causal inside each contiguous region that shares the same
    ``sequence_id``.
    """

    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)

    def __init__(
        self, attn_mask_type: str, fixed_block_size: int | None = None
    ) -> None:
        super().__init__()
        if attn_mask_type not in {"causal", "sequence_id_causal"}:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        if (
            attn_mask_type != "causal"
            and fixed_block_size is not None
            and fixed_block_size <= 0
        ):
            raise ValueError("fixed_block_size must be positive when provided.")
        self.attn_mask_type = attn_mask_type
        self.fixed_block_size = fixed_block_size
        self.requires_sequence_id = attn_mask_type == "sequence_id_causal"

        self._cached_block_mask: BlockMask | None = None
        self._cached_sequence_id: torch.Tensor | None = None

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return q_idx >= kv_idx

        causal_mask.__name__ = "causal_mask_mod"
        return causal_mask

    @staticmethod
    def _get_sequence_id_mask_mod(
        sequence_id: torch.Tensor,
    ) -> _mask_mod_signature:
        if sequence_id.ndim != 2:
            raise ValueError(
                "sequence_id must be a 2D tensor shaped [batch, sequence_length]."
            )

        seq_idx = sequence_id.to(dtype=torch.int32)

        def sequence_id_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        sequence_id_causal_mask.__name__ = "sequence_id_causal_mask_mod"
        return sequence_id_causal_mask

    @staticmethod
    def _fixed_block_mask_mod(
        mask_mod: _mask_mod_signature, fixed_block_size: int
    ) -> _mask_mod_signature:
        def blocked_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ) -> torch.Tensor:
            q_block = q_idx // fixed_block_size
            kv_block = kv_idx // fixed_block_size
            same_block = q_block == kv_block
            inner_mask = mask_mod(
                b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size
            )
            return same_block & inner_mask

        blocked_mask_mod.__name__ = (
            f"blocked_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
        )
        return blocked_mask_mod

    def _compute_block_mask(
        self, q: torch.Tensor, sequence_id: torch.Tensor | None
    ) -> BlockMask:
        seqlen = q.shape[-2]
        match self.attn_mask_type:
            case "causal":
                mask_mod = self._get_causal_mask_mod()
                batch_dimension = 1
            case "sequence_id_causal":
                if sequence_id is None:
                    raise ValueError(
                        "sequence_id must be provided for sequence_id_causal mask."
                    )
                mask_mod = self._get_sequence_id_mask_mod(sequence_id)
                batch_dimension = sequence_id.shape[0]
            case _:
                raise RuntimeError(f"Unsupported attn_mask_type {self.attn_mask_type}.")

        if self.fixed_block_size is not None and self.fixed_block_size > 0:
            mask_mod = self._fixed_block_mask_mod(mask_mod, self.fixed_block_size)

        return FlexAttention.compiled_create_block_mask(
            mask_mod, batch_dimension, None, seqlen, seqlen
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sequence_id: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        cache_key = None
        if sequence_id is not None:
            cache_key = sequence_id.detach().contiguous()
        if cache_key is not None and self._cached_sequence_id is not None:
            if cache_key.shape == self._cached_sequence_id.shape and torch.equal(
                cache_key, self._cached_sequence_id
            ):
                block_mask = self._cached_block_mask
            else:
                block_mask = None
        else:
            block_mask = self._cached_block_mask if cache_key is None else None

        if block_mask is None:
            block_mask = self._compute_block_mask(q, sequence_id)
            if cache_key is not None:
                self._cached_sequence_id = cache_key.clone()
            else:
                self._cached_sequence_id = None
            self._cached_block_mask = block_mask

        return FlexAttention.flex_attn(
            q, k, v, block_mask=block_mask, scale=scale
        )


def build_attention(
    use_flex_attn: bool,
    attn_mask_type: str,
    fixed_block_size: int | None = None,
) -> torch.nn.Module:
    if use_flex_attn:
        return FlexAttention(attn_mask_type, fixed_block_size)
    if fixed_block_size is not None:
        raise ValueError(
            "TorchTitan with SDPA currently does not support fixed_block_size."
        )
    return ScaledDotProductAttention(attn_mask_type)


__all__ = ["FlexAttention", "build_attention"]
