# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Shared RMSNorm utilities for MuP variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn


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


def build_norm_module(
    normalized_shape: int | Sequence[int],
    *,
    eps: float,
    torch_norm_cls: Type[nn.Module],
    prefer_torch: bool,
    elementwise_affine: bool = True,
    bias: bool = False,
    add_unit_offset: bool = True,
    force_bf16: bool = False,
) -> nn.Module:
    """Create a norm module using either a torch LayerNorm or functional RMSNorm."""
    if prefer_torch:
        return torch_norm_cls(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )
    return TitanRMSNorm(
        normalized_shape,
        eps=eps,
        elementwise_affine=elementwise_affine,
        add_unit_offset=add_unit_offset,
        force_bf16=force_bf16,
    )
