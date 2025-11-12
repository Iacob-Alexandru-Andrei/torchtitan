# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Initialization helpers tailored for Disco-enabled models."""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor, nn

_BASE_INIT_TYPES: Final[set[str]] = {
    "normal",
    "trunc_normal",
    "orthogonal",
    "scaled_orthogonal",
}
_DISCO_INIT_TYPES: Final[set[str]] = {
    "disco_normal",
    "disco_normal_input",
    "disco_normal_output",
}
_SCION_TO_DISCO_ALIASES: Final[dict[str, str]] = {
    "scion_normal": "disco_normal",
    "scion_normal_input": "disco_normal_input",
    "scion_normal_output": "disco_normal_output",
}
_CANONICAL_INIT_TYPES: Final[set[str]] = _BASE_INIT_TYPES | _DISCO_INIT_TYPES
ALLOWED_INIT_TYPES: Final[set[str]] = _CANONICAL_INIT_TYPES | set(_SCION_TO_DISCO_ALIASES.keys())
_DEFAULT_TRUNC_CUTOFF: Final[float] = 3.0


def _canonicalize_init_type(init_type: str) -> str:
    lowered = init_type.lower()
    canonical = _SCION_TO_DISCO_ALIASES.get(lowered, lowered)
    if canonical not in _CANONICAL_INIT_TYPES:
        msg = f"Unsupported initialization type {init_type!r}. Expected one of {sorted(ALLOWED_INIT_TYPES)}."
        raise ValueError(msg)
    return canonical


def disco_normal_(
    tensor: Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    norm_axis: int = 1,
    eps: float = 1e-12,
    scale_type: str | None = None,
) -> Tensor:
    """Initialize ``tensor`` with Disco-normalized rows."""
    if tensor.ndim != 2:
        msg = f"Disco initialization expects a 2-D tensor, received shape {tuple(tensor.shape)}"
        raise ValueError(msg)

    nn.init.normal_(tensor, mean=mean, std=std)

    if scale_type is None:
        target_scale = 1.0
    elif scale_type == "input":
        target_scale = math.sqrt(float(tensor.shape[norm_axis]))
    elif scale_type == "output":
        target_scale = 1.0 / math.sqrt(float(tensor.shape[norm_axis]))
    else:  # pragma: no cover - guarded by config
        msg = f"Unknown Disco scale_type {scale_type!r}"
        raise ValueError(msg)

    norms = torch.linalg.vector_norm(tensor, dim=norm_axis, keepdim=True)
    tensor.mul_(target_scale / (norms + eps))
    return tensor


def _init_orthogonal(tensor: Tensor, *, gain: float) -> None:
    if tensor.ndim != 2:
        msg = f"Orthogonal initialization expects a 2-D tensor, received shape {tuple(tensor.shape)}"
        raise ValueError(msg)
    nn.init.orthogonal_(tensor, gain=gain)


def _init_scaled_orthogonal(tensor: Tensor, *, gain: float) -> None:
    if tensor.ndim != 2:
        msg = f"Scaled orthogonal initialization expects a 2-D tensor, received shape {tuple(tensor.shape)}"
        raise ValueError(msg)
    fan_out, fan_in = tensor.shape
    if fan_in == 0:
        raise ValueError("Scaled orthogonal initialization requires fan_in > 0.")
    scaled_gain = gain * math.sqrt(float(fan_out) / float(fan_in))
    nn.init.orthogonal_(tensor, gain=scaled_gain)


def initialize_tensor(
    tensor: Tensor,
    *,
    init_type: str,
    init_std: float,
    scion_eps: float,
    trunc_normal_cutoff: float = _DEFAULT_TRUNC_CUTOFF,
    mean: float = 0.0,
) -> None:
    """Apply the requested initialization in-place."""
    init_key = _canonicalize_init_type(init_type)

    if init_key == "normal":
        nn.init.normal_(tensor, mean=mean, std=init_std)
        return
    if init_key == "trunc_normal":
        cutoff = trunc_normal_cutoff if trunc_normal_cutoff > 0 else _DEFAULT_TRUNC_CUTOFF
        a = mean - cutoff * init_std
        b = mean + cutoff * init_std
        nn.init.trunc_normal_(tensor, mean=mean, std=init_std, a=a, b=b)
        return
    if init_key == "orthogonal":
        _init_orthogonal(tensor, gain=init_std)
        return
    if init_key == "scaled_orthogonal":
        _init_scaled_orthogonal(tensor, gain=init_std)
        return
    if init_key == "disco_normal":
        disco_normal_(tensor, mean=mean, std=init_std, eps=scion_eps)
        return
    if init_key == "disco_normal_input":
        disco_normal_(tensor, mean=mean, std=init_std, eps=scion_eps, scale_type="input")
        return
    if init_key == "disco_normal_output":
        disco_normal_(tensor, mean=mean, std=init_std, eps=scion_eps, scale_type="output")
        return


def init_linear_weight(
    linear: nn.Linear,
    *,
    init_std: float,
    init_type: str,
    scion_eps: float,
    trunc_normal_cutoff: float,
) -> None:
    """Initialize a linear layer weight using the provided strategy."""
    initialize_tensor(
        linear.weight,
        init_type=init_type,
        init_std=init_std,
        scion_eps=scion_eps,
        trunc_normal_cutoff=trunc_normal_cutoff,
    )


# Backwards compatibility aliases -------------------------------------------------
scion_normal_ = disco_normal_
canonicalize_init_type = _canonicalize_init_type
