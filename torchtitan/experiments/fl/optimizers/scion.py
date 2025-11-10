# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Scion optimizers used in FL experiments."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["Scion", "ScionLight", "QHScion", "ScionAggMo", "zeroth_power_via_svd"]


# ---------------------------------------------------------------------------
# Norm backends
# ---------------------------------------------------------------------------


class Norm:
    """Base interface for applying norm-limited updates."""

    def lmo(self, grad: Tensor) -> Tensor:  # pragma: no cover - interface only
        raise NotImplementedError

    def init(self, param: Tensor) -> Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class ColNorm(Norm):
    """Column-wise normalisation."""

    def __init__(self, *, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, grad: Tensor) -> Tensor:
        eps = 1e-8
        working = grad.transpose(0, 1) if self.transpose else grad
        rms = torch.sum(working.square(), dim=0, keepdim=True).sqrt()
        rms.mul_(1.0 / math.sqrt(working.size(0)))
        if self.normalized:
            rms.mul_(working.size(1))
        working = working / (rms + eps)
        return working.transpose(0, 1) if self.transpose else working

    def init(self, param: Tensor) -> Tensor:
        dtype = param.dtype
        working = param.data.transpose(0, 1) if self.transpose else param.data
        torch.nn.init.normal_(working)
        working /= working.norm(dim=0, keepdim=True)
        working *= math.sqrt(working.size(0))
        if self.normalized:
            working /= working.size(1)
        param.data = (working.transpose(0, 1) if self.transpose else working).to(dtype=dtype)
        return param


class RowNorm(Norm):
    """Row-wise normalisation."""

    def __init__(self, *, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, grad: Tensor) -> Tensor:
        eps = 1e-8
        working = grad.transpose(0, 1) if self.transpose else grad
        rms = torch.sum(working.square(), dim=-1, keepdim=True).sqrt()
        if self.normalized:
            rms.mul_(math.sqrt(working.size(-1)))
        working = working / (rms + eps)
        return working.transpose(0, 1) if self.transpose else working

    def init(self, param: Tensor) -> Tensor:
        dtype = param.dtype
        working = param.data.transpose(0, 1) if self.transpose else param.data
        torch.nn.init.normal_(working)
        working /= working.norm(dim=-1, keepdim=True)
        if self.normalized:
            working /= math.sqrt(working.size(-1))
        param.data = (working.transpose(0, 1) if self.transpose else working).to(dtype=dtype)
        return param


class BiasRMS(Norm):
    """Normalisation for bias parameters using root-mean-square scaling."""

    def lmo(self, grad: Tensor) -> Tensor:
        eps = 1e-8
        rms = torch.mean(grad.square(), dim=0, keepdim=True).sqrt()
        return grad / (rms + eps)

    def init(self, param: Tensor) -> Tensor:
        torch.nn.init.zeros_(param)
        return param


class SpectralConv(Norm):
    """Spectral normalisation for convolutional kernels."""

    def __init__(self, *, steps: int = 5) -> None:
        self.steps = steps

    def lmo(self, grad: Tensor) -> Tensor:
        working = zeropower_via_newtonschulz5(grad.reshape(len(grad), -1), steps=self.steps)
        working = working.view_as(grad)
        out_channels, in_channels, kernel_h, kernel_w = working.shape
        scale = math.sqrt(out_channels / in_channels) / (kernel_h * kernel_w)
        working.mul_(scale)
        return working

    def init(self, param: Tensor) -> Tensor:
        working = param.data.double()
        kernel_sz = param.data.size(2)
        for kx in range(kernel_sz):
            for ky in range(kernel_sz):
                torch.nn.init.orthogonal_(working[:, :, kx, ky])
        out_channels, in_channels, _, _ = working.shape
        scale = math.sqrt(out_channels / in_channels) / (kernel_sz * kernel_sz)
        working.mul_(scale)
        param.data = working.to(dtype=param.data.dtype)
        return param


class Spectral(Norm):
    """Spectral normalisation for matrix parameters."""

    def __init__(self, *, max: bool = False, normalized: bool = True, steps: int = 5) -> None:
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, grad: Tensor) -> Tensor:
        working = zeropower_via_newtonschulz5(grad.reshape(len(grad), -1), steps=self.steps)
        working = working.view_as(grad)
        d_out, d_in = working.shape
        scale = math.sqrt(d_out / d_in) if self.normalized else math.sqrt(d_out)
        if self.max:
            scale = max(1.0, scale)
        working.mul_(scale)
        return working

    def init(self, param: Tensor) -> Tensor:
        working = param.data.double()
        torch.nn.init.orthogonal_(working)
        d_out, d_in = working.shape
        scale = math.sqrt(d_out / d_in) if self.normalized else math.sqrt(d_out)
        if self.max:
            scale = max(1.0, scale)
        working.mul_(scale)
        param.data = working.to(dtype=param.data.dtype)
        return param


class Sign(Norm):
    """Sign-based updates with optional normalisation."""

    def __init__(self, *, zero_init: bool = False, normalized: bool = True) -> None:
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, grad: Tensor) -> Tensor:
        working = grad.sign()
        if self.normalized and grad.ndim == 2:
            working.mul_(1.0 / grad.size(1))
        return working

    def init(self, param: Tensor) -> Tensor:
        if self.zero_init:
            torch.nn.init.zeros_(param)
            return param
        working = torch.randint_like(param, low=0, high=2)
        working = working * 2 - 1
        if self.normalized and param.ndim == 2:
            working.mul_(1.0 / param.size(1))
        param.data = working
        return param


class Auto(Norm):
    """Select a norm backend automatically based on parameter dimensionality."""

    def lmo(self, grad: Tensor) -> Tensor:
        if grad.ndim in (3, 4):
            return SpectralConv().lmo(grad)
        if grad.ndim == 2:
            return Spectral().lmo(grad)
        return BiasRMS().lmo(grad)

    def init(self, param: Tensor) -> Tensor:
        if param.ndim in (3, 4):
            return SpectralConv().init(param)
        if param.ndim == 2:
            return Spectral().init(param)
        return BiasRMS().init(param)


norm_dict = {
    "ColNorm": ColNorm,
    "RowNorm": RowNorm,
    "BiasRMS": BiasRMS,
    "SpectralConv": SpectralConv,
    "Spectral": Spectral,
    "Sign": Sign,
    "Auto": Auto,
}


def _resolve_norm(norm_name: str, kwargs: dict | None) -> Norm:
    backend_cls = norm_dict.get(norm_name)
    if backend_cls is None:
        msg = f"Unsupported norm backend: {norm_name}"
        raise ValueError(msg)
    return backend_cls(**(kwargs or {}))


def _ensure_positive(value: float, name: str) -> None:
    if value < 0.0:
        msg = f"Invalid {name}: {value}"
        raise ValueError(msg)


def _ensure_between(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], received {value}"
        raise ValueError(msg)


def _normalize_betas(
    *,
    betas: tuple[float, ...] | None,
    fallback: float | tuple[float, ...] | list[float] | None,
    default: tuple[float, ...],
    label: str,
) -> tuple[float, ...]:
    if betas is not None:
        values = tuple(float(beta) for beta in betas)
    elif fallback is not None:
        if isinstance(fallback, (list, tuple)):
            values = tuple(float(beta) for beta in fallback)
        else:
            values = (float(fallback),)
    else:
        values = default

    if len(values) == 0:
        raise ValueError(f"{label} must contain at least one entry.")

    for idx, beta in enumerate(values):
        _ensure_positive(beta, f"{label}[{idx}]")

    return values


def _ensure_single_beta(group: dict) -> float:
    return _ensure_beta_tuple(group, legacy_key="momentum", default=(1.0,))[0]


def _ensure_beta_tuple(
    group: dict,
    *,
    legacy_key: str | None,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    betas = group.get("betas")
    if betas is None:
        fallback = group.pop(legacy_key, None) if legacy_key else None
        betas = _normalize_betas(
            betas=None,
            fallback=fallback,
            default=default,
            label="betas",
        )
        group["betas"] = betas
    return tuple(float(beta) for beta in betas)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class _ScionBase(Optimizer):
    """Shared base-class for Scion variants."""

    def __init__(self, params: Iterable[Tensor], defaults: dict) -> None:
        super().__init__(params, defaults)

    def _init_parameters(self, group: dict) -> Norm:
        norm_backend = _resolve_norm(group["norm"], group.get("norm_kwargs"))
        scale = group["scale"]
        for param in group["params"]:
            norm_backend.init(param)
            param.data.mul_(scale)
        return norm_backend


class Scion(_ScionBase):
    """Core Scion optimiser."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        betas: tuple[float, ...] | None = None,
        momentum: float | None = None,
        norm: str = "Auto",
        norm_kwargs: dict | None = None,
        scale: float = 1.0,
        unconstrained: bool = False,
    ) -> None:
        _ensure_positive(lr, "learning rate")
        betas_tuple = _normalize_betas(
            betas=betas,
            fallback=momentum,
            default=(1.0,),
            label="betas",
        )
        defaults = {
            "lr": lr,
            "betas": betas_tuple,
            "scale": scale,
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = _ensure_single_beta(group)
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = _resolve_norm(group["norm"], group.get("norm_kwargs"))

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if beta1 != 1.0:
                    buf = state.setdefault("exp_avg", torch.zeros_like(grad))
                    buf.mul_(1.0 - beta1).add_(grad, alpha=beta1)
                    grad_to_use = buf
                else:
                    grad_to_use = grad

                update = scale * norm_backend.lmo(grad_to_use)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)


class ScionLight(_ScionBase):
    """Memory-efficient Scion variant using in-place gradient buffers."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        betas: tuple[float, ...] | None = None,
        momentum: float | None = None,
        norm: str = "Auto",
        norm_kwargs: dict | None = None,
        scale: float = 1.0,
        unconstrained: bool = False,
    ) -> None:
        _ensure_positive(lr, "learning rate")
        betas_tuple = _normalize_betas(
            betas=betas,
            fallback=momentum,
            default=(1.0,),
            label="betas",
        )
        defaults = {
            "lr": lr,
            "betas": betas_tuple,
            "scale": scale,
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
        }
        super().__init__(params, defaults)
        self._store_grads_in_state()
        self.register_state_dict_pre_hook(type(self)._store_grads_in_state)
        self.register_load_state_dict_post_hook(type(self)._load_grads_from_state)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = _ensure_single_beta(group)
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = _resolve_norm(group["norm"], group.get("norm_kwargs"))

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                update = scale * norm_backend.lmo(grad)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

                if beta1 != 1.0:
                    grad.mul_(1.0 - beta1)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)

    def __getstate__(self):  # pragma: no cover - save hook
        self._store_grads_in_state()
        return super().__getstate__()

    def __setstate__(self, state):  # pragma: no cover - load hook
        super().__setstate__(state)
        self._load_grads_from_state()

    def _store_grads_in_state(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    self.state.setdefault(param, {})["grad_state"] = param.grad

    def _load_grads_from_state(self):
        for param, state in self.state.items():
            if isinstance(param, torch.Tensor):
                param.grad = state.get("grad_state")


class QHScion(_ScionBase):
    """Quasi-hyperbolic Scion variant."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        betas: tuple[float, ...] | None = None,
        momentum: float | None = None,
        v: float = 1.0,
        norm: str = "Auto",
        norm_kwargs: dict | None = None,
        scale: float = 1.0,
        unconstrained: bool = False,
    ) -> None:
        _ensure_positive(lr, "learning rate")
        _ensure_between(v, "v")
        betas_tuple = _normalize_betas(
            betas=betas,
            fallback=momentum,
            default=(1.0,),
            label="betas",
        )
        defaults = {
            "lr": lr,
            "betas": betas_tuple,
            "v": v,
            "scale": scale,
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = _ensure_single_beta(group)
            v = group["v"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = _resolve_norm(group["norm"], group.get("norm_kwargs"))

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if beta1 != 1.0:
                    buf = state.setdefault("exp_avg", torch.zeros_like(grad))
                    buf.mul_(1.0 - beta1).add_(grad, alpha=beta1)
                    blended = v * grad + (1.0 - v) * buf
                else:
                    blended = grad

                update = scale * norm_backend.lmo(blended)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)


def _build_moment_specs(betas: tuple[float, ...]) -> list[tuple[float, str]]:
    return [(beta, f"exp_avg_{idx}") for idx, beta in enumerate(betas)]


def _normalise_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Sum of scion betas weights must be positive.")
    return tuple(weight / total for weight in weights)


class ScionAggMo(_ScionBase):
    """Scion variant that aggregates multiple first-moment buffers."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        betas: tuple[float, ...] | None = None,
        momentums: tuple[float, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        norm: str = "Auto",
        norm_kwargs: dict | None = None,
        scale: float = 1.0,
        unconstrained: bool = False,
    ) -> None:
        betas_tuple = _normalize_betas(
            betas=betas,
            fallback=momentums,
            default=(1.0,),
            label="betas",
        )
        weights = weights or tuple(1.0 for _ in betas_tuple)
        if len(weights) != len(betas_tuple):
            raise ValueError("scion_weights must match scion betas length.")
        normalised_weights = _normalise_weights(weights)
        defaults = {
            "lr": lr,
            "betas": betas_tuple,
            "weights": normalised_weights,
            "scale": scale,
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            betas_tuple = _ensure_beta_tuple(group, legacy_key="momentums", default=(1.0,))
            weights: tuple[float, ...] = group["weights"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            norm_backend = _resolve_norm(group["norm"], group.get("norm_kwargs"))

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                for legacy_key in list(state.keys()):
                    if legacy_key == "exp_avgs":
                        del state[legacy_key]

                blended = torch.zeros_like(grad)
                for (beta1, name), weight in zip(_build_moment_specs(betas_tuple), weights):
                    buffer = state.setdefault(
                        name,
                        torch.zeros_like(grad),
                    )
                    if beta1 != 1.0:
                        buffer.mul_(1.0 - beta1).add_(grad, alpha=beta1)
                        blended.add_(buffer, alpha=weight)
                    else:
                        blended.add_(grad, alpha=weight)

                update = scale * norm_backend.lmo(blended)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)


# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------


def _zeropower_impl(grad: Tensor, *, steps: int = 5) -> Tensor:
    if grad.ndim != 2:
        msg = f"zeropower expects a 2-D tensor, received shape {grad.shape}"
        raise ValueError(msg)
    a, b, c = (3.4445, -4.7750, 2.0315)
    working = grad.to(dtype=torch.bfloat16)
    if grad.size(0) > grad.size(1):
        working = working.T

    working /= working.norm() + 1e-7
    for _ in range(steps):
        mat = working @ working.T
        correction = b * mat + c * mat @ mat
        working = a * working + correction @ working

    if grad.size(0) > grad.size(1):
        working = working.T
    return working.to(dtype=grad.dtype)


if hasattr(torch, "compile"):

    @torch.compile
    def zeropower_via_newtonschulz5(grad: Tensor, *, steps: int = 5) -> Tensor:
        return _zeropower_impl(grad, steps=steps)

else:  # pragma: no cover - fallback for older torch versions

    def zeropower_via_newtonschulz5(grad: Tensor, *, steps: int = 5) -> Tensor:
        return _zeropower_impl(grad, steps=steps)


def zeroth_power_via_svd(grad: Tensor) -> Tensor:
    u, _, v = torch.linalg.svd(grad)
    return u @ v.T
