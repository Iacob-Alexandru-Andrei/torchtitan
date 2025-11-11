# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Scion optimizers used in FL experiments."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ._metric_utils import prepare_metrics_for_reduction, reduce_metrics_across_ranks

__all__ = ["Scion", "ScionLight", "QHScion", "ScionAggMo", "zeroth_power_via_svd"]


log = logging.getLogger(__name__)
MUON_ZEROpower_COEFFS = (3.4445, -4.7750, 2.0315)


MetricFn = Callable[[Tensor, dict[str, Any], Tensor], Tensor]


def _zero_metric_like(param: Tensor) -> Tensor:
    dtype = param.dtype if param.is_floating_point() else torch.float32
    return torch.zeros((), device=param.device, dtype=dtype)


def _moment_norm_metric(key: str) -> MetricFn:
    def _metric(param: Tensor, state: dict[str, Any], _step_tensor: Tensor) -> Tensor:
        buf = state.get(key)
        if buf is None:
            return _zero_metric_like(param)
        return torch.linalg.vector_norm(buf)

    return _metric


def _param_norm_metric(param: Tensor, _state: dict[str, Any], _step_tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(param.detach())


def _update_norm_metric(_param: Tensor, _state: dict[str, Any], step_tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(step_tensor)


def _grad_norm_metric(param: Tensor, state: dict[str, Any], _step_tensor: Tensor) -> Tensor:
    grad_state = state.get("grad_state")
    grad = grad_state if isinstance(grad_state, torch.Tensor) else param.grad
    if grad is None:
        return _zero_metric_like(param)
    return torch.linalg.vector_norm(grad)


def _increment_state_step(state: dict[str, Any], param: Tensor) -> torch.Tensor:
    step_tensor = state.get("step")
    if not isinstance(step_tensor, torch.Tensor):
        step_tensor = torch.zeros((), device=param.device, dtype=torch.float32)
    elif step_tensor.device != param.device:
        step_tensor = step_tensor.to(param.device)
    state["step"] = step_tensor
    step_tensor.add_(1.0)
    return step_tensor


def _get_step_tensor_for_metrics(state: dict[str, Any], param: Tensor) -> torch.Tensor:
    step_state = state.get("step")
    if isinstance(step_state, torch.Tensor):
        tensor = step_state.detach().clone()
        if tensor.device != param.device:
            tensor = tensor.to(param.device)
        return tensor
    if step_state is None:
        return torch.tensor(0.0, device=param.device)
    return torch.tensor(float(step_state), device=param.device)


# ---------------------------------------------------------------------------
# Scion-style norm helpers
# ---------------------------------------------------------------------------


def _standardize_norm_name(norm_value: str | None, *, default: str = "spectral") -> str:
    if norm_value is None:
        return default
    key = norm_value.replace("_", "").lower()
    mapping = {
        "auto": "spectral",
        "spectral": "spectral",
        "spectralconv": "conv_spectral",
        "sign": "sign",
        "biasrms": "bias_rms",
        "biasrmsnorm": "bias_rms",
        "embed": "embed_linear",
        "embedlinear": "embed_linear",
        "embedsqrt": "embed_sqrt",
        "unembed": "unembed_linear",
        "unembedlinear": "unembed_linear",
        "unembedsqrt": "unembed_sqrt",
        "scaleonly": "scale_only",
        "scale_only": "scale_only",
    }
    return mapping.get(key, norm_value.lower())


def _ensure_positive(value: float, name: str) -> None:
    if value < 0.0:
        msg = f"Invalid {name}: {value}"
        raise ValueError(msg)


def _ensure_between(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], received {value}"
        raise ValueError(msg)


def _resolve_norm_settings(group: dict) -> tuple[str, str, int, float, dict[str, Any]]:
    has_explicit_norm = ("norm" in group and group.get("norm") is not None) or (
        "norm_factor" in group and group.get("norm_factor") is not None
    )
    default_norm = "scale_only" if (not has_explicit_norm and "scale" in group) else "spectral"
    norm_factor = _standardize_norm_name(group.get("norm_factor") or group.get("norm"), default=default_norm)
    group["norm_factor"] = norm_factor

    norm_kwargs = dict(group.get("norm_kwargs") or {})
    backend = group.get("zeropower_backend") or norm_kwargs.get("backend", "newtonschulz5")
    backend_steps = int(group.get("backend_steps") or norm_kwargs.get("backend_steps", 5))
    eps = float(group.get("eps", norm_kwargs.get("eps", 1e-8)))

    group["zeropower_backend"] = backend
    group["backend_steps"] = backend_steps
    group["eps"] = eps
    group["norm_kwargs"] = norm_kwargs

    return norm_factor, backend, backend_steps, eps, norm_kwargs


def _run_zeropower(
    grad: Tensor,
    backend: str,
    steps: int,
    coefficients: tuple[float, float, float] | None,
) -> Tensor:
    backend_key = backend.lower()
    if backend_key in {"identity", "none"}:
        return grad
    if backend_key == "newtonschulz5":
        if grad.ndim != 2:
            msg = f"zeropower backend expects a 2-D tensor, received {grad.shape}"
            raise ValueError(msg)
        return zeropower_via_newtonschulz5(grad, steps=steps, coeffs=coefficients)
    msg = f"Unsupported zeropower backend: {backend}"
    raise ValueError(msg)


def _normalize_grad(
    tensor: Tensor,
    norm_factor: str,
    eps: float,
    norm_kwargs: dict[str, Any],
) -> Tensor:
    if norm_factor == "spectral":
        if tensor.ndim != 2:
            msg = "spectral norm expects a 2-D tensor"
            raise ValueError(msg)
        d_out, d_in = tensor.shape
        if d_in == 0:
            return tensor
        normalized = bool(norm_kwargs.get("normalized", True))
        if normalized:
            scale = math.sqrt(max(float(d_out), 1.0) / float(d_in))
        else:
            scale = math.sqrt(max(float(d_out), 1.0))
        return tensor * scale
    if norm_factor == "conv_spectral":
        if tensor.ndim not in (4, 5):
            msg = "conv_spectral norm expects a 4-D or 5-D tensor"
            raise ValueError(msg)
        out_ch, in_ch = tensor.shape[0], tensor.shape[1]
        spatial = math.prod(tensor.shape[2:])
        if in_ch == 0 or spatial == 0:
            return tensor
        scale = math.sqrt(max(float(out_ch), 1.0) / float(in_ch)) / float(spatial)
        return tensor * scale
    if norm_factor == "bias_rms":
        rms = tensor.pow(2).mean().sqrt()
        return tensor / (rms + eps)
    if norm_factor == "sign":
        normalized = bool(norm_kwargs.get("normalized", True))
        if normalized:
            dim = float(tensor.size(-1))
            if dim > 0:
                return tensor.sign() * (1.0 / dim)
        return tensor.sign()
    if norm_factor in {"embed_linear", "embed_sqrt", "unembed_linear", "unembed_sqrt"}:
        if tensor.ndim != 2:
            msg = f"{norm_factor} expects a 2-D tensor"
            raise ValueError(msg)
        rms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
        normalized = tensor / (rms + eps)
        dim = float(tensor.size(1))
        if norm_factor == "embed_linear":
            return normalized * dim
        if norm_factor == "embed_sqrt":
            return normalized * math.sqrt(dim)
        if norm_factor == "unembed_linear":
            return normalized / dim
        return normalized / math.sqrt(dim)
    if norm_factor == "none":
        return tensor
    msg = f"Unsupported norm factor: {norm_factor}"
    raise ValueError(msg)


def _apply_lmo(
    grad: Tensor,
    *,
    norm_factor: str,
    backend: str,
    backend_steps: int,
    eps: float,
    coefficients: tuple[float, float, float],
    norm_kwargs: dict[str, Any],
) -> Tensor:
    if norm_factor == "scale_only":
        return grad
    if grad.ndim == 1:
        working = grad
    elif grad.ndim == 2:
        working = _run_zeropower(grad, backend, backend_steps, coefficients)
    elif grad.ndim in (4, 5):
        if norm_factor != "conv_spectral":
            msg = f"Tensor with shape {grad.shape} requires conv_spectral norm factor."
            raise ValueError(msg)
        flat = grad.reshape(grad.size(0), -1)
        flat = _run_zeropower(flat, backend, backend_steps, coefficients)
        working = flat.view_as(grad)
    else:
        msg = f"Unsupported tensor shape for Scion LMO: {grad.shape}"
        raise ValueError(msg)
    return _normalize_grad(working, norm_factor, eps, norm_kwargs)


def _apply_scale_if_configured(update: Tensor, group: dict) -> Tensor:
    scale_value = group.get("scale")
    if scale_value is None:
        return update
    if isinstance(scale_value, torch.Tensor):
        if scale_value.numel() == 1:
            multiplier = float(scale_value.detach().item())
        else:
            multiplier = float(scale_value.reshape(-1)[0].detach().item())
    else:
        multiplier = float(scale_value)
    if multiplier == 1.0:
        return update
    return update * multiplier


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


def _normalize_vs(
    vs: tuple[float, ...] | Sequence[float] | float | None,
    *,
    fallback: float | None,
    label: str = "vs",
) -> tuple[float, ...]:
    if vs is not None:
        if isinstance(vs, (list, tuple)):
            values = tuple(float(v) for v in vs)
        else:
            values = (float(vs),)
    elif fallback is not None:
        values = (float(fallback),)
    else:
        values = (1.0,)

    if len(values) == 0:
        raise ValueError(f"{label} must contain at least one entry.")

    for idx, value in enumerate(values):
        _ensure_between(value, f"{label}[{idx}]")

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


def _resolve_group_v(group: dict) -> float:
    vs_entry = group.get("vs")
    if vs_entry is None:
        legacy_v = group.get("v")
        vs_entry = _normalize_vs(vs=None, fallback=legacy_v)
        group["vs"] = vs_entry

    if isinstance(vs_entry, torch.Tensor):
        v_scalar = float(vs_entry.reshape(-1)[0].detach().item())
    elif isinstance(vs_entry, Sequence) and not isinstance(vs_entry, (str, bytes)):
        v_scalar = float(vs_entry[0])
    else:
        v_scalar = float(vs_entry)

    group["v"] = v_scalar  # legacy alias for checkpoints/metrics expecting 'v'
    return v_scalar


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


class _ScionBase(Optimizer):
    """Shared base-class for Scion variants."""

    def __init__(self, params: Iterable[Tensor], defaults: dict) -> None:
        super().__init__(params, defaults)
        self.metric_functions: dict[str, MetricFn] = self._build_metric_functions()
        self._zeropower_coeffs = tuple(defaults.get("zeropower_coeffs", MUON_ZEROpower_COEFFS))
        self._log_norm_configuration()

    def _init_parameters(self, group: dict) -> None:  # pragma: no cover - retained for API parity
        return None

    def _build_metric_functions(self) -> dict[str, MetricFn]:
        return {
            "l2_norm/moment": _moment_norm_metric("exp_avg"),
            "l2_norm/param": _param_norm_metric,
            "l2_norm/update": _update_norm_metric,
        }

    def _log_norm_configuration(self) -> None:
        if not log.isEnabledFor(logging.INFO):
            return

        optimizer_name = type(self).__name__
        for idx, group in enumerate(self.param_groups):
            group_label = str(group.get("name") or f"group_{idx}")
            norm_factor, backend, backend_steps, eps, norm_kwargs = _resolve_norm_settings(group)
            scale_value = group.get("scale")
            scale_str = "n/a" if scale_value is None else f"{self._to_float(scale_value):.6f}"
            log.info(
                "%s norm configuration | group=%s norm_factor=%s scale=%s backend=%s backend_steps=%s eps=%.2e muon_coeffs=%s norm_kwargs=%s",
                optimizer_name,
                group_label,
                norm_factor,
                scale_str,
                backend,
                backend_steps,
                eps,
                tuple(group.get("zeropower_coeffs", self._zeropower_coeffs)),
                norm_kwargs or {},
            )

    def _find_param_group(self, param: Tensor) -> dict:
        for group in self.param_groups:
            for group_param in group["params"]:
                if group_param is param:
                    return group
        return self.param_groups[0]

    def _compute_direction(
        self, param: Tensor, group: dict, state: dict
    ) -> Tensor | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _build_step_tensor(
        self,
        update: Tensor,
        param: Tensor,
        lr: float | torch.Tensor,
        unconstrained: bool,
    ) -> Tensor:
        lr_value = self._to_float(lr)
        update_tensor = update.detach()
        if unconstrained:
            return update_tensor * lr_value

        denom = 1.0 - lr_value
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        return (update_tensor + param.detach()) * (lr_value / denom)

    def _record_metrics(
        self,
        param: Tensor,
        state: dict[str, Any],
        name: str,
        optimizer_metrics: dict[str, torch.Tensor],
        step_tensor: Tensor,
    ) -> None:
        if "max/optimizer_step" not in optimizer_metrics:
            optimizer_metrics["max/optimizer_step"] = _get_step_tensor_for_metrics(state, param)
        for metric_name, metric_fn in self.metric_functions.items():
            metric_value = metric_fn(param, state, step_tensor)
            if metric_value is not None:
                optimizer_metrics[f"{metric_name}/{name}"] = metric_value

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if param.grad is None:
            return optimizer_metrics

        state = self.state.get(param)
        if state is None:
            return optimizer_metrics

        group = self._find_param_group(param)
        direction = self._compute_direction(param, group, state)
        if direction is None:
            return optimizer_metrics

        norm_factor, backend, backend_steps, eps, norm_kwargs = _resolve_norm_settings(group)
        update = _apply_lmo(
            direction,
            norm_factor=norm_factor,
            backend=backend,
            backend_steps=backend_steps,
            eps=eps,
            coefficients=tuple(group.get("zeropower_coeffs", self._zeropower_coeffs)),
            norm_kwargs=norm_kwargs,
        )
        update = _apply_scale_if_configured(update, group)
        step_tensor = self._build_step_tensor(update, param, group["lr"], group["unconstrained"]).detach()
        self._record_metrics(param, state, name, optimizer_metrics, step_tensor)
        return optimizer_metrics

    @staticmethod
    def dist_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return reduce_metrics_across_ranks(optimizer_metrics)

    @staticmethod
    def pre_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return prepare_metrics_for_reduction(optimizer_metrics)

    @staticmethod
    def _to_float(value: float | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)


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
        unconstrained: bool = False,
        zeropower_coeffs: tuple[float, float, float] | None = None,
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
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
            "zeropower_coeffs": tuple(zeropower_coeffs or MUON_ZEROpower_COEFFS),
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = _ensure_single_beta(group)
            unconstrained = group["unconstrained"]
            norm_factor, backend, backend_steps, eps, norm_kwargs = _resolve_norm_settings(group)

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                _increment_state_step(state, param)
                if beta1 != 1.0:
                    buf = state.setdefault("exp_avg", torch.zeros_like(grad))
                    buf.mul_(1.0 - beta1).add_(grad, alpha=beta1)
                    grad_to_use = buf
                else:
                    grad_to_use = grad

                update = _apply_lmo(
                    grad_to_use,
                    norm_factor=norm_factor,
                    backend=backend,
                    backend_steps=backend_steps,
                    eps=eps,
                    coefficients=tuple(group.get("zeropower_coeffs", self._zeropower_coeffs)),
                    norm_kwargs=norm_kwargs,
                )
                update = _apply_scale_if_configured(update, group)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)

    def _compute_direction(self, param: Tensor, group: dict, state: dict) -> Tensor | None:
        grad = param.grad
        if grad is None:
            return None
        beta1 = _ensure_single_beta(group)
        if beta1 != 1.0:
            buf = state.get("exp_avg")
            return buf if isinstance(buf, torch.Tensor) else grad
        return grad


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
        unconstrained: bool = False,
        zeropower_coeffs: tuple[float, float, float] | None = None,
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
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
            "zeropower_coeffs": tuple(zeropower_coeffs or MUON_ZEROpower_COEFFS),
        }
        super().__init__(params, defaults)
        self._store_grads_in_state()
        self.register_state_dict_pre_hook(type(self)._store_grads_in_state)
        self.register_load_state_dict_post_hook(type(self)._load_grads_from_state)

    def _build_metric_functions(self) -> dict[str, MetricFn]:
        return {
            "l2_norm/moment": _grad_norm_metric,
            "l2_norm/param": _param_norm_metric,
            "l2_norm/update": _update_norm_metric,
        }

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = _ensure_single_beta(group)
            unconstrained = group["unconstrained"]
            norm_factor, backend, backend_steps, eps, norm_kwargs = _resolve_norm_settings(group)

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                _increment_state_step(state, param)
                update = _apply_lmo(
                    grad,
                    norm_factor=norm_factor,
                    backend=backend,
                    backend_steps=backend_steps,
                    eps=eps,
                    coefficients=tuple(group.get("zeropower_coeffs", self._zeropower_coeffs)),
                    norm_kwargs=norm_kwargs,
                )
                update = _apply_scale_if_configured(update, group)
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

    def _compute_direction(self, param: Tensor, group: dict, state: dict) -> Tensor | None:  # noqa: ARG002
        return param.grad


class QHScion(_ScionBase):
    """Quasi-hyperbolic Scion variant."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        *,
        betas: tuple[float, ...] | None = None,
        momentum: float | None = None,
        vs: tuple[float, ...] | Sequence[float] = (1.0,),
        v: float | None = None,
        norm: str = "Auto",
        norm_kwargs: dict | None = None,
        unconstrained: bool = False,
        zeropower_coeffs: tuple[float, float, float] | None = None,
    ) -> None:
        _ensure_positive(lr, "learning rate")
        betas_tuple = _normalize_betas(
            betas=betas,
            fallback=momentum,
            default=(1.0,),
            label="betas",
        )
        vs_tuple = _normalize_vs(vs, fallback=v)
        defaults = {
            "lr": lr,
            "betas": betas_tuple,
            "vs": vs_tuple,
            "v": vs_tuple[0],
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
            "zeropower_coeffs": tuple(zeropower_coeffs or MUON_ZEROpower_COEFFS),
        }
        super().__init__(params, defaults)

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = _ensure_single_beta(group)
            v = _resolve_group_v(group)
            unconstrained = group["unconstrained"]
            norm_factor, backend, backend_steps, eps, norm_kwargs = _resolve_norm_settings(group)

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                _increment_state_step(state, param)
                if beta1 != 1.0:
                    buf = state.setdefault("exp_avg", torch.zeros_like(grad))
                    buf.mul_(1.0 - beta1).add_(grad, alpha=beta1)
                    blended = v * grad + (1.0 - v) * buf
                else:
                    blended = grad

                update = _apply_lmo(
                    blended,
                    norm_factor=norm_factor,
                    backend=backend,
                    backend_steps=backend_steps,
                    eps=eps,
                    coefficients=tuple(group.get("zeropower_coeffs", self._zeropower_coeffs)),
                    norm_kwargs=norm_kwargs,
                )
                update = _apply_scale_if_configured(update, group)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)

    def _compute_direction(self, param: Tensor, group: dict, state: dict) -> Tensor | None:
        grad = param.grad
        if grad is None:
            return None
        beta1 = _ensure_single_beta(group)
        if beta1 != 1.0:
            buf = state.get("exp_avg")
            if buf is None:
                return None
            v = _resolve_group_v(group)
            return v * grad + (1.0 - v) * buf
        return grad


def _build_moment_specs(betas: tuple[float, ...]) -> list[tuple[float, str]]:
    return [(beta, f"exp_avg_{idx}") for idx, beta in enumerate(betas)]


def _normalise_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("Sum of Scion betas weights must be positive.")
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
        unconstrained: bool = False,
        zeropower_coeffs: tuple[float, float, float] | None = None,
    ) -> None:
        betas_tuple = _normalize_betas(
            betas=betas,
            fallback=momentums,
            default=(1.0,),
            label="betas",
        )
        weights = weights or tuple(1.0 for _ in betas_tuple)
        if len(weights) != len(betas_tuple):
            raise ValueError("Scion_weights must match Scion betas length.")
        normalised_weights = _normalise_weights(weights)
        defaults = {
            "lr": lr,
            "betas": betas_tuple,
            "weights": normalised_weights,
            "unconstrained": unconstrained,
            "norm": norm,
            "norm_kwargs": norm_kwargs or {},
            "zeropower_coeffs": tuple(zeropower_coeffs or MUON_ZEROpower_COEFFS),
        }
        super().__init__(params, defaults)

    def _build_metric_functions(self) -> dict[str, MetricFn]:
        metrics: dict[str, MetricFn] = {
            "l2_norm/param": _param_norm_metric,
            "l2_norm/update": _update_norm_metric,
        }
        moment_names: set[str] = set()
        for group in self.param_groups:
            betas_tuple = _ensure_beta_tuple(group, legacy_key="momentums", default=(1.0,))
            for _, name in _build_moment_specs(betas_tuple):
                moment_names.add(name)
        for name in sorted(moment_names):
            metrics[f"l2_norm/{name}"] = _moment_norm_metric(name)
        return metrics

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            betas_tuple = _ensure_beta_tuple(group, legacy_key="momentums", default=(1.0,))
            weights: tuple[float, ...] = group["weights"]
            unconstrained = group["unconstrained"]
            norm_factor, backend, backend_steps, eps, norm_kwargs = _resolve_norm_settings(group)

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                _increment_state_step(state, param)
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

                update = _apply_lmo(
                    blended,
                    norm_factor=norm_factor,
                    backend=backend,
                    backend_steps=backend_steps,
                    eps=eps,
                    coefficients=tuple(group.get("zeropower_coeffs", self._zeropower_coeffs)),
                    norm_kwargs=norm_kwargs,
                )
                update = _apply_scale_if_configured(update, group)
                if not unconstrained:
                    param.data.mul_(1.0 - lr)
                param.data.add_(update, alpha=-lr)

    def init(self) -> None:
        for group in self.param_groups:
            self._init_parameters(group)

    def _compute_direction(self, param: Tensor, group: dict, state: dict) -> Tensor | None:
        grad = param.grad
        if grad is None:
            return None
        betas_tuple = _ensure_beta_tuple(group, legacy_key="momentums", default=(1.0,))
        weights: tuple[float, ...] = group["weights"]
        blended = torch.zeros_like(grad)
        for (beta1, name), weight in zip(_build_moment_specs(betas_tuple), weights):
            if beta1 != 1.0:
                buf = state.get(name)
                if buf is None:
                    continue
                blended.add_(buf, alpha=weight)
            else:
                blended.add_(grad, alpha=weight)
        return blended


# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------


def _zeropower_impl(
    grad: Tensor,
    *,
    steps: int = 5,
    coeffs: tuple[float, float, float] | None = None,
) -> Tensor:
    if grad.ndim != 2:
        msg = f"zeropower expects a 2-D tensor, received shape {grad.shape}"
        raise ValueError(msg)
    a, b, c = coeffs or MUON_ZEROpower_COEFFS
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
    def zeropower_via_newtonschulz5(
        grad: Tensor,
        *,
        steps: int = 5,
        coeffs: tuple[float, float, float] | None = None,
    ) -> Tensor:
        return _zeropower_impl(grad, steps=steps, coeffs=coeffs)

else:  # pragma: no cover - fallback for older torch versions

    def zeropower_via_newtonschulz5(
        grad: Tensor,
        *,
        steps: int = 5,
        coeffs: tuple[float, float, float] | None = None,
    ) -> Tensor:
        return _zeropower_impl(grad, steps=steps, coeffs=coeffs)


def zeroth_power_via_svd(grad: Tensor) -> Tensor:
    u, _, v = torch.linalg.svd(grad)
    return u @ v.T
