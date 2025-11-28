# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Aggregated-momentum variant of the Muon optimizer."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable, Sequence
import time

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim import Muon as TorchMuon
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from .aggmo_adopt import _build_moment_specs, _sum_weights, _WEIGHT_SUM_TOL
from ._metric_utils import prepare_metrics_for_reduction, reduce_metrics_across_ranks

MetricFn = Callable[[Tensor, dict[str, Any], Tensor], Tensor]

__all__ = ["AggMoMuon"]


def _zero_metric_like(param: Tensor) -> Tensor:
    dtype = param.dtype if param.is_floating_point() else torch.float32
    return torch.zeros((), device=param.device, dtype=dtype)


def _param_norm_metric(param: Tensor, _state: dict[str, Any], _step_tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(param.detach())


def _update_norm_metric(_param: Tensor, _state: dict[str, Any], step_tensor: Tensor) -> Tensor:
    return torch.linalg.vector_norm(step_tensor)


class AggMoMuon(TorchMuon):
    """Muon optimizer that aggregates multiple first-moment buffers."""

    metric_functions: ClassVar[dict[str, MetricFn]] = {
        "l2_norm/param": _param_norm_metric,
        "l2_norm/update": _update_norm_metric,
    }

    def __init__(  # noqa: PLR0913
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, ...] = (0.95,),
        vs: tuple[float, ...] = (0.7,),
        weight_decay: float = 0.1,
        *,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "match_rms_adamw",
    ) -> None:
        if len(betas) != len(vs):
            msg = f"Length of betas must equal length of vs. Got {len(betas)} betas for {len(vs)} vs."
            raise ValueError(msg)
        if _sum_weights(_build_moment_specs(vs)) > 1.0 + _WEIGHT_SUM_TOL:
            msg = "Sum of vs coefficients must be <= 1."
            raise ValueError(msg)
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=betas[0] if betas else 0.0,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        # Override defaults for AggMo-specific fields.
        for group in self.param_groups:
            group["betas"] = tuple(float(b) for b in betas)
            group["vs"] = tuple(float(v) for v in vs)
            group["nesterov"] = bool(nesterov)
        self._setup_metric_functions(vs)

    def _setup_metric_functions(self, vs: Sequence[float]) -> None:
        moment_specs = _build_moment_specs(vs)
        metrics: dict[str, MetricFn] = dict(self.metric_functions)
        for _, name in moment_specs:
            metrics[f"l2_norm/{name}"] = (
                lambda _param, state, _step_tensor, key=name: torch.linalg.vector_norm(state.get(key, _zero_metric_like(_param)))
            )
        self.metric_functions = metrics

    def _prepare_state(self, param: Tensor, moment_specs: Sequence[tuple[float, str]]) -> list[Tensor]:
        state = self.state[param]
        buffers: list[Tensor] = []
        if "step" not in state:
            state["step"] = torch.tensor(0.0, device=param.device)
        for _, name in moment_specs:
            if name not in state:
                state[name] = torch.zeros_like(param, memory_format=torch.preserve_format)
            buffers.append(state[name])
        return buffers

    def _compute_direction(
        self,
        grad: Tensor,
        buffers: Sequence[Tensor],
        betas: Sequence[float],
        moment_specs: Sequence[tuple[float, str]],
        grad_coeff: float,
        nesterov: bool,
    ) -> Tensor:
        for buf, beta in zip(buffers, betas, strict=True):
            buf.lerp_(grad, 1 - beta)

        direction = grad.mul(grad_coeff)
        for (weight, _), buf, beta in zip(moment_specs, buffers, betas, strict=True):
            blended = grad.lerp(buf, beta) if nesterov else buf
            direction.add_(blended, alpha=weight)
        return direction

    @torch.no_grad()
    @Optimizer.profile_hook_step
    def step(self, closure: Callable[[], torch.Tensor] | None = None):  # type: ignore[override]  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            betas: tuple[float, ...] = group["betas"]
            vs: tuple[float, ...] = group["vs"]
            moment_specs = _build_moment_specs(vs)
            grad_coeff = 1.0 - _sum_weights(moment_specs)
            if grad_coeff < -_WEIGHT_SUM_TOL:
                msg = "Sum of vs coefficients must be <= 1 for each parameter group."
                raise ValueError(msg)

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            nesterov = bool(group["nesterov"])
            ns_coefficients = group["ns_coefficients"]
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group.get("adjust_lr_fn")

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    msg = "AggMoMuon does not support sparse gradients."
                    raise RuntimeError(msg)
                if grad.ndim != 2:
                    msg = "AggMoMuon expects 2-D parameters."
                    raise ValueError(msg)

                buffers = self._prepare_state(param, moment_specs)
                direction = self._compute_direction(grad, buffers, betas, moment_specs, grad_coeff, nesterov)
                update = _zeropower_via_newtonschulz(direction, ns_coefficients, ns_steps, eps)
                adjusted_lr = _adjust_lr(float(lr), adjust_lr_fn, param.shape)

                if weight_decay != 0.0:
                    param.mul_(1 - float(lr) * float(weight_decay))
                param.add_(update, alpha=-adjusted_lr)

                state = self.state[param]
                state["step"] = state.get("step", torch.tensor(0.0, device=param.device)) + 1
        return loss

    def _build_step_tensor(
        self,
        param: Tensor,
        grad: Tensor,
        group: dict[str, Any],
        buffers: Sequence[Tensor],
        betas: Sequence[float],
        moment_specs: Sequence[tuple[float, str]],
        grad_coeff: float,
    ) -> Tensor:
        direction = self._compute_direction(
            grad,
            buffers,
            betas,
            moment_specs,
            grad_coeff,
            bool(group["nesterov"]),
        )
        update = _zeropower_via_newtonschulz(
            direction,
            group["ns_coefficients"],
            int(group["ns_steps"]),
            float(group["eps"]),
        )
        step_tensor = update.detach() * _adjust_lr(float(group["lr"]), group.get("adjust_lr_fn"), param.shape)
        weight_decay = float(group.get("weight_decay", 0.0))
        if weight_decay != 0.0:
            step_tensor = step_tensor + param.detach() * (weight_decay * float(group["lr"]))
        return step_tensor

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        optimizer_label = type(self).__name__
        if param.grad is None:
            return optimizer_metrics
        state = self.state.get(param)
        if state is None:
            return optimizer_metrics

        group = self.param_groups[0]
        moment_specs = _build_moment_specs(group["vs"])
        grad_coeff = 1.0 - _sum_weights(moment_specs)
        buffers = [state.get(name, torch.zeros_like(param)) for _, name in moment_specs]
        step_tensor = self._build_step_tensor(param, param.grad, group, buffers, group["betas"], moment_specs, grad_coeff)

        step_key = f"max/{optimizer_label}/optimizer_step"
        if step_key not in optimizer_metrics:
            step_state = state.get("step", 0)
            if isinstance(step_state, torch.Tensor):
                step_value = step_state.detach().clone()
                if step_value.device != param.device:
                    step_value = step_value.to(param.device)
            else:
                step_value = torch.tensor(float(step_state), device=param.device)
            optimizer_metrics[step_key] = step_value

        for metric_name, metric_fn in self.metric_functions.items():
            key = f"{metric_name}/{optimizer_label}/{name}"
            exp_avg = state.get("exp_avg")
            ptr = exp_avg.data_ptr() if isinstance(exp_avg, torch.Tensor) else None
            optimizer_metrics[key] = metric_fn(param, state, step_tensor)
            now = time.time()
            print(
                f"[DESLOC DEBUG] metrics read param={name} owner={optimizer_label} metric={metric_name} "
                f"norm={optimizer_metrics[key] if torch.is_tensor(optimizer_metrics[key]) else optimizer_metrics[key]} "
                f"exp_avg_ptr={ptr} step_tensor={step_tensor.item() if isinstance(step_tensor, torch.Tensor) else step_tensor} time={now}"
            )
        return optimizer_metrics

    @staticmethod
    def dist_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reduce metrics across ranks."""
        return reduce_metrics_across_ranks(optimizer_metrics)

    @staticmethod
    def pre_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess metrics before reduction."""
        return prepare_metrics_for_reduction(optimizer_metrics)
