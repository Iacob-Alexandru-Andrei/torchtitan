# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Thin FL wrapper around :class:`torch.optim.Muon` with metric reporting."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Muon as TorchMuon
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from ._metric_utils import prepare_metrics_for_reduction, reduce_metrics_across_ranks

__all__ = ["Muon"]

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


class Muon(TorchMuon):
    """Muon optimizer with TorchTitan metric reporting hooks."""

    metric_functions: ClassVar[dict[str, MetricFn]] = {
        "l2_norm/moment": _moment_norm_metric("momentum_buffer"),
        "l2_norm/param": _param_norm_metric,
        "l2_norm/update": _update_norm_metric,
    }

    def __init__(  # noqa: PLR0913
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: Optional[str] = "match_rms_adamw",
    ) -> None:
        """Initialize the Muon optimizer wrapper.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            weight_decay: Weight decay coefficient.
            momentum: Muon momentum factor.
            nesterov: Whether to apply Nesterov momentum.
            ns_coefficients: Newton–Schulz polynomial coefficients (a, b, c).
            eps: Numerical stability epsilon for the zeropower operator.
            ns_steps: Number of Newton–Schulz iterations.
            adjust_lr_fn: Optional learning-rate adjustment strategy.
        """
        super().__init__(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Run a Muon optimization step and track per-parameter step counts."""
        loss = super().step(closure)
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state.get(param)
                if state is None:
                    continue
                prev_step = state.get("step", 0)
                if isinstance(prev_step, torch.Tensor):
                    state["step"] = prev_step + 1
                else:
                    state["step"] = int(prev_step) + 1
        return loss

    @staticmethod
    def dist_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reduce metrics across ranks."""
        return reduce_metrics_across_ranks(optimizer_metrics)

    @staticmethod
    def pre_reduce_metrics(optimizer_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Preprocess metrics before reduction."""
        return prepare_metrics_for_reduction(optimizer_metrics)

    def _find_param_group(self, param: Tensor) -> dict[str, Any]:
        for group in self.param_groups:
            for group_param in group["params"]:
                if group_param is param:
                    return group
        return self.param_groups[0]

    def _prepare_step_tensor(
        self,
        *,
        param: Tensor,
        grad: Tensor,
        group: dict[str, Any],
        state: dict[str, Any],
    ) -> Tensor:
        momentum_buffer = state.get("momentum_buffer")
        if momentum_buffer is None:
            return _zero_metric_like(param)

        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])

        if nesterov:
            direction = grad.lerp(momentum_buffer, momentum)
        else:
            direction = momentum_buffer

        if direction.ndim == 2:
            update = _zeropower_via_newtonschulz(
                direction,
                group["ns_coefficients"],
                int(group["ns_steps"]),
                float(group["eps"]),
            )
        else:
            update = direction

        adjusted_lr = _adjust_lr(float(group["lr"]), group.get("adjust_lr_fn"), param.shape)
        step_tensor = update.detach() * adjusted_lr

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
        """Report Muon per-parameter metrics.

        Args:
            param: Parameter tensor to inspect.
            name: Name of the parameter used for metric keys.
            optimizer_metrics: Mapping that accumulates metric values.

        Returns:
            The input ``optimizer_metrics`` with Muon metrics populated for ``param``.
        """
        if param.grad is None:
            return optimizer_metrics

        state = self.state.get(param)
        if state is None:
            return optimizer_metrics

        group = self._find_param_group(param)
        grad = param.grad
        step_tensor = self._prepare_step_tensor(param=param, grad=grad, group=group, state=state)

        if "max/optimizer_step" not in optimizer_metrics:
            step_state = state.get("step", 0)
            if isinstance(step_state, torch.Tensor):
                step_value = step_state.detach().clone()
                if step_value.device != param.device:
                    step_value = step_value.to(param.device)
            else:
                step_value = torch.tensor(float(step_state), device=param.device)
            optimizer_metrics["max/optimizer_step"] = step_value

        for metric_name, metric_fn in self.metric_functions.items():
            optimizer_metrics[f"{metric_name}/{name}"] = metric_fn(param, state, step_tensor)

        return optimizer_metrics
