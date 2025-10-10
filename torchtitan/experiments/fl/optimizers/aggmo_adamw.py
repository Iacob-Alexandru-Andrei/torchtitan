# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Aggregated-momentum variant of the QHAdamW optimizer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torch.optim.optimizer import (
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _use_grad_for_differentiable,
    _view_as_real,
    _device_dtype_check_for_fused,
    ParamsT,
)

from ._decoupled_decay import compute_decoupled_weight_decay_factor
from .aggmo_adopt import _WEIGHT_SUM_TOL, _build_moment_specs, _is_moment_key, _sum_weights

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
from .qhadamw import QHAdamW


class AggMoAdamW(QHAdamW):
    """QHAdamW optimizer supporting an arbitrary number of first moment buffers."""

    def __init__(  # noqa: PLR0913
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        vs: tuple[float, ...] = (0.7,),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        *,
        amsgrad: bool = False,
        decouple: bool = True,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        self._validate_vs_tuple(vs)
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            vs=vs,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            decouple=decouple,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        self._validate_param_groups()

    def _validate_vs_tuple(self, vs: Sequence[float]) -> None:
        moment_specs = _build_moment_specs(vs)
        total = _sum_weights(moment_specs)
        if total > 1.0 + _WEIGHT_SUM_TOL:
            msg = f"Sum of vs coefficients must be <= 1. Got {total}."
            raise ValueError(msg)

    def _validate_param_groups(self) -> None:
        for group in self.param_groups:
            group_vs = cast("Sequence[float]", group["vs"])
            self._validate_vs_tuple(group_vs)

    def _prepare_param_state(  # noqa: C901
        self,
        group: dict[str, Any],
        param: Tensor,
        moment_specs: Sequence[tuple[float, str]],
    ) -> tuple[list[Tensor], Tensor, Tensor | None, Tensor]:
        state = self.state[param]
        if len(state) == 0:
            if group["fused"]:
                _device_dtype_check_for_fused(param)
            state["step"] = (
                torch.zeros(
                    (),
                    dtype=_get_scalar_dtype(is_fused=group["fused"]),
                    device=param.device,
                )
                if group["capturable"] or group["fused"]
                else torch.tensor(0.0, dtype=_get_scalar_dtype())
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param,
                memory_format=torch.preserve_format,
            )
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
        else:
            if group["amsgrad"] and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
            if not group["amsgrad"] and "max_exp_avg_sq" in state:
                del state["max_exp_avg_sq"]

        if group["differentiable"] and state["step"].requires_grad:
            msg = "`requires_grad` not supported for `step` in differentiable mode"
            raise RuntimeError(msg)

        if (
            group["foreach"]
            and isinstance(group["lr"], Tensor)
            and not group["capturable"]
        ):
            msg = "Tensor lr not supported for capturable=False and foreach=True"
            raise RuntimeError(msg)

        expected_names = [name for _, name in moment_specs]
        for key in list(state.keys()):
            if _is_moment_key(key) and key not in expected_names:
                del state[key]

        buffers: list[Tensor] = []
        for _, name in moment_specs:
            if name not in state:
                state[name] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
            buffers.append(state[name])

        max_exp_avg_sq = state.get("max_exp_avg_sq")
        return buffers, state["exp_avg_sq"], max_exp_avg_sq, state["step"]

    @_use_grad_for_differentiable
    def step(  # type: ignore[override]  # noqa: D102
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> Tensor | None:
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = cast("tuple[float, float]", group["betas"])
            moment_specs = _build_moment_specs(cast("Sequence[float]", group["vs"]))
            grad_coeff = 1.0 - _sum_weights(moment_specs)
            if grad_coeff < -_WEIGHT_SUM_TOL:
                msg = "Sum of vs coefficients must be <= 1 for each parameter group"
                raise ValueError(msg)

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            moment_buffers: list[list[Tensor]] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            has_complex = False

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    msg = "AggMoAdamW does not support sparse gradients"
                    raise RuntimeError(msg)

                has_complex |= torch.is_complex(param)

                buffers, exp_avg_sq, max_exp_avg_sq, step = self._prepare_param_state(
                    group,
                    param,
                    moment_specs,
                )

                params_with_grad.append(param)
                grads.append(param.grad)
                moment_buffers.append(buffers)
                exp_avg_sqs.append(exp_avg_sq)
                state_steps.append(step)
                if group["amsgrad"] and max_exp_avg_sq is not None:
                    max_exp_avg_sqs.append(max_exp_avg_sq)

            if not params_with_grad:
                continue

            aggmo_qhadamw(
                params_with_grad,
                grads,
                moment_buffers,
                exp_avg_sqs,
                max_exp_avg_sqs if group["amsgrad"] else None,
                state_steps,
                initial_lr=group["initial_lr"],
                decouple=group["decouple"],
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                vs=[weight for weight, _ in moment_specs],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
                grad_coeff=grad_coeff,
            )

        return loss

    def report_per_parameter_metrics(  # noqa: D102
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        lr = self.param_groups[0]["lr"]
        eps = self.param_groups[0]["eps"]
        weight_decay = self.param_groups[0]["weight_decay"]
        initial_lr = self.param_groups[0]["initial_lr"]
        decouple = self.param_groups[0]["decouple"]
        beta1, beta2 = self.param_groups[0]["betas"]
        moment_specs = _build_moment_specs(
            cast("Sequence[float]", self.param_groups[0]["vs"])
        )
        grad_coeff = 1.0 - _sum_weights(moment_specs)

        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state["step"].item()
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (
                param_optim_state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
            ).add_(eps)

            numerator = (
                param.grad.mul(grad_coeff)
                if param.grad is not None
                else torch.zeros_like(param)
            )

            for weight, name_key in moment_specs:
                m_hat = param_optim_state[name_key] / bias_correction1
                numerator.add_(m_hat, alpha=weight)

            step_tensor = (numerator / denom) * lr

            if weight_decay != 0 and decouple:
                decay_factor = compute_decoupled_weight_decay_factor(lr, initial_lr)
                effective_weight_decay = decay_factor * weight_decay
                step_tensor = step_tensor.add(param, alpha=effective_weight_decay)

            for metric in self.metric_functions:
                optimizer_metrics[f"{metric}/{name}"] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )

        return optimizer_metrics


def _single_tensor_aggmo_qhadamw(  # noqa: C901, PLR0913, PLR0912
    params: list[Tensor],
    grads: list[Tensor],
    moment_buffers: list[list[Tensor]],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor] | None,
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    initial_lr: float | Tensor | None,
    decouple: bool,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    vs: Sequence[float],
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,  # noqa: ARG001
    grad_coeff: float,
) -> None:
    assert grad_scale is None
    assert found_inf is None

    if torch.jit.is_scripting():  # type: ignore[attr-defined]
        assert isinstance(lr, float)

    moment_specs = _build_moment_specs(vs)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        buffers = moment_buffers[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if not torch._utils.is_compiling() and capturable:  # type: ignore[attr-defined]
            device = _get_capturable_supported_devices()
            assert param.device.type == step_t.device.type
            assert (
                param.device.type in device
            ), f"If capturable=True, params, state_steps must be on: {device}."

        step_t += 1
        step = step_t if capturable or differentiable else _get_value(step_t)

        if weight_decay != 0 and not decouple:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = _view_as_real(grad)
            exp_avg_sq = _view_as_real(exp_avg_sq)
            buffers = [_view_as_real(buf) for buf in buffers]
            param_data = _view_as_real(param)
            if amsgrad and max_exp_avg_sqs is not None:
                max_exp_avg_sqs[i] = _view_as_real(max_exp_avg_sqs[i])
        else:
            param_data = param

        if weight_decay != 0 and decouple:
            decay_factor = compute_decoupled_weight_decay_factor(lr, initial_lr)
            param_data.mul_(1.0 - decay_factor * weight_decay)

        for buf in buffers:
            buf.mul_(beta1).add_(grad, alpha=1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
        else:
            step_val = float(step)
            bias_correction1 = 1.0 - beta1**step_val
            bias_correction2 = 1.0 - beta2**step_val

        bc2_sqrt = (
            torch.sqrt(bias_correction2)  # type: ignore[arg-type]
            if capturable or differentiable
            else math.sqrt(bias_correction2)
        )

        if amsgrad and max_exp_avg_sqs is not None:
            max_exp_avg_sq = max_exp_avg_sqs[i]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)

        m_hats = [buf / bias_correction1 for buf in buffers]
        qh_numerator = grad.mul(grad_coeff)
        for (weight, _), m_hat in zip(moment_specs, m_hats, strict=True):
            qh_numerator.add_(m_hat, alpha=weight)

        param_data.addcdiv_(qh_numerator, denom, value=-_get_value(lr))


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_aggmo_qhadamw)
def aggmo_qhadamw(  # noqa: PLR0913, D103
    params: list[Tensor],
    grads: list[Tensor],
    moment_buffers: list[list[Tensor]],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor] | None,
    state_steps: list[Tensor],
    *,
    initial_lr: float | Tensor | None = None,
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    has_complex: bool = False,
    amsgrad: bool = False,
    beta1: float,
    beta2: float,
    vs: Sequence[float],
    lr: float | Tensor,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    grad_coeff: float,
) -> None:
    if fused:
        msg = "AggMoAdamW does not support fused kernels"
        raise RuntimeError(msg)
    if foreach:
        msg = "AggMoAdamW does not support foreach implementations"
        raise RuntimeError(msg)

    _single_tensor_aggmo_qhadamw(
        params,
        grads,
        moment_buffers,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        initial_lr=initial_lr,
        decouple=decouple,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        vs=vs,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        grad_coeff=grad_coeff,
    )


__all__ = ["AggMoAdamW", "aggmo_qhadamw"]
