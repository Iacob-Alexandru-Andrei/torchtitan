# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AggMo-ADOPT optimizer implementation."""

from __future__ import annotations

import logging
from typing import Any, cast, ClassVar, TYPE_CHECKING

import torch
import torch.distributed.distributed_c10d as c10d
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)
from torch.types import Number


if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

__all__ = ["AggMoAdopt", "aggmoadopt"]


def _default_clip_lambda(step: Number | Tensor) -> float:
    internal_step: int
    if isinstance(step, (Tensor)):
        internal_step = int(step.item())
    elif isinstance(step, (Number)):
        internal_step = int(step)
    else:
        msg = f"step must be a Number or Tensor, but got {type(step)}"
        raise TypeError(msg)
    return internal_step**0.25


class AggMoAdopt(Optimizer):
    """AggMo-ADOPT optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float or Tensor, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square (default: (0.999, 0.9999)).
        vs (Tuple[float, ...], optional): Coefficients used for quasi-hyperbolic
            moment estimates (default: (0.9,)). The number of elements in this tuple
            determines the number of first momenta used.
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-6).
        clip_lambda (callable, optional): A function that maps the step number to
            the gradient clipping value (default: step^0.25).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        decouple (bool, optional): Whether to decouple the weight decay from
            the gradient-based update (default: False).
        foreach (bool, optional): Whether to use the foreach implementation
            (default: None, which means the value of fused is used).
        maximize (bool, optional): Maximize the params based on the objective,
            instead of minimizing (default: False).
        capturable (bool, optional): Whether the optimizer should be capturable
            (default: False). See https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
        differentiable (bool, optional): Whether to make the optimizer
            differentiable (default: False). This allows to compute higher order
            derivatives via autograd, but it incurs a performance penalty.
        fused (bool, optional): Whether to use the fused implementation
            (default: None, which means to use fused if available).
    """

    metric_functions: ClassVar = {
        "l2_norm/moment2": (
            lambda _param, optim_state, _step_tensor: torch.linalg.vector_norm(
                optim_state["exp_avg_sq"],
            )
        ),
        "min/moment2": lambda _param, optim_state, _step_tensor: torch.min(
            optim_state["exp_avg_sq"],
        ),
        "max/moment2": lambda _param, optim_state, _step_tensor: torch.max(
            optim_state["exp_avg_sq"],
        ),
        "l2_norm/param": (
            lambda param, _optim_state, _step_tensor: torch.linalg.vector_norm(
                param.data,
            )
        ),
        "l2_norm/update": (
            lambda _param, _optim_state, step_tensor: torch.linalg.vector_norm(
                step_tensor,
            )
        ),
    }

    def __init__(  # noqa: C901, PLR0913, PLR0912
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.999, 0.9999),
        vs: tuple[float, ...] = (0.9,),
        eps: float = 1e-6,
        clip_lambda: (
            Callable[[Number | Tensor | Any], float] | None
        ) = _default_clip_lambda,
        weight_decay: float = 0.0,
        *,
        decouple: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if isinstance(lr, Tensor) and foreach and not capturable:
            msg = (
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )
            raise ValueError(
                msg,
            )
        if not eps >= 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= betas[0] < 1.0:
            msg = f"Invalid beta parameter at index 0: {betas[0]}"
            raise ValueError(msg)
        if not 0.0 <= betas[1] < 1.0:
            msg = f"Invalid beta parameter at index 1: {betas[1]}"
            raise ValueError(msg)
        if not weight_decay >= 0.0:
            msg = f"Invalid weight_decay value: {weight_decay}"
            raise ValueError(msg)
        # Validate vs parameters
        if not vs:
            msg = "vs must be a non-empty tuple"
            raise ValueError(msg)
        if sum(vs) > 1.0:
            msg = f"The sum of vs coefficients cannot be greater than 1.0, but got {sum(vs)}"
            raise ValueError(msg)
        for i, v in enumerate(vs):
            if not 0.0 <= v <= 1.0:
                msg = (
                    f"Invalid vs parameter at index {i}: {v}. Must be between 0 and 1."
                )
                raise ValueError(msg)

        self.clip_lambda = clip_lambda

        defaults = {
            "lr": lr,
            "betas": betas,
            "vs": vs,
            "eps": eps,
            "weight_decay": weight_decay,
            "decouple": decouple,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

        if fused:
            if differentiable:
                msg = "`fused` does not support `differentiable`"
                raise RuntimeError(msg)
            self._step_supports_amp_scaling = True
            if foreach:
                msg = "`fused` and `foreach` cannot be `True` together."
                raise RuntimeError(msg)
            msg = "`fused` is not currently supported"
            raise RuntimeError(msg)

    def __setstate__(self, state: dict) -> None:
        """Set the optimizer state.

        This method is called when loading a checkpoint.

        Args:
        state (dict): The state dictionary containing the optimizer state.
            This typically includes the state of each parameter and other
            hyperparameters.

        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decouple", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group: dict,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs_list: list[list[Tensor]],
        exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
    ) -> bool:
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                msg = "AggMoAdopt does not support sparse gradients"
                raise RuntimeError(msg)
            grads.append(p.grad)

            state = self.state[p]
            vs = group["vs"]

            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                for i, v in enumerate(vs):
                    if v > 0:
                        state[f"exp_avg_{i+1}"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            for i, v in enumerate(vs):
                if v > 0:
                    exp_avgs_list[i].append(state[f"exp_avg_{i+1}"])

            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Tensor | None:
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            vs = group["vs"]
            exp_avgs_list: list[list[Tensor]] = [[] for _ in vs]
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs_list,
                exp_avg_sqs,
                state_steps,
            )

            aggmoadopt(
                params_with_grad,
                grads,
                exp_avgs_list,
                exp_avg_sqs,
                state_steps,
                initial_lr=group["initial_lr"],
                decouple=group["decouple"],
                clip_lambda=self.clip_lambda,
                beta1=beta1,
                beta2=beta2,
                vs=vs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )

        return loss

    # dist_reduce_metrics, pre_reduce_metrics and report_per_parameter_metrics are omitted for brevity
    # but would need to be adapted for multiple momenta.

def _single_tensor_aggmoadopt(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_list: list[list[Tensor]],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    initial_lr: float | None,
    decouple: bool,
    clip_lambda: Callable[[Number | Tensor | Any], float] | None,
    beta1: float,
    beta2: float,
    vs: tuple[float, ...],
    lr: float | Tensor,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,  # noqa: ARG001
) -> None:
    assert grad_scale is None
    assert found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step = step_t.item()

        if weight_decay != 0 and not decouple:
            grad = grad.add(param, alpha=weight_decay)

        if step == 0:
            exp_avg_sq.addcmul_(grad, grad)
            step_t += 1
            continue

        if weight_decay != 0 and decouple:
            decay_factor = (lr / initial_lr) if initial_lr != 0 else 1.0
            param.mul_(1 - decay_factor * weight_decay)

        denom = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad.div(denom)
        if clip_lambda is not None:
            clip = clip_lambda(step)
            normed_grad.clamp_(-clip, clip)

        # Update momenta
        for j, v in enumerate(vs):
            if v > 0:
                exp_avgs_list[j][i].lerp_(normed_grad, 1 - beta1)

        # Form the update
        update = normed_grad.mul(1 - sum(vs))
        for j, v in enumerate(vs):
            if v > 0:
                update.add_(exp_avgs_list[j][i], alpha=v)

        param.add_(update, alpha=-lr)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_t += 1

def aggmoadopt(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_list: list[list[Tensor]],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    initial_lr: float | None = None,
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    has_complex: bool = False,
    beta1: float,
    beta2: float,
    vs: tuple[float, ...],
    lr: float | Tensor,
    clip_lambda: Callable[[Number | Tensor | Any], float] | None,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
) -> None:
    if foreach or fused:
        raise NotImplementedError("foreach and fused modes are not implemented for AggMoAdopt")

    _single_tensor_aggmoadopt(
        params,
        grads,
        exp_avgs_list,
        exp_avg_sqs,
        state_steps,
        initial_lr=initial_lr,
        decouple=decouple,
        clip_lambda=clip_lambda,
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
    )