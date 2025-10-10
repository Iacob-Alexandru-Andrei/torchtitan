# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""AggMo-AdamW optimizer implementation."""

from __future__ import annotations

import logging
import math
from typing import Any, cast, ClassVar, TYPE_CHECKING

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT, _use_grad_for_differentiable


if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

__all__ = ["AggMoAdamW", "aggmoadamw"]


class AggMoAdamW(Optimizer):
    """AggMo-AdamW optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float or Tensor, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        vs (Tuple[float, ...], optional): Coefficients used for quasi-hyperbolic
            moment estimates (default: (0.9,)). The number of elements in this tuple
            determines the number of first momenta used.
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 1e-2).
        amsgrad (bool, optional): Whether to use the AMSGrad variant of this
            algorithm (default: False).
        decouple (bool, optional): Whether to decouple the weight decay from
            the gradient-based update, as in the AdamW optimizer (default: True).
        foreach (bool, optional): Whether to use the foreach implementation
            (default: None).
        maximize (bool, optional): Maximize the params based on the objective,
            instead of minimizing (default: False).
        capturable (bool, optional): Whether the optimizer should be capturable
            (default: False).
        differentiable (bool, optional): Whether to make the optimizer
            differentiable (default: False).
    """

    def __init__(  # noqa: C901, PLR0913, PLR0912
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        vs: tuple[float, ...] = (0.9,),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        amsgrad: bool = False,
        decouple: bool = True,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ) -> None:
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not vs:
            raise ValueError("vs must be a non-empty tuple")
        if sum(vs) > 1.0:
            raise ValueError(f"The sum of vs coefficients cannot be greater than 1.0, but got {sum(vs)}")
        for i, v in enumerate(vs):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"Invalid vs parameter at index {i}: {v}. Must be between 0 and 1.")

        defaults = dict(
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
        )
        super().__init__(params, defaults)

    def _init_group(
        self,
        group: dict,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs_list: list[list[Tensor]],
        exp_avg_sqs: list[Tensor],
        max_exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
    ) -> bool:
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AggMoAdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            vs = group["vs"]

            # State initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                for i, v in enumerate(vs):
                    if v > 0:
                        state[f"exp_avg_{i+1}"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            for i, v in enumerate(vs):
                if v > 0:
                    exp_avgs_list[i].append(state[f"exp_avg_{i+1}"])

            exp_avg_sqs.append(state["exp_avg_sq"])
            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
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
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]
            amsgrad = group["amsgrad"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs_list,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            aggmoadamw(
                params_with_grad,
                grads,
                exp_avgs_list,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                vs=vs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decouple=group["decouple"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )
        return loss

def _single_tensor_aggmoadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_list: list[list[Tensor]],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    vs: tuple[float, ...],
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    decouple: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        if maximize:
            grad = -grad

        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if weight_decay != 0 and not decouple:
            grad = grad.add(param, alpha=weight_decay)

        # update step
        step_t += 1
        step = step_t.item()

        # Perform stepweight decay
        if weight_decay != 0 and decouple:
            param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        for j, v in enumerate(vs):
            if v > 0:
                exp_avgs_list[j][i].mul_(beta1).add_(grad, alpha=1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        # Form the update
        update = grad.mul(1 - sum(vs))
        for j, v in enumerate(vs):
            if v > 0:
                m_hat = exp_avgs_list[j][i] / bias_correction1
                update.add_(m_hat, alpha=v)

        param.addcdiv_(update, denom, value=-lr)

def aggmoadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_list: list[list[Tensor]],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    vs: tuple[float, ...],
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    decouple: bool,
    foreach: bool | None,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    if foreach or capturable or differentiable:
        raise NotImplementedError("foreach, capturable, and differentiable modes are not implemented for AggMoAdamW")

    _single_tensor_aggmoadamw(
        params,
        grads,
        exp_avgs_list,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        vs=vs,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decouple=decouple,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )