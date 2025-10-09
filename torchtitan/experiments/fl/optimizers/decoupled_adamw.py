# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizers with weight decay decoupled from the learning rate.

These optimizers are based off of `Decoupled Weight Decay Regularization
<https://arxiv.org/abs/1711.05101>`_, which proposes this decoupling. In general,
it is recommended to use these optimizers over their native PyTorch equivalents.
"""

from __future__ import annotations

import logging
import math
from typing import ClassVar, TYPE_CHECKING

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor
from torch.optim import AdamW

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)


__all__ = ["DecoupledAdamW"]


class DecoupledAdamW(AdamW):
    """Adam optimizer with the weight decay term decoupled from the learning rate.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate, by default 1e-3.
        betas: Coefficients used for computing running averages of gradient and
            its square, by default (0.9, 0.95).
        eps: Term added to the denominator to improve numerical
            stability, by default 1e-8.
        weight_decay: Decoupled weight decay factor, by default 1e-5.
        amsgrad: Enables the amsgrad variant of Adam, by default False.
        decouple: Whether to decouple the learning rate from the weight decay, by default True.
        foreach: Whether to use the foreach implementation, by default None.
        maximize: Maximize the objective with respect to the params, by default False.
        capturable: Whether this instance is safe to capture in a CUDA graph, by default False.
        differentiable: Whether autograd should occur through the optimizer step, by default False.
        fused: Whether to use the fused implementation, by default None.

    Notes:
        Since `weight_decay` is no longer scaled by `lr`, you will likely want to use
        much smaller values for `weight_decay` than you would if using `torch.optim.Adam`
        or `torch.optim.AdamW`. In this optimizer, the value `weight_decay` translates
        exactly to: 'On every optimizer update, every weight element will be multiplied by
        `(1.0 - weight_decay_t)`'. The term `weight_decay_t` will follow the same schedule
        as `lr_t` but crucially will not be scaled by `lr`.

        Argument defaults are similar to :class:`torch.optim.AdamW` but we make two changes:
        * The default for ``weight_decay`` is changed from ``1e-2`` -> ``1e-5`` because in
        `DecoupledAdamW`, the weight decay is decoupled and no longer scaled by the
        `lr=1e-3`.
        * The default for ``betas`` is changed from ``(0.9, 0.999)`` to ``(0.9, 0.95)``
        to reflect community best-practices for the beta2 hyperparameter.

        Why use this optimizer? The standard
        `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW>`_
        optimizer explicitly couples the weight decay term with the learning rate.
        This ties the optimal value of :attr:`weight_decay` to :attr:`lr` and can also hurt
        generalization in practice. For more details on why decoupling might be desirable,
        see `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_.
    """

    metric_functions: ClassVar = {
        "l2_norm/moment": (
            lambda _param, optim_state, _step_tensor: torch.linalg.vector_norm(
                optim_state["exp_avg"],
            )
        ),
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

    def __init__(  # noqa: PLR0913
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
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
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
            group["decouple"] = decouple
        self.amsgrad = amsgrad

    @staticmethod
    def adamw(  # noqa: PLR0913
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        max_exp_avg_sqs: list[torch.Tensor],
        state_steps: list[torch.Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        initial_lr: float,
        weight_decay: float,
        decouple: bool,
        eps: float,
    ) -> None:
        r"""Functional API that performs AdamW with decoupled weight decay.

        Parameters
        ----------
        params : list[torch.Tensor]
            list of parameters to update.
        grads : list[torch.Tensor]
            list of parameter gradients.
        exp_avgs : list[torch.Tensor]
            list of average gradients.
        exp_avg_sqs : list[torch.Tensor]
            list of average squared gradients.
        max_exp_avg_sqs : list[torch.Tensor]
            list of max average squared gradients for amsgrad updates.
        state_steps : list[torch.Tensor]
            list of steps taken for all parameters.
        amsgrad : bool
             Enables amsgrad variant of Adam.
        beta1 : float
            Coefficient for computing the moving average of gradient values.
        beta2 : float
            Coefficient for computing the moving average of squared gradient values.
        lr : float
            Learning rate.
        initial_lr : float
            Initial learning rate.
        weight_decay : float
            Factor for decoupled weight decay
        decouple : bool
            Whether to decouple the learning rate from the weight decay.
        eps : float
            Term added to the denominator to improve numerical stability.

        """
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i].item()

            # Perform step weight decay
            if weight_decay != 0:
                if decouple:
                    decay_factor = (lr / initial_lr) if initial_lr else 1.0
                    param.mul_(1 - decay_factor * weight_decay)
                else:
                    param.mul_(1 - lr * weight_decay)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(
                    eps,
                )
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()  # type: ignore[reportUntypedFunctionDecorator]
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss, by default None.

        Returns:
            The loss, if the closure is provided and called, otherwise None.

        Raises:
            RuntimeError: AdamW does not support sparse gradients.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            if "initial_lr" not in group:
                group["initial_lr"] = lr
            initial_lr = group["initial_lr"]
            weight_decay = group["weight_decay"]
            decouple = group["decouple"]

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    msg = "AdamW does not support sparse gradients"
                    raise RuntimeError(msg)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = torch.zeros((), dtype=torch.float, device=p.device)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # Update the steps for each param group update
                state["step"] += 1
                # Record the step after step update
                state_steps.append(state["step"])

            self.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                initial_lr=initial_lr,
                weight_decay=weight_decay,
                decouple=decouple,
                eps=eps,
            )

        return loss

    @staticmethod
    def dist_reduce_metrics(  # noqa: C901, PLR0912
        optimizer_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute the optimizer metrics across all workers.

        Args:
            optimizer_metrics: The optimizer metrics per workers.

        Returns:
            The optimizer metrics reduced across all workers.
        """
        local_keys = list(optimizer_metrics.keys())
        all_gathered_keys = [None for _ in range(c10d.get_world_size())]
        c10d.all_gather_object(all_gathered_keys, local_keys)
        all_keys = set()
        for keys in all_gathered_keys:
            if keys is not None:
                all_keys.update(set(keys))

        # Sort keys to ensure every rank has the same keys order
        # Only L2 norm metric keys are present, can apply regular sort
        list_all_keys = sorted(all_keys)

        # Determine device from existing metrics, fallback to default device
        device = None
        for value in optimizer_metrics.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is None:
            # Use default device (respects torch.set_default_device if set)
            device = torch.tensor(0.0).device

        for metric in list_all_keys:
            reduced = optimizer_metrics.get(
                metric,
                torch.tensor(0.0, device=device),
            )

            # Convert DTensor to regular tensor if needed (FSDP2 compatibility)
            # DTensors don't work with c10d.all_reduce directly
            if isinstance(reduced, DTensor):
                reduced = reduced.full_tensor()

            if c10d.get_world_size() > 1:
                if metric.startswith("l2_norm"):
                    c10d.all_reduce(reduced, op=c10d.ReduceOp.SUM)
                    optimizer_metrics[metric] = torch.sqrt(reduced)
                elif metric.startswith("min"):
                    c10d.all_reduce(reduced, op=c10d.ReduceOp.MIN)
                    optimizer_metrics[metric] = reduced
                elif metric.startswith("max"):
                    c10d.all_reduce(reduced, op=c10d.ReduceOp.MAX)
                    optimizer_metrics[metric] = reduced
                else:
                    c10d.all_reduce(reduced, op=c10d.ReduceOp.SUM)
                    optimizer_metrics[metric] = reduced / c10d.get_world_size()
            elif metric.startswith("l2_norm"):
                optimizer_metrics[metric] = torch.sqrt(reduced)
            else:
                optimizer_metrics[metric] = reduced

        return optimizer_metrics

    @staticmethod
    def pre_reduce_metrics(
        optimizer_metrics: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess metrics to reduce across ranks correctly.

        Args:
            optimizer_metrics: The optimizer metrics containing only the L2 norm metrics.

        Returns:
            The optimizer metrics containing the squared L2 norms.
        """
        # Only L2 norm metric keys are present, can skip sorting at this stage
        for metric in optimizer_metrics:
            # L2 norms need to be squared, before they are reduced via summation
            optimizer_metrics[metric] **= 2

        return optimizer_metrics

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Report the per-parameter metrics.

        Args:
            param: The parameter for which to compute metrics.
            name: The name of the parameter to be reported.
            optimizer_metrics: The optimizer metrics.

        Returns:
            The optimizer metrics containing the per-parameter metrics.
        """
        lr = self.param_groups[0]["lr"]
        eps = self.param_groups[0]["eps"]
        weight_decay = self.param_groups[0]["weight_decay"]
        initial_lr = self.param_groups[0]["initial_lr"]
        decouple = self.param_groups[0]["decouple"]

        beta1, beta2 = self.param_groups[0]["betas"]
        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state["step"].item()
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (
                param_optim_state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
            ).add_(eps)
            step_size = lr / bias_correction1
            step_tensor = step_size * param_optim_state["exp_avg"].div(denom)
            # NOTE: This is inverting the AdamW update step to get the actual
            # update step. The original implementation was wrong
            if weight_decay != 0 and decouple:
                decay_factor = (lr / initial_lr) if initial_lr else 1.0
                scaling_factor = (decay_factor * weight_decay) / (
                    1 - decay_factor * weight_decay
                )
                # Apply decoupled weight decay adjustment to the parameter update.
                # This formula implements the decoupled weight decay as described in
                # "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017).
                # The scaling_factor adjusts the update to account for the decoupling of
                # weight decay from the learning rate, ensuring the correct parameter update:
                #   param = param - step_tensor - scaling_factor * (param + step_tensor)
                # which is equivalent to:
                #   step_tensor.mul_(1 + scaling_factor).add_(param, alpha=scaling_factor)
                step_tensor.mul_(1 + scaling_factor).add_(param, alpha=scaling_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f"{metric}/{name}"] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )

        return optimizer_metrics
