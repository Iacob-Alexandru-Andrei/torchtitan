# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Quasi-Hyperbolic AdamW optimizer implementation."""

from __future__ import annotations

import logging
import math
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
    Optimizer,
    ParamsT,
)

from ._decoupled_decay import _compute_decay_factor

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

__all__ = ["QHAdamW", "qhadamw"]


class QHAdamW(Optimizer):
    """Quasi-Hyperbolic AdamW optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float or Tensor, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.95)).
        vs (Tuple[float, ...], optional): Coefficients used for quasi-hyperbolic
            moment estimates (default: (0.7,)). Only the first value is used.
        eps (float, optional): Term added to the denominator to improve
            numerical stability (default: 1e-8).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 1e-5).
        amsgrad (bool, optional): Whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
            (default: False).
        decouple (bool, optional): Whether to decouple the weight decay from
            the gradient-based update, as in the AdamW optimizer (default: True).
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

    def __init__(  # noqa: PLR0913, PLR0912, C901
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
        if not vs or len(vs) < 1:
            msg = "vs must be a non-empty tuple with at least one element"
            raise ValueError(msg)
        for i, v in enumerate(vs):
            if not 0.0 <= v <= 1.0:
                msg = (
                    f"Invalid vs parameter at index {i}: {v}. Must be between 0 and 1."
                )
                raise ValueError(msg)

        defaults = {
            "lr": lr,
            "betas": betas,
            "vs": vs,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
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

        Args:
            state (dict): optimizer state. Should be an object returned
                from a call to `self.state_dict()`.

        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decouple", True)
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

    def _init_group(  # noqa: PLR0913
        self,
        group: dict,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        max_exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
    ) -> bool:
        """Initialize the state for a parameter group.

        Args:
            group (dict): The parameter group to initialize.
            params_with_grad (list[Tensor]): List to be populated with parameters that have gradients.
            grads (list[Tensor]): List to be populated with gradients of the parameters.
            exp_avgs (list[Tensor]): List to be populated with exponential moving averages of gradients.
            exp_avg_sqs (list[Tensor]): List to be populated with exponential moving averages of squared gradients.
            max_exp_avg_sqs (list[Tensor]): List to be populated with
                maximum of exponential moving averages of squared gradients (if AMSGrad is used).
            state_steps (list[Tensor]): List to be populated with step counts for each parameter.
        """
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                msg = "QHAdamW does not support sparse gradients"
                raise RuntimeError(msg)
            grads.append(p.grad)

            state = self.state[p]

            if len(state) == 0:
                if group["fused"]:
                    _device_dtype_check_for_fused(p)
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                state["exp_avg"] = torch.zeros_like(
                    p,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p,
                    memory_format=torch.preserve_format,
                )
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            if group["differentiable"] and state["step"].requires_grad:
                msg = "`requires_grad` not supported for `step` in differentiable mode"
                raise RuntimeError(
                    msg,
                )

            if (
                group["foreach"]
                and isinstance(group["lr"], Tensor)
                and not group["capturable"]
            ):
                msg = "Tensor lr not supported for capturable=False and foreach=True"
                raise RuntimeError(
                    msg,
                )

            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> Tensor | None:
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. Optional for most optimizers.

        Returns:
            loss (Tensor, optional): The loss returned by the closure, if provided.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = cast("tuple[float, float]", group["betas"])
            v1, *_ = cast("tuple[float,...]", group["vs"])
            amsgrad = group["amsgrad"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            qhadamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                initial_lr=group["initial_lr"],
                decouple=group["decouple"],
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                v1=v1,
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

    @staticmethod
    def dist_reduce_metrics(  # noqa: PLR0912, C901
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
        v1, *_ = cast("tuple[float,...]", self.param_groups[0]["vs"])

        beta1, beta2 = self.param_groups[0]["betas"]
        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state["step"].item()
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (
                param_optim_state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
            ).add_(eps)
            m_hat = param_optim_state["exp_avg"] / bias_correction1

            # Compute quasi-hyperbolic update
            if param.grad is not None:
                qh_numerator = param.grad.lerp(m_hat, v1)
                step_tensor = (qh_numerator / denom) * lr
            else:
                step_tensor = (m_hat / denom) * lr

            # Apply weight decay adjustment if decoupled
            if weight_decay != 0 and decouple:
                decay_factor = _compute_decay_factor(lr, initial_lr)
                scaling_factor = (decay_factor * weight_decay) / (
                    1 - decay_factor * weight_decay
                )
                step_tensor = (
                    step_tensor * (1 + scaling_factor) + param * scaling_factor
                )

            for metric in self.metric_functions:
                optimizer_metrics[f"{metric}/{name}"] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )

        return optimizer_metrics


def _single_tensor_qhadamw(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    initial_lr: float | None,
    decouple: bool,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    v1: float,
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

    if torch.jit.is_scripting():  # pyright: ignore[reportPrivateImportUsage]
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if (
            not torch._utils.is_compiling()
            and capturable  # pyright: ignore[reportAttributeAccessIssue]
        ):  # pyright: ignore[reportAttributeAccessIssue]
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
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param_data = torch.view_as_real(param)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
        else:
            param_data = param

        if weight_decay != 0 and decouple:
            decay_factor = _compute_decay_factor(lr, initial_lr)
            param_data.mul_(1.0 - decay_factor * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
        else:
            step_val = float(step)
            bias_correction1 = 1.0 - beta1**step_val
            bias_correction2 = 1.0 - beta2**step_val

        bc2_sqrt = (
            torch.sqrt(bias_correction2)  # pyright: ignore[reportArgumentType]
            if capturable or differentiable
            else math.sqrt(bias_correction2)
        )

        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)

        m_hat = exp_avg / bias_correction1
        # Compute lerp manually to avoid DTensor dispatch issues with scalar v1
        # Equivalent to: grad.lerp(m_hat, v1)
        qh_numerator = grad * (1.0 - v1) + m_hat * v1
        param_data.addcdiv_(qh_numerator, denom, value=-_get_value(lr))


def _multi_tensor_qhadamw(  # noqa: C901, PLR0913, PLR0912, PLR0915
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    initial_lr: float | None,
    has_complex: bool,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    v1: float,
    lr: float | Tensor,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
) -> None:
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        msg = "lr as a Tensor is not supported for capturable=False and foreach=True"
        raise RuntimeError(
            msg,
        )

    if (
        not torch._utils.is_compiling()
        and capturable  # pyright: ignore[reportAttributeAccessIssue]
    ):
        device = _get_capturable_supported_devices(
            supports_xla=False,
        )
        assert all(
            p.device.type == step.device.type and p.device.type in device
            for p, step in zip(params, state_steps, strict=False)
        ), f"If capturable=True, params, state_steps must be on: {device}."

    assert not differentiable, "_foreach ops don't support autograd"
    assert grad_scale is None
    assert found_inf is None

    tensor_lists: list[list[Tensor | None]] = cast(
        "list[list[Tensor | None]]", [params, grads, exp_avgs, exp_avg_sqs, state_steps]
    )
    if amsgrad:
        tensor_lists.insert(4, cast("list[Tensor | None]", max_exp_avg_sqs))

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        tensor_lists,
    )

    for device_tuple, _ in grouped_tensors.values():
        if amsgrad:
            (
                device_params_,
                device_grads_,
                device_exp_avgs_,
                device_exp_avg_sqs_,
                device_max_exp_avg_sqs_,
                device_state_steps_,
            ) = device_tuple
            device_max_exp_avg_sqs = cast("list[Tensor]", device_max_exp_avg_sqs_)
        else:
            (
                device_params_,
                device_grads_,
                device_exp_avgs_,
                device_exp_avg_sqs_,
                device_state_steps_,
            ) = device_tuple
            device_max_exp_avg_sqs = []

        device_params = cast("list[Tensor]", device_params_)
        device_grads = cast("list[Tensor]", device_grads_)
        device_exp_avgs = cast("list[Tensor]", device_exp_avgs_)
        device_exp_avg_sqs = cast("list[Tensor]", device_exp_avg_sqs_)
        device_state_steps = cast("list[Tensor]", device_state_steps_)

        if (
            not torch._utils.is_compiling()  # pyright: ignore[reportAttributeAccessIssue]
            and device_state_steps[0].is_cpu
        ):
            torch._foreach_add_(
                device_state_steps,
                torch.tensor(1.0, device="cpu"),
                alpha=1.0,
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if has_complex:
            args_to_view = [
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
            ]
            if amsgrad:
                args_to_view.append(device_max_exp_avg_sqs)
            for j, t in enumerate(args_to_view):
                for i in range(len(t)):
                    args_to_view[j][i] = torch.view_as_real(t[i])

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if weight_decay != 0 and not decouple:
            if maximize:
                torch._foreach_add_(
                    device_grads,
                    device_params,
                    alpha=weight_decay,
                )
            else:
                device_grads = torch._foreach_add(
                    device_grads,
                    device_params,
                    alpha=weight_decay,
                )

        if weight_decay != 0 and decouple:
            decay_factor = _compute_decay_factor(lr, initial_lr)

            weight_decay_term = (
                decay_factor * weight_decay
                if capturable
                else _get_value(decay_factor) * weight_decay
            )
            torch._foreach_mul_(device_params, 1.0 - weight_decay_term)

        torch._foreach_mul_(device_exp_avgs, beta1)
        torch._foreach_add_(device_exp_avgs, device_grads, alpha=1 - beta1)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs,
            device_grads,
            device_grads,
            value=1 - beta2,
        )

        step = device_state_steps[0]

        if capturable:
            bias_correction1 = 1.0 - torch.pow(beta1, step)
            bias_correction2 = 1.0 - torch.pow(beta2, step)
            bc2_sqrt = torch.sqrt(bias_correction2)
        else:
            step_val = float(_get_value(step))
            bias_correction1 = 1.0 - beta1**step_val
            bias_correction2 = 1.0 - beta2**step_val
            bc2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
            exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

        torch._foreach_div_(exp_avg_sq_sqrt, bc2_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)

        m_hats = torch._foreach_div(device_exp_avgs, bias_correction1)
        # Compute lerp manually to avoid DTensor dispatch issues with scalar v1
        # Equivalent to: lerp(device_grads, m_hats, v1)
        qh_numerators = torch._foreach_mul(device_grads, 1.0 - v1)
        torch._foreach_add_(qh_numerators, m_hats, alpha=v1)

        if capturable:
            torch._foreach_addcdiv_(
                device_params,
                qh_numerators,
                exp_avg_sq_sqrt,
                value=-_get_value(lr),
            )
        else:
            torch._foreach_addcdiv_(
                device_params,
                qh_numerators,
                exp_avg_sq_sqrt,
                value=-_get_value(lr),
            )


def _fused_qhadamw(
    *args: Any,
    **kwargs: Any,
) -> None:
    msg = "Fused QHAdamW is not implemented."
    raise NotImplementedError(msg)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_qhadamw)
def qhadamw(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
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
    amsgrad: bool,
    beta1: float,
    beta2: float,
    v1: float,
    lr: float | Tensor,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
) -> None:
    """Functional API that performs QHAdamW algorithm computation.

    Args:
        params (list[Tensor]): List of parameters to optimize.
        grads (list[Tensor]): List of gradients for the parameters.
        exp_avgs (list[Tensor]): List of exponential moving averages of gradients.
        exp_avg_sqs (list[Tensor]): List of exponential moving averages of squared gradients.
        max_exp_avg_sqs (list[Tensor]): List of maximum of exponential moving averages of
            squared gradients (if AMSGrad is used).
        state_steps (list[Tensor]): List of step counts for each parameter.
        initial_lr (float, optional): Initial learning rate.
        foreach (bool, optional): Whether to use the foreach implementation.
        capturable (bool, optional): Whether the optimizer should be capturable.
        differentiable (bool, optional): Whether to make the optimizer differentiable.
        fused (bool, optional): Whether to use the fused implementation.
        grad_scale (Tensor, optional): Gradient scale for mixed precision training.
        found_inf (Tensor, optional): Tensor indicating if any inf/nan values were found in
            gradients.
        has_complex (bool, optional): Whether any of the parameters are complex tensors.
        amsgrad (bool): Whether to use the AMSGrad variant of this algorithm.
        beta1 (float): Coefficient used for computing running averages of gradient.
        beta2 (float): Coefficient used for computing running averages of squared gradient.
        v1 (float): Coefficient used for quasi-hyperbolic moment estimates.
        lr (float or Tensor): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
        decouple (bool): Whether to decouple the weight decay from the gradient-based update.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the params based on the objective, instead of minimizing.
    """
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params,
            differentiable,
            use_fused=False,
        )
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if (
        foreach and torch.jit.is_scripting()
    ):  # pyright: ignore[reportPrivateImportUsage]
        msg = "torch.jit.script not supported with foreach optimizers"
        raise RuntimeError(msg)
    if fused and torch.jit.is_scripting():  # pyright: ignore[reportPrivateImportUsage]
        msg = "torch.jit.script not supported with fused optimizers"
        raise RuntimeError(msg)

    if (
        fused and not torch.jit.is_scripting()
    ):  # pyright: ignore[reportPrivateImportUsage]
        func = _fused_qhadamw
    elif (
        foreach and not torch.jit.is_scripting()
    ):  # pyright: ignore[reportPrivateImportUsage]
        func = _multi_tensor_qhadamw
    else:
        func = _single_tensor_qhadamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        initial_lr=initial_lr,
        decouple=decouple,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        v1=v1,
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
