# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""ADOPT optimizer implementation."""

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
    _view_as_real,
    Optimizer,
    ParamsT,
)
from torch.types import Number

from ._decoupled_decay import _compute_decay_factor


if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

__all__ = ["ADOPT", "adopt"]


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


class ADOPT(Optimizer):
    """ADOPT optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float or Tensor, optional): Learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.9999)).
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

    def __init__(  # noqa: C901, PLR0913
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
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
            msg = "Tensor lr not supported for capturable=False and foreach=True"
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

        self.clip_lambda = clip_lambda

        defaults = {
            "lr": lr,
            "betas": betas,
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
        """Restore optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
            group.setdefault("decouple", True)
            group.setdefault("initial_lr", group["lr"])
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(),
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
        state_steps: list[Tensor],
    ) -> bool:
        """Initialize optimizer state for parameters in a group."""
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                msg = "ADOPT does not support sparse gradients"
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

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

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
    def step(  # type: ignore[override]
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> Tensor | None:
        """Perform a single optimization step."""
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
            state_steps: list[Tensor] = []
            beta1, beta2 = cast("tuple[float, float]", group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            if grads:
                pre_step_grad_norm = math.sqrt(
                    sum(g.detach().float().pow(2).sum().item() for g in grads)
                )
                log.error("ADOPT pre-step grad norm: %.6f", pre_step_grad_norm)

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                initial_lr=group["initial_lr"],
                decouple=group["decouple"],
                clip_lambda=self.clip_lambda,
                beta1=beta1,
                beta2=beta2,
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

            if grads:
                grad_norm = math.sqrt(
                    sum(g.detach().float().pow(2).sum().item() for g in grads)
                )
                exp_avg_norm = math.sqrt(
                    sum(m.detach().float().pow(2).sum().item() for m in exp_avgs)
                )
                exp_avg_sq_norm = math.sqrt(
                    sum(v.detach().float().pow(2).sum().item() for v in exp_avg_sqs)
                )
                log.error(
                    "ADOPT step stats - grad_norm: %.6f, moment_norm: %.6f, moment2_norm: %.6f",
                    grad_norm,
                    exp_avg_norm,
                    exp_avg_sq_norm,
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
        clip_lambda = self.clip_lambda

        if param in self.state:
            param_optim_state = self.state[param]
            step = param_optim_state["step"].item()

            # Compute ADOPT update (normed gradient with clipping, no quasi-hyperbolic)
            denom = torch.clamp(param_optim_state["exp_avg_sq"].sqrt(), eps)
            if param.grad is not None:
                normed_grad = param.grad.div(denom)
                if clip_lambda is not None:
                    clip_value = clip_lambda(step)
                    normed_grad = normed_grad.clamp(-clip_value, clip_value)
            else:
                normed_grad = torch.zeros_like(param)

            # ADOPT uses exp_avg directly (no quasi-hyperbolic interpolation)
            step_tensor = param_optim_state["exp_avg"] * lr

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


def _single_tensor_adopt(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
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

    if torch.jit.is_scripting():  # type: ignore[attr-defined]
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if not torch._utils.is_compiling() and capturable:  # type: ignore[attr-defined]
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == exp_avg.device.type
            ), f"param and exp_avg must be on same device type, got {param.device.type} and {exp_avg.device.type}"
            assert (
                param.device.type == exp_avg_sq.device.type
            ), f"param and exp_avg_sq must be on same device type, got {param.device.type} and {exp_avg_sq.device.type}"
            assert (
                param.device.type in capturable_supported_devices
            ), f"If capturable=True, params must be on supported devices: {capturable_supported_devices}, got {param.device.type}"

        step = step_t if capturable or differentiable else step_t.item()

        # Apply weight decay if not decoupled
        if weight_decay != 0 and not decouple:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)  # noqa: PLW2901

        # During step 0, only initialize exp_avg_sq and skip parameter update
        if step == 0:
            exp_avg_sq.addcmul_(grad, grad)
            step_t += 1
            continue

        # Apply decoupled weight decay
        if weight_decay != 0 and decouple:
            decay_factor = _compute_decay_factor(lr, initial_lr)
            param.mul_(1 - decay_factor * weight_decay)

        # Compute normalized gradient
        denom = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad / denom

        # Apply clipping
        if clip_lambda is not None:
            clip_value = clip_lambda(step)
            normed_grad.clamp_(-clip_value, clip_value)

        # Update first moment
        exp_avg.lerp_(normed_grad, 1 - beta1)

        # Update parameters
        param.add_(exp_avg, alpha=-lr)  # type: ignore[arg-type]

        # Update second moment
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Update step
        step_t += 1


def _multi_tensor_adopt(  # noqa: C901, PLR0912, PLR0913, PLR0915
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    initial_lr: float | None,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    clip_lambda: Callable[[Number | Tensor | Any], float] | None,
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

    if not torch._utils.is_compiling() and capturable:  # type: ignore[attr-defined]
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

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(  # type: ignore[arg-type]
        cast(
            "list[list[Tensor | None]]",
            [params, grads, exp_avgs, exp_avg_sqs, state_steps],
        ),
    )
    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast("list[Tensor]", device_params_)
        device_grads = cast("list[Tensor]", device_grads_)
        device_exp_avgs = cast("list[Tensor]", device_exp_avgs_)
        device_exp_avg_sqs = cast("list[Tensor]", device_exp_avg_sqs_)
        device_state_steps = cast("list[Tensor]", device_state_steps_)

        if has_complex:
            _view_as_real(
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
            )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        # Apply weight decay if not decoupled
        if weight_decay != 0 and not decouple:
            decay_factor = _compute_decay_factor(lr, initial_lr)
            weight_decay_unscaled = decay_factor * weight_decay
            if maximize:
                torch._foreach_add_(
                    device_grads,
                    device_params,
                    alpha=_get_value(weight_decay_unscaled),
                )
            else:
                device_grads = torch._foreach_add(
                    device_grads,
                    device_params,
                    alpha=-_get_value(weight_decay_unscaled),
                )

        # During step 0, only initialize exp_avg_sq and skip parameter update
        if device_state_steps[0] == 0:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)

            if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:  # type: ignore[attr-defined]
                torch._foreach_add_(
                    device_state_steps,
                    torch.tensor(1.0, device="cpu"),
                    alpha=1.0,
                )
            else:
                torch._foreach_add_(device_state_steps, 1)
            continue

        # Apply decoupled weight decay
        if weight_decay != 0 and decouple:
            decay_factor = _compute_decay_factor(lr, initial_lr)
            weight_decay_unscaled = decay_factor * weight_decay
            torch._foreach_add_(
                device_params,
                device_params,
                alpha=-_get_value(weight_decay_unscaled),
            )

        # Compute normalized gradient
        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        # Use clamp instead of maximum_ to avoid DTensor dispatch issues with scalars
        exp_avg_sq_sqrt = [torch.clamp(t, min=eps) for t in exp_avg_sq_sqrt]

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)

        # Apply clipping
        if clip_lambda is not None:
            clip = clip_lambda(_get_value(device_state_steps[0]))
            # Use clamp instead of maximum_/minimum_ to avoid DTensor dispatch issues
            normed_grad = [torch.clamp(t, min=-clip, max=clip) for t in normed_grad]

        # Update first moment
        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)

        # Update parameters
        torch._foreach_add_(device_params, device_exp_avgs, alpha=-_get_value(lr))

        # Update second moment
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs,
            device_grads,
            device_grads,
            value=1 - beta2,
        )

        # Update step
        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:  # type: ignore[attr-defined]
            torch._foreach_add_(
                device_state_steps,
                torch.tensor(1.0, device="cpu"),
                alpha=1.0,
            )
        else:
            torch._foreach_add_(device_state_steps, 1)


def _fused_adopt(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    initial_lr: float | None,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    clip_lambda: Callable[[int], float] | None,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
) -> None:
    raise NotImplementedError


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adopt)
def adopt(  # noqa: PLR0913
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    initial_lr: float | None = None,
    *,
    foreach: bool | None = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
    grad_scale: Tensor | None = None,
    found_inf: Tensor | None = None,
    has_complex: bool = False,
    beta1: float,
    beta2: float,
    lr: float | Tensor,
    clip_lambda: Callable[[Number | Tensor | Any], float] | None,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
) -> None:
    """Functional API for ADOPT optimizer.

    See :class:`~ADOPT` for details on arguments and behavior.
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

    if foreach and torch.jit.is_scripting():  # type: ignore[attr-defined]
        msg = "torch.jit.script not supported with foreach optimizers"
        raise RuntimeError(msg)
    if fused and torch.jit.is_scripting():  # type: ignore[attr-defined]
        msg = "torch.jit.script not supported with fused optimizers"
        raise RuntimeError(msg)

    if fused and not torch.jit.is_scripting():  # type: ignore[attr-defined]
        func = _fused_adopt
    elif foreach and not torch.jit.is_scripting():  # type: ignore[attr-defined]
        func = _multi_tensor_adopt
    else:
        func = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        initial_lr=initial_lr,
        decouple=decouple,
        clip_lambda=clip_lambda,
        beta1=beta1,
        beta2=beta2,
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
