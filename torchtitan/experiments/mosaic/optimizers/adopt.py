import math
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _use_grad_for_differentiable,
    _view_as_real,
)
from torch.types import Number


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
    def __init__(
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
        exp_avgs: list[Tensor],
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
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> Tensor | None:
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

        return loss


def _single_tensor_adopt(
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
    has_complex: bool,
) -> None:
    assert grad_scale is None
    assert found_inf is None

    if torch.jit.is_scripting():
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if not torch._utils.is_compiling() and capturable:
            device = _get_capturable_supported_devices()
            assert param.device.type == step_t.device.type
            assert param.device.type in device, (
                f"If capturable=True, params, state_steps must be on: {device}."
            )

        step = step_t if capturable or differentiable else _get_value(step_t)

        if weight_decay != 0 and not decouple:
            decay_factor = (lr / initial_lr) if initial_lr != 0 else 1.0
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

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

        exp_avg.lerp_(normed_grad, 1 - beta1)

        param.add_(exp_avg, alpha=-_get_value(lr))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_t += 1


def _multi_tensor_adopt(
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

    if not torch._utils.is_compiling() and capturable:
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

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps],
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

        if weight_decay != 0 and not decouple:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
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

        if device_state_steps[0] == 0:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)

            if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
                torch._foreach_add_(
                    device_state_steps,
                    torch.tensor(1.0, device="cpu"),
                    alpha=1.0,
                )
            else:
                torch._foreach_add_(device_state_steps, 1)
            continue

        if weight_decay != 0 and decouple:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            weight_decay_unscaled = decay_factor * weight_decay
            torch._foreach_add_(
                device_params,
                device_params,
                alpha=-_get_value(weight_decay_unscaled),
            )

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_maximum_(exp_avg_sq_sqrt, eps)

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)
        if clip_lambda is not None:
            clip = clip_lambda(_get_value(device_state_steps[0]))
            torch._foreach_maximum_(normed_grad, -clip)
            torch._foreach_minimum_(normed_grad, clip)

        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)

        torch._foreach_add_(device_params, device_exp_avgs, alpha=-_get_value(lr))
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs,
            device_grads,
            device_grads,
            value=1 - beta2,
        )

        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps,
                torch.tensor(1.0, device="cpu"),
                alpha=1.0,
            )
        else:
            torch._foreach_add_(device_state_steps, 1)


def _fused_adopt(
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
def adopt(
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

    if foreach and torch.jit.is_scripting():
        msg = "torch.jit.script not supported with foreach optimizers"
        raise RuntimeError(msg)
    if fused and torch.jit.is_scripting():
        msg = "torch.jit.script not supported with fused optimizers"
        raise RuntimeError(msg)

    if fused and not torch.jit.is_scripting():
        func = _fused_adopt
    elif foreach and not torch.jit.is_scripting():
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