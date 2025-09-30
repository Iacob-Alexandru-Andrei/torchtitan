"""Implementation of the QHADOPT optimizer for Mosaic/TorchTitan examples."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
from composer.utils import dist
from llmfoundry.registry import optimizers
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

__all__ = ["QHADOPT", "qhadopt", "get_report_curvature"]


def get_report_curvature() -> Callable[[Tensor, str], dict[str, Tensor]]:
    """Factory that mirrors Photon curvature reporting helpers."""

    prev_params: dict[str, Tensor] = {}
    prev_grads: dict[str, Tensor] = {}

    def report_curvature(param: Tensor, name: str) -> dict[str, Tensor]:
        if param.grad is None:
            return {}
        if name not in prev_params or name not in prev_grads:
            prev_params[name] = param.detach().clone().cpu()
            prev_grads[name] = param.grad.detach().clone().cpu()
            return {}

        grad_diff = param.grad - prev_grads[name].to(
            device=param.grad.device,
            dtype=param.grad.dtype,
        )
        param_diff = param - prev_params[name].to(
            device=param.device,
            dtype=param.dtype,
        )

        param_diff_norm = torch.linalg.vector_norm(param_diff)
        grad_diff_norm = torch.linalg.vector_norm(grad_diff)

        long_bb = param_diff_norm**2.0 / torch.sum(torch.mul(grad_diff, param_diff))
        short_bb = torch.sum(torch.mul(grad_diff, param_diff)) / grad_diff_norm**2.0

        second_derivative_estimate = grad_diff / param_diff
        second_to_first_ratio = torch.linalg.vector_norm(
            second_derivative_estimate / param.grad
        )

        prev_params[name] = param.detach().clone().cpu()
        prev_grads[name] = param.grad.detach().clone().cpu()

        return {
            f"curvature/param_diff_norm/{name}": param_diff_norm,
            f"curvature/grad_diff_norm/{name}": grad_diff_norm,
            f"curvature/long_bb/{name}": long_bb,
            f"curvature/short_bb/{name}": short_bb,
            f"curvature/l2_norm/second_to_first_derivative_estimate_ratio/{name}": (
                second_to_first_ratio
            ),
            f"curvature/l2_norm/second_derivative_estimate/{name}": (
                torch.linalg.vector_norm(second_derivative_estimate)
            ),
        }

    return report_curvature


def _default_clip_lambda(step: Number | Tensor) -> float:
    if isinstance(step, Tensor):
        step_value = int(step.item())
    elif isinstance(step, Number):
        step_value = int(step)
    else:
        msg = f"step must be a Number or Tensor, but got {type(step)}"
        raise TypeError(msg)
    return step_value**0.25


@optimizers.register_class("qhadopt")
class QHADOPT(Optimizer):
    """Quasi-hyperbolic ADOPT optimizer with Composer metric hooks."""

    metric_functions: ClassVar[dict[str, Callable[[Tensor, dict[str, Any], Tensor], Tensor]]] = {
        "l2_norm/moment": lambda _param, optim_state, step_tensor: torch.linalg.vector_norm(
            optim_state["exp_avg"]
        ),
        "l2_norm/moment2": lambda _param, optim_state, step_tensor: torch.linalg.vector_norm(
            optim_state["exp_avg_sq"]
        ),
        "min/moment2": lambda _param, optim_state, step_tensor: torch.min(
            optim_state["exp_avg_sq"]
        ),
        "max/moment2": lambda _param, optim_state, step_tensor: torch.max(
            optim_state["exp_avg_sq"]
        ),
        "l2_norm/param": lambda param, _optim_state, step_tensor: torch.linalg.vector_norm(
            param.data
        ),
        "l2_norm/update": lambda _param, _optim_state, step_tensor: torch.linalg.vector_norm(
            step_tensor
        ),
        "l2_norm/grad": lambda param, _optim_state, _step_tensor: torch.linalg.vector_norm(
            param.grad
        ),
    }

    def __init__(  # noqa: PLR0917, PLR0913, C901
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.999, 0.9999),
        v1: float = 0.9,
        eps: float = 1e-6,
        clip_lambda: Callable[[Number | Tensor], float] | None = _default_clip_lambda,
        weight_decay: float = 0.0,
        *,
        decouple: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        report_curvature: bool = False,
    ) -> None:
        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if isinstance(lr, Tensor) and foreach and not capturable:
            msg = "lr as a Tensor is not supported for capturable=False and foreach=True"
            raise ValueError(msg)
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
            "vs": (v1,),
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
            if foreach:
                msg = "`fused` and `foreach` cannot be `True` together."
                raise RuntimeError(msg)
            msg = "`fused` is not currently supported"
            raise RuntimeError(msg)

        self.curvature_metric_function: Callable[[Tensor, str], dict[str, Tensor]] | None = None
        if report_curvature:
            self.curvature_metric_function = get_report_curvature()

    def __setstate__(self, state: dict[str, Any]) -> None:  # noqa: D401
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decouple", False)
            fused = group.setdefault("fused", None)
            for param in group["params"]:
                p_state = self.state.get(param, {})
                if p_state and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=param.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group: dict[str, Any],
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
    ) -> bool:
        has_complex = False
        for param in group["params"]:
            if param.grad is None:
                continue
            has_complex |= torch.is_complex(param)
            params_with_grad.append(param)
            if param.grad.is_sparse:
                msg = "QHADOPT does not support sparse gradients"
                raise RuntimeError(msg)
            grads.append(param.grad)

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
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

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

            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(  # type: ignore # noqa: PGH003
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
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
            v1, *_ = cast("tuple[float, ...]", group["vs"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            qhadopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                initial_lr=group.get("initial_lr"),
                decouple=group["decouple"],
                clip_lambda=self.clip_lambda,
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

    def dist_reduce_metrics(self, optimizer_metrics: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: PLR6301
        if dist.get_world_size() <= 1:
            return optimizer_metrics

        local_keys = list(optimizer_metrics.keys())
        all_gathered_keys = dist.all_gather_object(local_keys)
        all_keys: set[str] = set()
        for keys in all_gathered_keys:
            all_keys.update(keys)

        for metric in sorted(all_keys):
            if metric.startswith("l2_norm"):
                reduced = optimizer_metrics.get(
                    metric,
                    torch.tensor(0.0, device=torch.cuda.current_device()),
                )
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="SUM")
                optimizer_metrics[metric] = math.sqrt(reduced)
            elif metric.startswith("min"):
                reduced = optimizer_metrics.get(
                    metric,
                    torch.tensor(0.0, device=torch.cuda.current_device()),
                )
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="MIN")
                optimizer_metrics[metric] = reduced
            elif metric.startswith("max"):
                reduced = optimizer_metrics.get(
                    metric,
                    torch.tensor(0.0, device=torch.cuda.current_device()),
                )
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="MAX")
                optimizer_metrics[metric] = reduced
            else:
                reduced = optimizer_metrics.get(
                    metric,
                    torch.tensor(0.0, device=torch.cuda.current_device()),
                )
                if dist.get_world_size() > 1:
                    dist.all_reduce(reduced, reduce_operation="SUM")
                optimizer_metrics[metric] = reduced / dist.get_world_size()

        return optimizer_metrics

    def pre_reduce_metrics(self, optimizer_metrics: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: PLR6301
        for metric in optimizer_metrics:
            optimizer_metrics[metric] **= 2
        return optimizer_metrics

    def report_per_parameter_metrics(
        self,
        param: torch.Tensor,
        name: str,
        optimizer_metrics: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if param not in self.state:
            return optimizer_metrics

        grad = param.grad
        if grad is None:
            return optimizer_metrics

        param_state = self.state[param]
        exp_avg = cast(Tensor, param_state["exp_avg"])
        exp_avg_sq = cast(Tensor, param_state["exp_avg_sq"])
        step = cast(Tensor, param_state["step"])

        old_exp_avg_sq = (exp_avg_sq - grad**2 * (1 - self.param_groups[0]["betas"][1])) / self.param_groups[0]["betas"][1]
        denom = torch.clamp(old_exp_avg_sq.sqrt(), self.param_groups[0]["eps"])
        normed_grad = grad.div(denom)

        if self.clip_lambda is not None:
            clip = self.clip_lambda(step)
            normed_grad.clamp_(-clip, clip)

        v1 = self.param_groups[0]["vs"][0]
        normed_grad.mul_(1 - v1)
        normed_grad.add_(exp_avg, alpha=v1)

        step_tensor = self.param_groups[0]["lr"] * normed_grad
        weight_decay = self.param_groups[0]["weight_decay"]
        if weight_decay != 0 and self.param_groups[0]["decouple"]:
            initial_lr = self.param_groups[0]["initial_lr"]
            decay_factor = (self.param_groups[0]["lr"] / initial_lr) if initial_lr else 1.0
            scaling_factor = (decay_factor * weight_decay) / (1 - decay_factor * weight_decay)
            step_tensor.mul_(1 + scaling_factor).add_(param, alpha=scaling_factor)

        for metric_name, metric_fn in self.metric_functions.items():
            optimizer_metrics[f"{metric_name}/{name}"] = metric_fn(param, param_state, step_tensor)

        if self.curvature_metric_function is not None:
            optimizer_metrics.update(self.curvature_metric_function(param, name))

        return optimizer_metrics


def _single_tensor_qhadopt(
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
    v1: float,
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

    if torch.jit.is_scripting():  # type: ignore[PGH003]
        assert isinstance(lr, float)

    for param, grad, exp_avg, exp_avg_sq, step_t in zip(
        params, grads, exp_avgs, exp_avg_sqs, state_steps, strict=False
    ):
        grad = grad if not maximize else -grad

        if not torch._utils.is_compiling() and capturable:  # type: ignore[reportGeneralTypeIssues]
            device = _get_capturable_supported_devices()
            assert param.device.type == step_t.device.type
            assert param.device.type in device

        step = step_t if capturable or differentiable else _get_value(step_t)

        if weight_decay != 0 and not decouple:
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
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            param.mul_(1 - decay_factor * weight_decay)

        denom = torch.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad.div(denom)
        if clip_lambda is not None:
            clip = clip_lambda(step)
            normed_grad.clamp_(-clip, clip)

        exp_avg.lerp_(normed_grad, 1 - beta1)
        normed_grad.mul_(1 - v1)
        normed_grad.add_(exp_avg, alpha=v1)
        param.add_(normed_grad, alpha=-_get_value(lr))
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        step_t += 1


def _multi_tensor_qhadopt(
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
    v1: float,
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
        raise RuntimeError(msg)
    assert grad_scale is None
    assert found_inf is None
    assert not differentiable

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(  # noqa: SLF001
        [params, grads, exp_avgs, exp_avg_sqs, state_steps],
    )
    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_state_steps,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params)
        device_grads = cast(list[Tensor], device_grads)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs)
        device_state_steps = cast(list[Tensor], device_state_steps)

        if has_complex:
            _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0 and not decouple:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            weight_decay_unscaled = decay_factor * weight_decay
            device_grads = torch._foreach_add(  # type: ignore[assignment]
                device_grads,
                device_params,
                alpha=_get_value(weight_decay_unscaled),
            )

        if device_state_steps[0] == 0:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)
            if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:  # type: ignore[reportGeneralTypeIssues]
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
        torch._foreach_mul_(normed_grad, 1 - v1)
        torch._foreach_add_(normed_grad, device_exp_avgs, alpha=v1)
        torch._foreach_add_(device_params, normed_grad, alpha=-_get_value(lr))
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs,
            device_grads,
            device_grads,
            value=1 - beta2,
        )

        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:  # type: ignore[reportGeneralTypeIssues]
            torch._foreach_add_(
                device_state_steps,
                torch.tensor(1.0, device="cpu"),
                alpha=1.0,
            )
        else:
            torch._foreach_add_(device_state_steps, 1)


def _fused_qhadopt(*args: Any, **kwargs: Any) -> None:  # noqa: D401
    raise NotImplementedError


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_qhadopt)
def qhadopt(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
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
    v1: float,
    lr: float | Tensor,
    clip_lambda: Callable[[Number | Tensor | Any], float] | None,
    weight_decay: float,
    decouple: bool,
    eps: float,
    maximize: bool,
) -> None:
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():  # type: ignore[PGH003]
        msg = "torch.jit.script not supported with foreach optimizers"
        raise RuntimeError(msg)
    if fused and torch.jit.is_scripting():  # type: ignore[PGH003]
        msg = "torch.jit.script not supported with fused optimizers"
        raise RuntimeError(msg)

    if fused:
        func = _fused_qhadopt
    elif foreach:
        func = _multi_tensor_qhadopt
    else:
        func = _single_tensor_qhadopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        grad_scale,
        found_inf,
        initial_lr=initial_lr,
        decouple=decouple,
        clip_lambda=clip_lambda,
        beta1=beta1,
        beta2=beta2,
        v1=v1,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )
