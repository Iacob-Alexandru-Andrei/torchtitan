# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Utility helpers to log optimizer statistics during TorchTitan training."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from torchtitan.components.optimizer import OptimizersContainer

__all__ = [
    "OptimizerMonitor",
    "get_report_curvature",
    "accumulate_curvature_metrics",
    "finalize_curvature_metrics",
]


def get_report_curvature() -> Callable[[torch.Tensor, str], dict[str, torch.Tensor]]:
    """Return a callable that computes curvature metrics per parameter."""

    prev_params: dict[str, torch.Tensor] = {}
    prev_grads: dict[str, torch.Tensor] = {}

    def report_curvature(param: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
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

        denom = torch.sum(torch.mul(grad_diff, param_diff)).item()
        if denom == 0.0:
            long_bb = torch.tensor(float("inf"), device=param.device, dtype=param.dtype)
        else:
            long_bb = param_diff_norm**2.0 / denom

        grad_diff_norm_val = grad_diff_norm.item()
        if grad_diff_norm_val == 0.0:
            short_bb = torch.tensor(float("inf"), device=param.device, dtype=param.dtype)
        else:
            short_bb = denom / (grad_diff_norm_val**2.0)

        with torch.no_grad():
            second_derivative_estimate = grad_diff / param_diff

        ratio = second_derivative_estimate / param.grad
        ratio_norm = torch.linalg.vector_norm(ratio)

        param_diff_norm_val = param_diff_norm.item()
        if param_diff_norm_val == 0.0:
            local_lipschitz = float("inf")
        else:
            local_lipschitz = grad_diff_norm_val / param_diff_norm_val

        prev_params[name] = param.detach().clone().cpu()
        prev_grads[name] = param.grad.detach().clone().cpu()

        return {
            f"curvature/param_diff_norm/{name}": param_diff_norm,
            f"curvature/grad_diff_norm/{name}": grad_diff_norm,
            f"curvature/long_bb/{name}": long_bb,
            f"curvature/short_bb/{name}": short_bb,
            f"curvature/l2_norm/second_to_first_derivative_estimate_ratio/{name}": ratio_norm,
            f"curvature/l2_norm/second_derivative_estimate/{name}": torch.linalg.vector_norm(
                second_derivative_estimate
            ),
            f"curvature/local_lipschitz/{name}": local_lipschitz,
        }

    return report_curvature


def accumulate_curvature_metrics(
    optimizer_metrics: dict[str, float | torch.Tensor],
    metric_name_prefixes: tuple[str, ...] = (
        "curvature/param_diff_norm",
        "curvature/grad_diff_norm",
        "curvature/long_bb",
        "curvature/short_bb",
        "curvature/l2_norm/second_to_first_derivative_estimate_ratio",
        "curvature/l2_norm/second_derivative_estimate",
        "curvature/local_lipschitz",
    ),
) -> dict[str, Any]:
    sums_for_norms = {
        "curvature/param_diff_norm": 0.0,
        "curvature/grad_diff_norm": 0.0,
    }
    list_for_stats: dict[str, list[float]] = {
        "curvature/long_bb": [],
        "curvature/short_bb": [],
        "curvature/l2_norm/second_to_first_derivative_estimate_ratio": [],
        "curvature/l2_norm/second_derivative_estimate": [],
        "curvature/local_lipschitz": [],
    }

    for metric_name, metric_value in optimizer_metrics.items():
        for prefix in metric_name_prefixes:
            if metric_name.startswith(prefix):
                val = (
                    metric_value.item()
                    if isinstance(metric_value, torch.Tensor)
                    else float(metric_value)
                )
                if prefix in sums_for_norms:
                    sums_for_norms[prefix] += val**2
                elif prefix in list_for_stats:
                    list_for_stats[prefix].append(val)
    return {
        "sums_for_norms": sums_for_norms,
        "list_for_stats": list_for_stats,
    }


def finalize_curvature_metrics(acc: dict[str, Any]) -> dict[str, float]:
    final_metrics: dict[str, float] = {}
    sums_for_norms = acc["sums_for_norms"]
    for prefix, squared_sum in sums_for_norms.items():
        final_metrics[f"{prefix}/global"] = math.sqrt(squared_sum)

    grad_diff_g = final_metrics.get("curvature/grad_diff_norm/global", 0.0)
    param_diff_g = final_metrics.get("curvature/param_diff_norm/global", 1.0)
    final_metrics["curvature/local_lipschitz/global"] = (
        grad_diff_g / param_diff_g if param_diff_g != 0.0 else float("inf")
    )

    list_for_stats = acc["list_for_stats"]
    for prefix, values in list_for_stats.items():
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        finite_tensor = tensor[torch.isfinite(tensor)]
        if finite_tensor.numel() == 0:
            continue
        final_metrics[f"{prefix}/mean"] = finite_tensor.mean().item()
        final_metrics[f"{prefix}/median"] = finite_tensor.median().item()
        final_metrics[f"{prefix}/min"] = finite_tensor.min().item()
        final_metrics[f"{prefix}/max"] = finite_tensor.max().item()
    return final_metrics


class OptimizerMonitor:
    """Compute optimizer-related metrics after each ``optimizer.step`` call."""

    def __init__(
        self,
        *,
        interval: int,
        only_global: bool = False,
        log_optimizer_metrics: bool = True,
        report_curvature: bool = False,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")
        self.interval = interval
        self.only_global = only_global
        self.log_optimizer_metrics = log_optimizer_metrics
        self.report_curvature = report_curvature
        self._curvature_reporter = get_report_curvature() if report_curvature else None

    def after_step(
        self,
        step: int,
        model_parts: list[torch.nn.Module],
        optimizers: OptimizersContainer,
    ) -> dict[str, float]:
        if (step + 1) % self.interval != 0:
            return {}

        metrics: dict[str, float | torch.Tensor] = {}
        global_accumulators = {
            "l2_norm/grad": 0.0,
            "l2_norm/param": 0.0,
            "l2_norm/moment": 0.0,
            "l2_norm/moment2": 0.0,
        }

        for part_idx, (model_part, optimizer) in enumerate(zip(model_parts, optimizers)):
            prefix = f"part{part_idx}"
            for name, param in model_part.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{prefix}.{name}"

                if self.log_optimizer_metrics and param.grad is not None:
                    grad_norm = torch.linalg.vector_norm(param.grad.detach()).item()
                    metrics[f"l2_norm/grad/{full_name}"] = grad_norm
                    global_accumulators["l2_norm/grad"] += grad_norm**2

                if self.log_optimizer_metrics:
                    param_norm = torch.linalg.vector_norm(param.detach()).item()
                    metrics[f"l2_norm/param/{full_name}"] = param_norm
                    global_accumulators["l2_norm/param"] += param_norm**2

                if self.log_optimizer_metrics:
                    state = optimizer.state.get(param, {})
                    exp_avg = state.get("exp_avg")
                    if exp_avg is not None:
                        moment_norm = torch.linalg.vector_norm(exp_avg.detach()).item()
                        metrics[f"l2_norm/moment/{full_name}"] = moment_norm
                        global_accumulators["l2_norm/moment"] += moment_norm**2
                    exp_avg_sq = state.get("exp_avg_sq")
                    if exp_avg_sq is not None:
                        moment2_norm = torch.linalg.vector_norm(
                            exp_avg_sq.detach()
                        ).item()
                        metrics[f"l2_norm/moment2/{full_name}"] = moment2_norm
                        global_accumulators["l2_norm/moment2"] += moment2_norm**2
                        metrics[f"min/moment2/{full_name}"] = exp_avg_sq.min().item()
                        metrics[f"max/moment2/{full_name}"] = exp_avg_sq.max().item()

                if self.report_curvature and self._curvature_reporter is not None:
                    curvature_metrics = self._curvature_reporter(param, full_name)
                    for key, value in curvature_metrics.items():
                        metrics[key] = (
                            value.item() if isinstance(value, torch.Tensor) else value
                        )

        if self.log_optimizer_metrics:
            for key, value in global_accumulators.items():
                metrics[f"{key}/global"] = math.sqrt(value)

        if self.report_curvature:
            curvature_acc = accumulate_curvature_metrics(metrics)
            metrics.update(finalize_curvature_metrics(curvature_acc))

        if self.only_global:
            metrics = {k: v for k, v in metrics.items() if k.endswith("/global")}

        # Ensure values are floats for downstream logging
        return {k: (v if isinstance(v, float) else float(v)) for k, v in metrics.items()}
