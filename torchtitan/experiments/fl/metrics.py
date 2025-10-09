from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.distributed as dist
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer


class OptimizerMonitor:
    """Compute and log the L2 norm of gradients."""

    def __init__(
        self,
        interval: int = 10,
        *,
        only_global: bool = False,
        log_optimizer_metrics: bool = True,
        report_curvature: bool = False,
    ) -> None:
        """Initialise the optimizer monitor."""
        self.log_optimizer_metrics = log_optimizer_metrics
        self.only_global = only_global
        self.interval = interval
        self.report_curvature = report_curvature

    def batch_end(self, step: int, model: torch.nn.Module, optimizers: OptimizersContainer, logger: Any) -> None:
        """Calculate the statistics at the end of the batch."""
        if step % self.interval != 0:
            return

        optimizer_metrics: dict = {}
        optimizer = optimizers[0]

        for name, p in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                metric_reporter: Callable[[Any, Any, Any], dict] = getattr(
                    optimizer,
                    "report_per_parameter_metrics",
                    None,
                )
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics.update(
                        metric_reporter(p, name, optimizer_metrics),
                    )

                if f"l2_norm/grad/{name}" not in optimizer_metrics:
                    param_grad_norm = torch.linalg.vector_norm(p.grad)
                    optimizer_metrics[f"l2_norm/grad/{name}"] = param_grad_norm

        if dist.is_initialized() and dist.get_world_size() > 0 and self.log_optimizer_metrics:
            pre_reduce_metrics: Callable[[Any], dict] = getattr(
                optimizer,
                "pre_reduce_metrics",
                None,
            )
            if callable(pre_reduce_metrics) and self.log_optimizer_metrics:
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            dist_reduce_metrics: Callable[[Any], dict] = getattr(
                optimizer,
                "dist_reduce_metrics",
                None,
            )
            if callable(dist_reduce_metrics) and self.log_optimizer_metrics:
                optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        grad_norm, moment_norm, moment2_norm, update_norm, param_norm = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        min_moment2 = float("inf")
        max_moment2 = float("-inf")
        min_moment = float("inf")
        max_moment = float("-inf")
        min_param = float("inf")
        max_param = float("-inf")

        for metric in optimizer_metrics:
            if metric.startswith("l2_norm/grad"):
                grad_norm += optimizer_metrics[metric] ** 2
            if metric.startswith("l2_norm/moment"):
                moment_norm += optimizer_metrics[metric] ** 2
            if metric.startswith("l2_norm/moment2"):
                moment2_norm += optimizer_metrics[metric] ** 2
            if metric.startswith("min/moment2"):
                min_moment2 = min(min_moment2, optimizer_metrics[metric])
            if metric.startswith("max/moment2"):
                max_moment2 = max(max_moment2, optimizer_metrics[metric])
            if metric.startswith("min/moment"):
                min_moment = min(min_moment, optimizer_metrics[metric])
            if metric.startswith("max/moment"):
                max_moment = max(max_moment, optimizer_metrics[metric])
            if metric.startswith("min/param"):
                min_param = min(min_param, optimizer_metrics[metric])
            if metric.startswith("max/param"):
                max_param = max(max_param, optimizer_metrics[metric])
            if metric.startswith("l2_norm/update"):
                update_norm += optimizer_metrics[metric] ** 2
            if metric.startswith("l2_norm/param"):
                param_norm += optimizer_metrics[metric] ** 2

        if self.only_global:
            optimizer_metrics = {}

        optimizer_metrics["l2_norm/grad/global"] = grad_norm**0.5
        optimizer_metrics["l2_norm/moment/global"] = moment_norm**0.5
        optimizer_metrics["l2_norm/moment2/global"] = moment2_norm**0.5
        optimizer_metrics["l2_norm/update/global"] = update_norm**0.5
        optimizer_metrics["l2_norm/param/global"] = param_norm**0.5
        optimizer_metrics["min/moment2/global"] = min_moment2
        optimizer_metrics["max/moment2/global"] = max_moment2
        optimizer_metrics["min/moment/global"] = min_moment
        optimizer_metrics["max/moment/global"] = max_moment
        optimizer_metrics["min/param/global"] = min_param
        optimizer_metrics["max/param/global"] = max_param

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()
        logger.log(optimizer_metrics, step)


class FLMetricsProcessor(MetricsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_monitor = OptimizerMonitor()

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        super().log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)
        if self.model_parts and self.optimizers:
            self.optimizer_monitor.batch_end(
                step, self.model_parts[0], self.optimizers, self.logger
            )