# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Metrics for FL experiments."""

from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed.distributed_c10d as c10d

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.distributed import utils as dist_utils

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh

    from torchtitan.components.optimizer import OptimizersContainer


class AggregationType(Enum):
    """Types of metric aggregation."""

    L2_NORM = "l2_norm"
    MIN = "min"
    MAX = "max"


class OptimizerMonitor:
    """Compute and log the L2 norm of gradients.

    Args:
        interval: How often to log metrics (every N steps).
        only_global: Whether to only log global metrics.
        log_optimizer_metrics: Whether to log optimizer-specific metrics.
    """

    def __init__(
        self,
        interval: int = 10,
        *,
        only_global: bool = True,
        log_optimizer_metrics: bool = True,
    ) -> None:
        self.log_optimizer_metrics = log_optimizer_metrics
        self.only_global = only_global
        self.interval = interval

    def _reduce_metrics_across_ranks(
        self, optimizer_metrics: dict[str, torch.Tensor], mesh: DeviceMesh
    ) -> dict[str, torch.Tensor]:
        """Reduce optimizer metrics across all ranks using TorchTitan's distributed utilities.

        Follows the pattern from torchtitan.distributed.utils._dist_reduce.
        """
        reduced_metrics = {}

        for metric_name, metric_value in optimizer_metrics.items():
            if not isinstance(metric_value, torch.Tensor):
                # Skip non-tensor metrics
                reduced_metrics[metric_name] = metric_value
                continue

            # Determine reduction operation based on metric name
            if "l2_norm" in metric_name or "norm" in metric_name:
                # For L2 norms, the values are already squared by pre_reduce_metrics
                # Just sum them and take sqrt
                sum_squared = dist_utils.dist_sum(metric_value, mesh)
                # Convert back to tensor for sqrt
                reduced_metrics[metric_name] = torch.tensor(sum_squared).sqrt()
            elif "max" in metric_name:
                reduced_metrics[metric_name] = torch.tensor(
                    dist_utils.dist_max(metric_value, mesh)
                )
            elif "min" in metric_name:
                # dist_min not implemented, use -dist_max(-x)
                reduced_metrics[metric_name] = torch.tensor(
                    -dist_utils.dist_max(-metric_value, mesh)
                )
            elif "mean" in metric_name or "avg" in metric_name:
                reduced_metrics[metric_name] = torch.tensor(
                    dist_utils.dist_mean(metric_value, mesh)
                )
            else:
                # Default to sum for other metrics
                reduced_metrics[metric_name] = torch.tensor(
                    dist_utils.dist_sum(metric_value, mesh)
                )

        return reduced_metrics

    def batch_end(  # noqa: C901, PLR0912, PLR0915
        self,
        step: int,
        model: torch.nn.Module,
        optimizers: OptimizersContainer,
        logger: Any,
        mesh: DeviceMesh | None = None,
    ) -> None:
        """Calculate the statistics at the end of the batch."""
        if step % self.interval != 0:
            return

        optimizer_metrics: dict = {}
        optimizer = optimizers.optimizers[0]

        for name, p in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                metric_reporter: Callable[[Any, Any, Any], dict] | None = getattr(
                    optimizer,
                    "report_per_parameter_metrics",
                    None,
                )
                if callable(metric_reporter) and self.log_optimizer_metrics:
                    optimizer_metrics.update(
                        metric_reporter(p, name, optimizer_metrics),
                    )

        if (
            c10d.is_initialized()
            and c10d.get_world_size() > 1
            and self.log_optimizer_metrics
            and mesh is not None
        ):
            # Pre-process metrics before reduction
            pre_reduce_metrics: Callable[[Any]] | None = getattr(
                optimizer,
                "pre_reduce_metrics",
                None,
            )
            if callable(pre_reduce_metrics):
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            # Reduce metrics across all ranks using TorchTitan's distributed utilities
            optimizer_metrics = self._reduce_metrics_across_ranks(
                optimizer_metrics, mesh
            )

        # Dynamically aggregate all metric names found in optimizer_metrics
        agg_type_values = {agg_type.value for agg_type in AggregationType}
        agg_dict: dict[tuple[AggregationType, str], float] = {}
        metric_parts_required = 2

        # Initialize aggregation dictionary
        for metric in optimizer_metrics:
            parts = metric.split("/")
            if len(parts) < metric_parts_required:
                continue
            agg_type_str, metric_name = parts[0], parts[1]

            if agg_type_str not in agg_type_values:
                msg = f"Unknown aggregation type: {agg_type_str}"
                raise ValueError(msg)

            # Get the corresponding enum member
            agg_type = AggregationType(agg_type_str)
            key = (agg_type, metric_name)

            if key not in agg_dict:
                # Pattern match on aggregation type to get initial value
                match agg_type:
                    case AggregationType.L2_NORM:
                        agg_dict[key] = 0.0
                    case AggregationType.MIN:
                        agg_dict[key] = float("inf")
                    case AggregationType.MAX:
                        agg_dict[key] = float("-inf")

        # Aggregate metrics
        for metric in optimizer_metrics:
            parts = metric.split("/")
            if len(parts) < metric_parts_required:
                continue
            agg_type_str, metric_name = parts[0], parts[1]
            if agg_type_str not in agg_type_values:
                continue

            agg_type = AggregationType(agg_type_str)
            key = (agg_type, metric_name)
            value = optimizer_metrics[metric]

            # Pattern match on aggregation type to perform aggregation
            match agg_type:
                case AggregationType.L2_NORM:
                    agg_dict[key] += value**2
                case AggregationType.MIN:
                    agg_dict[key] = min(agg_dict[key], value)
                case AggregationType.MAX:
                    agg_dict[key] = max(agg_dict[key], value)

        # Report all aggregated metrics as agg_type/metric_name/global
        for (agg_type, metric_name), agg_value in agg_dict.items():
            # Pattern match on aggregation type to finalize value
            match agg_type:
                case AggregationType.L2_NORM:
                    final_value = agg_value**0.5
                case AggregationType.MIN | AggregationType.MAX:
                    final_value = agg_value

            optimizer_metrics[f"{agg_type.value}/{metric_name}/global"] = final_value

        # If only_global is set, remove all non-global metrics
        if self.only_global:
            optimizer_metrics = {
                k: v for k, v in optimizer_metrics.items() if k.endswith("/global")
            }

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()
        logger.log(optimizer_metrics, step)


class FLMetricsProcessor(MetricsProcessor):
    """Extension of MetricsProcessor to include optimizer monitoring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Get the optimizer monitor parameters from config with defaults
        interval = getattr(self.job_config, "optimizer_monitor_interval", 10)
        only_global = getattr(self.job_config, "optimizer_monitor_only_global", True)
        log_optimizer_metrics = getattr(
            self.job_config, "optimizer_monitor_log_metrics", True
        )

        self.optimizer_monitor = OptimizerMonitor(
            interval=interval,
            only_global=only_global,
            log_optimizer_metrics=log_optimizer_metrics,
        )

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log the metrics at the end of the step.

        Args:
            step: The current training step.
            global_avg_loss: The average loss across all workers.
            global_max_loss: The maximum loss across all workers.
            grad_norm: The gradient norm.
            extra_metrics: Any additional metrics to log.

        """
        super().log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics)

        if self.model_parts and self.optimizers:
            # Pass the dp_cp mesh for metric reduction across data parallel ranks
            mesh = self.parallel_dims.world_mesh["dp_cp"]
            self.optimizer_monitor.batch_end(
                step, self.model_parts[0], self.optimizers, self.logger, mesh
            )
