# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Metrics for FL experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any, TYPE_CHECKING

import torch

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


def compute_skewness(value: torch.Tensor) -> torch.Tensor:
    """Compute the skewness of a tensor.

    Args:
        value: Input tensor of shape (..., N).

    Returns:
        Tensor containing the skewness value.
    """
    mean = value.mean(dim=-1, keepdim=True)
    diffs = value - mean
    m_3 = torch.mean(torch.pow(diffs, 3), dim=-1)
    var = torch.mean(torch.pow(diffs, 2), dim=-1)
    eps = torch.finfo(var.dtype).eps if var.dtype.is_floating_point else 1e-12
    var = torch.clamp(var, min=eps)
    return (m_3 / (var * torch.sqrt(var))).mean()


def compute_kurtosis(value: torch.Tensor) -> torch.Tensor:
    """Compute the kurtosis of a tensor.

    Args:
        value: Input tensor of shape (..., N).

    Returns:
        Tensor containing the kurtosis value.
    """
    mean = value.mean(dim=-1, keepdim=True)
    diffs = value - mean
    m_4 = torch.mean(torch.pow(diffs, 4), dim=-1)
    var = torch.mean(torch.pow(diffs, 2), dim=-1)
    eps = torch.finfo(var.dtype).eps if var.dtype.is_floating_point else 1e-12
    var = torch.clamp(var, min=eps)
    return (m_4 / (var**2)).mean()


class ActivationMonitor:
    """Collects activation statistics across the full model.

    By default, only the following metrics are collected:
    - activations/average/max/full_model_input
    - activations/average/max/full_model_output
    - activations/average/median/full_model_input
    - activations/average/median/full_model_output
    - activations/l2_norm/full_model_input
    - activations/l2_norm/full_model_output
    - activations/max/full_model_input
    - activations/max/full_model_output
    """

    def __init__(
        self,
        *,
        interval: int = 25,
        ignore_module_types: Sequence[str] | None = None,
        gradient_accumulation_steps: int = 1,
        enabled_metrics: set[str] | None = None,
    ) -> None:
        self.interval = interval
        self.ignore_module_types = (
            tuple(ignore_module_types) if ignore_module_types is not None else None
        )
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        # Default enabled metrics - only the essential ones
        if enabled_metrics is None:
            self.enabled_metrics = {
                "activations/average/max/full_model_input",
                "activations/average/max/full_model_output",
                "activations/average/median/full_model_input",
                "activations/average/median/full_model_output",
                "activations/l2_norm/full_model_input",
                "activations/l2_norm/full_model_output",
                "activations/max/full_model_input",
                "activations/max/full_model_output",
            }
        else:
            self.enabled_metrics = enabled_metrics

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pre_handle: torch.utils.hooks.RemovableHandle | None = None
        self._module_names: dict[torch.nn.Module, str] = {}
        self._metrics: dict[str, float | list[float]] = {}
        self._collect_this_step = False
        self._microbatch_counter = 0
        self._device: torch.device | None = None
        self._registered = False

    def _is_metric_enabled(self, metric_key: str) -> bool:
        """Check if a metric is enabled for collection."""
        return metric_key in self.enabled_metrics

    @property
    def enabled(self) -> bool:
        """Check if the monitor is enabled based on the interval.

        Returns:
            bool: True if the monitor is enabled, False otherwise.
        """
        return self.interval > 0

    def should_log_step(self, step: int) -> bool:
        """Determine if metrics should be logged at the current step.

        Args:
            step: Current training step.

        Returns:
            bool: True if metrics should be logged this step, False otherwise.
        """
        return self.enabled and step % self.interval == 0

    @property
    def is_registered(self) -> bool:
        """Check if hooks are registered.

        Returns:
            bool: True if hooks are registered, False otherwise.
        """
        return self._registered

    def register(self, model: torch.nn.Module) -> None:
        """Register forward hooks on the model to collect activations.

        Args:
            model: The model to register hooks on.
        """
        if not self.enabled or self._registered:
            return

        self._module_names = {module: name for name, module in model.named_modules()}
        self._pre_handle = model.register_forward_pre_hook(
            self._forward_pre_hook, with_kwargs=True
        )
        model.apply(self._register_forward_hook)
        self._registered = True

    def _register_forward_hook(self, module: torch.nn.Module) -> None:
        self._handles.append(module.register_forward_hook(self._forward_hook))

    def _forward_pre_hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module, args, kwargs
        # Pre-hook doesn't need to do anything; collection is controlled by finalize()

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not self._collect_this_step:
            return

        module_name = self._module_names.get(module, "")
        if self.ignore_module_types is not None:
            lowered_name = module_name.lower()
            if any(
                ignore.lower() in lowered_name for ignore in self.ignore_module_types
            ):
                return

        self._recursively_add_metrics("_input", inputs)
        self._recursively_add_metrics("_output", output)

    def _recursively_add_metrics(self, suffix: str, values: Any) -> None:
        if values is None:
            return
        if isinstance(values, (str, bytes)):
            return
        if isinstance(values, dict):
            for val in values.values():
                self._recursively_add_metrics(suffix, val)
            return
        if isinstance(values, torch.Tensor):
            self._add_metrics(suffix, values)
            return
        if isinstance(values, Sequence):
            for value in values:
                self._recursively_add_metrics(suffix, value)

    def _add_metrics(  # noqa: C901, PLR0912, PLR0915
        self, suffix: str, value: torch.Tensor
    ) -> None:
        if value.dtype == torch.bool:
            return
        if not (value.is_floating_point() or value.is_complex()):
            return

        with torch.no_grad():
            tensor = value.detach()
            if tensor.is_complex():
                tensor = tensor.real
            if self._device is None:
                self._device = tensor.device

            # Accumulate sum of squares for L2 norm (will sqrt after gathering)
            l2_key = f"activations/l2_norm/full_model{suffix}"
            if self._is_metric_enabled(l2_key):
                current_l2 = self._metrics.get(l2_key, 0.0)
                if isinstance(current_l2, float):
                    self._metrics[l2_key] = current_l2 + float(
                        torch.sum(tensor**2).item()
                    )

            avg_key = f"activations/average/full_model{suffix}"
            # Check if any average metrics are enabled (max, min, or median)
            avg_max_key = f"activations/average/max/full_model{suffix}"
            avg_min_key = f"activations/average/min/full_model{suffix}"
            avg_median_key = f"activations/average/median/full_model{suffix}"
            if (
                self._is_metric_enabled(avg_max_key)
                or self._is_metric_enabled(avg_min_key)
                or self._is_metric_enabled(avg_median_key)
            ):
                avg_list = self._metrics.setdefault(avg_key, [])
                if isinstance(avg_list, list):
                    avg_list.append(float(tensor.mean().item()))

            if tensor.numel() == 0:
                return

            # Compute max over last dimension and take mean (consistent with reference)
            max_key = f"activations/max/full_model{suffix}"
            if self._is_metric_enabled(max_key):
                if tensor.ndim >= 1 and tensor.shape[-1] > 0:
                    max_value = tensor.max(dim=-1).values.mean().item()
                else:
                    max_value = tensor.max().item()
                max_list = self._metrics.setdefault(max_key, [])
                if isinstance(max_list, list):
                    max_list.append(float(max_value))

            # Check if skewness or kurtosis metrics are enabled
            if tensor.ndim >= 1 and tensor.shape[-1] > 0:
                skew_max_key = f"activations/skewness/max/full_model{suffix}"
                skew_min_key = f"activations/skewness/min/full_model{suffix}"
                skew_median_key = f"activations/skewness/median/full_model{suffix}"
                kurt_max_key = f"activations/kurtosis/max/full_model{suffix}"
                kurt_min_key = f"activations/kurtosis/min/full_model{suffix}"
                kurt_median_key = f"activations/kurtosis/median/full_model{suffix}"

                need_skewness = (
                    self._is_metric_enabled(skew_max_key)
                    or self._is_metric_enabled(skew_min_key)
                    or self._is_metric_enabled(skew_median_key)
                )
                need_kurtosis = (
                    self._is_metric_enabled(kurt_max_key)
                    or self._is_metric_enabled(kurt_min_key)
                    or self._is_metric_enabled(kurt_median_key)
                )

                if need_skewness or need_kurtosis:
                    skew_key = f"activations/skewness/full_model{suffix}"
                    kurt_key = f"activations/kurtosis/full_model{suffix}"

                    if need_skewness:
                        skewness = compute_skewness(tensor)
                        skew_list = self._metrics.setdefault(skew_key, [])
                        if isinstance(skew_list, list):
                            skew_list.append(float(skewness.item()))

                    if need_kurtosis:
                        kurtosis = compute_kurtosis(tensor)
                        kurt_list = self._metrics.setdefault(kurt_key, [])
                        if isinstance(kurt_list, list):
                            kurt_list.append(float(kurtosis.item()))

    def finalize(
        self,
        step: int,
        logger: Any,
        mesh: DeviceMesh | None,
    ) -> None:
        """Finalize metric collection for the current step.

        Args:
            step: Current training step.
            logger: Logger to log metrics.
            mesh: Device mesh for distributed reduction.
        """
        if not self.enabled or not self._registered:
            return

        # If this IS a logging step, log the metrics we collected during this step
        if self.should_log_step(step) and self._metrics:
            metrics = self._prepare_local_metrics()
            if metrics:
                reduced_metrics = self._reduce_metrics(metrics, mesh)
                if reduced_metrics:
                    logger.log(reduced_metrics, step)

        # Prepare for next step: enable collection if next step should be logged
        next_step = step + 1
        if self.should_log_step(next_step):
            self._reset_metrics()
            self._collect_this_step = True
        else:
            self._collect_this_step = False
            self._reset_metrics()

    def _prepare_local_metrics(self) -> dict[str, float | list[float]]:  # noqa: C901
        prepared: dict[str, float | list[float]] = {}
        for suffix in ("_input", "_output"):
            l2_key = f"activations/l2_norm/full_model{suffix}"
            if l2_key in self._metrics and self._is_metric_enabled(l2_key):
                l2_val = self._metrics[l2_key]
                if isinstance(l2_val, float):
                    prepared[l2_key] = l2_val

            max_key = f"activations/max/full_model{suffix}"
            if self._is_metric_enabled(max_key):
                max_vals = self._metrics.get(max_key)
                if max_vals and isinstance(max_vals, list):
                    prepared[max_key] = float(max(max_vals))

            for metric_name in ("average", "skewness", "kurtosis"):
                key = f"activations/{metric_name}/full_model{suffix}"
                values = self._metrics.get(key)
                if not values or not isinstance(values, list):
                    continue
                tensor_values = torch.tensor(values)

                max_metric_key = f"activations/{metric_name}/max/full_model{suffix}"
                if self._is_metric_enabled(max_metric_key):
                    prepared[max_metric_key] = float(tensor_values.max().item())

                min_metric_key = f"activations/{metric_name}/min/full_model{suffix}"
                if self._is_metric_enabled(min_metric_key):
                    prepared[min_metric_key] = float(tensor_values.min().item())

                median_metric_key = (
                    f"activations/{metric_name}/median/full_model{suffix}"
                )
                if self._is_metric_enabled(median_metric_key):
                    prepared[median_metric_key] = values

        return prepared

    def _reduce_metrics(  # noqa: C901, PLR0912
        self, metrics: dict[str, float | list[float]], mesh: DeviceMesh | None
    ) -> dict[str, float]:
        reduced: dict[str, float] = {}

        # Handle single-rank or no mesh case
        if mesh is None:
            for key, value in metrics.items():
                if "l2_norm" in key:
                    if isinstance(value, list):
                        if not value:
                            continue
                        # Compute sqrt of sum of squares for L2 norm
                        reduced[key] = math.sqrt(sum(x**2 for x in value))
                    else:
                        reduced[key] = math.sqrt(value)
                elif isinstance(value, list):
                    if not value:
                        continue
                    # Use statistics.median for efficiency (no tensor conversion)
                    sorted_values = sorted(value)
                    n = len(sorted_values)
                    if n % 2 == 0:
                        reduced[key] = (
                            sorted_values[n // 2 - 1] + sorted_values[n // 2]
                        ) / 2
                    else:
                        reduced[key] = sorted_values[n // 2]
                else:
                    reduced[key] = value
            return reduced

        device = self._device or torch.device("cpu")

        for key, value in metrics.items():
            if isinstance(value, list):
                if not value:
                    continue
                # Convert to tensor and compute median locally
                local_median = torch.tensor(value, device=device).median()
                # Take max of medians across ranks (for consistency with reference)
                reduced[key] = dist_utils.dist_max(local_median, mesh)
                continue

            # Create tensor once and reuse
            tensor_value = torch.tensor(value, device=device)
            if "l2_norm" in key:
                # For L2 norm: sum across ranks then sqrt
                # Note: value is already squared locally
                reduced[key] = math.sqrt(dist_utils.dist_sum(tensor_value, mesh))
            elif "max" in key:
                reduced[key] = dist_utils.dist_max(tensor_value, mesh)
            elif "min" in key:
                reduced[key] = -dist_utils.dist_max(-tensor_value, mesh)
            else:
                # Default to mean for other metrics
                reduced[key] = dist_utils.dist_mean(tensor_value, mesh)

        return reduced

    def _reset_metrics(self) -> None:
        self._metrics = {}


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
    ) -> dict[str, float]:
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
                # Sum across ranks then sqrt (dist_utils returns float)
                sum_squared = dist_utils.dist_sum(metric_value, mesh)
                reduced_metrics[metric_name] = math.sqrt(sum_squared)
            elif "max" in metric_name:
                reduced_metrics[metric_name] = dist_utils.dist_max(metric_value, mesh)
            elif "min" in metric_name:
                # dist_min not implemented, use -dist_max(-x)
                reduced_metrics[metric_name] = -dist_utils.dist_max(-metric_value, mesh)
            elif "mean" in metric_name or "avg" in metric_name:
                reduced_metrics[metric_name] = dist_utils.dist_mean(metric_value, mesh)
            else:
                # Default to sum for other metrics
                reduced_metrics[metric_name] = dist_utils.dist_sum(metric_value, mesh)

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
        # Early exit if monitoring is disabled (interval <= 0)
        if self.interval <= 0:
            return

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

        if mesh is not None and self.log_optimizer_metrics:
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

        # Convert any remaining tensors to floats (shouldn't be any after reduction, but just in case)
        optimizer_metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer_metrics.items()
        }
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

        activation_enabled = getattr(
            self.job_config, "activation_monitor_enabled", False
        )
        if activation_enabled:
            activation_interval = getattr(
                self.job_config, "activation_monitor_interval", 25
            )
            ignore_types = getattr(
                self.job_config,
                "activation_monitor_ignore_module_types",
                (),
            )
            if ignore_types is None:
                ignore_types = ()
            grad_accum_steps = getattr(
                getattr(self.job_config, "training", None),
                "gradient_accumulation_steps",
                1,
            )

            # Get enabled metrics from config, or use None for defaults
            enabled_metrics_list = getattr(
                self.job_config,
                "activation_monitor_enabled_metrics",
                None,
            )
            enabled_metrics = (
                set(enabled_metrics_list) if enabled_metrics_list is not None else None
            )

            self.activation_monitor: ActivationMonitor | None = ActivationMonitor(
                interval=activation_interval,
                ignore_module_types=ignore_types,
                gradient_accumulation_steps=grad_accum_steps,
                enabled_metrics=enabled_metrics,
            )
        else:
            self.activation_monitor = None

    def should_log(self, step: int) -> bool:
        """Determine if metrics should be logged at the current step.

        Args:
            step (int): Current training step.

        Returns:
            bool: True if metrics should be logged this step, False otherwise.
        """
        base_decision = super().should_log(step)
        if self.activation_monitor and self.activation_monitor.should_log_step(step):
            return True
        return base_decision

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

        mesh = (
            self.parallel_dims.world_mesh["dp_cp"]
            if self.parallel_dims.dp_cp_enabled
            else None
        )

        if self.model_parts and self.optimizers:
            self.optimizer_monitor.batch_end(
                step, self.model_parts[0], self.optimizers, self.logger, mesh
            )

        if self.activation_monitor and self.model_parts:
            if not self.activation_monitor.is_registered:
                self.activation_monitor.register(self.model_parts[0])
            self.activation_monitor.finalize(step, self.logger, mesh)
