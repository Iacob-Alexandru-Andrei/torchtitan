# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Metrics for FL experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torchmetrics import Metric

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.fl.callbacks import (
    Callback,
    CallbackSetupContext,
    CallbackStepContext,
    CallbackValidationContext,
)
from torchtitan.experiments.fl.configs.config import (
    ActivationMonitorConfig,
    BetasMonitorConfig,
    HyperparameterSwitchConfig,
    LRMonitorConfig,
    OptimizerMonitorConfig,
    VSMonitorConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh


class AggregationType(Enum):
    """Types of metric aggregation."""

    L2_NORM = "l2_norm"
    MIN = "min"
    MAX = "max"


class PureUnigramCrossEntropy(Metric):
    """TorchMetric that computes unigram cross entropy for LM targets.

    This metric accumulates the per-token cross entropy between the provided
    targets and a pre-computed unigram distribution. Ignored indices are
    excluded from both the loss and item count.
    """

    full_state_update = False

    def __init__(
        self,
        unigram_probabilities: Tensor,
        ignore_index: int = -100,
        *,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if unigram_probabilities.dim() != 1:
            msg = "unigram_probabilities must be a 1D tensor."
            raise ValueError(msg)

        if torch.any(unigram_probabilities < 0):
            msg = "unigram_probabilities must contain non-negative values."
            raise ValueError(msg)

        if not torch.any(unigram_probabilities > 0):
            msg = "unigram_probabilities must include at least one positive value."
            raise ValueError(msg)

        prob_dtype = (
            unigram_probabilities.dtype
            if unigram_probabilities.is_floating_point()
            else torch.float32
        )
        self.ignore_index = ignore_index
        # Store as buffer so it moves with the metric across devices.
        self.register_buffer(
            "unigram_probabilities",
            unigram_probabilities.clone().detach().to(dtype=prob_dtype),
        )
        self.add_state(
            "sum_loss",
            default=torch.tensor(0.0, dtype=prob_dtype),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_items",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, output: Mapping | Tensor, target: Tensor) -> None:  # noqa: ARG002
        """Update the metric state with a batch of targets."""
        target = target.view(-1).to(torch.long)

        valid_mask = target != self.ignore_index
        if not torch.any(valid_mask):
            return

        target_filtered = target[valid_mask]
        vocab_size = self.unigram_probabilities.shape[0]
        in_vocab_mask = (target_filtered >= 0) & (target_filtered < vocab_size)
        if not torch.any(in_vocab_mask):
            return

        # Use the unigram probabilities corresponding to the valid targets.
        unigram_probs = self.unigram_probabilities
        if unigram_probs.device != target.device:
            unigram_probs = unigram_probs.to(target.device)

        target_in_vocab = target_filtered[in_vocab_mask]
        selected_probs = unigram_probs[target_in_vocab]
        eps = torch.finfo(selected_probs.dtype).tiny
        losses = -torch.log(selected_probs.clamp_min(eps))

        loss_sum = losses.sum().to(self.sum_loss.device)
        self.sum_loss += loss_sum

        items = int(in_vocab_mask.sum().item())
        self.total_items += items

    def compute(self) -> Tensor:
        """Return the average unigram cross entropy across all updates."""
        if int(self.total_items.item()) == 0:
            return self.sum_loss.new_zeros(())
        total_items = self.total_items.to(self.sum_loss.dtype)
        return self.sum_loss / total_items


_UNIGRAM_METRICS: list[PureUnigramCrossEntropy] = []


def add_unigram_metric(metric: PureUnigramCrossEntropy) -> None:
    """Track a unigram cross-entropy metric for logging."""
    _UNIGRAM_METRICS.append(metric)


def collect_unigram_metrics(*, reset: bool = True) -> tuple[float, int]:
    """Return the local summed loss and token count for unigram metrics."""
    total_loss = 0.0
    total_items = 0

    for metric in _UNIGRAM_METRICS:
        items = int(metric.total_items.item())
        if items == 0:
            continue
        loss_value = float(metric.sum_loss.item())
        total_loss += loss_value
        total_items += items
        if reset:
            metric.sum_loss.zero_()
            metric.total_items.zero_()

    return total_loss, total_items


def reset_unigram_metrics() -> None:
    for metric in _UNIGRAM_METRICS:
        metric.sum_loss.zero_()
        metric.total_items.zero_()


def update_unigram_metrics(labels: Tensor) -> None:
    if not _UNIGRAM_METRICS:
        return
    for metric in _UNIGRAM_METRICS:
        metric.update({}, labels)


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


class ActivationMonitor(Callback):
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
        self._model_ref: torch.nn.Module | None = None

    def setup(self, context: CallbackSetupContext) -> None:
        if not self.enabled or self._registered:
            return
        if not context.model_parts:
            return
        model = context.model_parts[0]
        self.register(model)
        self._model_ref = model

    def on_step_end(self, context: CallbackStepContext) -> None:
        if not self.enabled or not self._registered:
            return
        self.finalize(context.step, context.logger, context.mesh)

    def close(self) -> None:
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._module_names.clear()
        self._metrics.clear()
        self._collect_this_step = False
        self._registered = False
        self._model_ref = None

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


class OptimizerMonitor(Callback):
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
        self._model_ref: torch.nn.Module | None = None

    def setup(self, context: CallbackSetupContext) -> None:
        if context.model_parts:
            self._model_ref = context.model_parts[0]

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

    def on_step_end(  # noqa: C901, PLR0912, PLR0915
        self,
        context: CallbackStepContext,
    ) -> None:
        """Calculate the statistics at the end of the batch."""
        optimizers = context.optimizers
        if optimizers is None:
            return
        if optimizers.optimizers is None:
            return
        if len(optimizers.optimizers) == 0:
            return

        model = context.model_parts[0] if context.model_parts else self._model_ref
        if model is None:
            return

        step = context.step
        mesh = context.mesh
        logger = context.logger

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


class LRMonitor(Callback):
    """Logs the learning rate of each optimizer parameter group."""

    def __init__(self, *, interval: int = 1, enabled: bool = True) -> None:
        self.interval = interval
        self.enabled = enabled

    def on_step_end(self, context: CallbackStepContext) -> None:
        if not self.enabled or self.interval <= 0:
            return
        if context.optimizers is None:
            return
        if context.step % self.interval != 0:
            return

        metrics: dict[str, float] = {}
        for optimizer in context.optimizers:
            name = optimizer.__class__.__name__
            for idx, group in enumerate(optimizer.param_groups):
                lr = group.get("lr")
                if lr is None:
                    continue
                metrics[f"lr-{name}/group{idx}"] = float(lr)

        if metrics:
            context.logger.log(metrics, context.step)


class BetasMonitor(Callback):
    """Logs optimizer beta hyperparameters."""

    def __init__(self, *, interval: int = 0, enabled: bool = False) -> None:
        self.interval = interval
        self.enabled = enabled

    def on_step_end(self, context: CallbackStepContext) -> None:
        if not self.enabled or self.interval <= 0:
            return
        if context.optimizers is None:
            return
        if context.step % self.interval != 0:
            return

        metrics: dict[str, float] = {}
        for optimizer in context.optimizers:
            name = optimizer.__class__.__name__
            for idx, group in enumerate(optimizer.param_groups):
                betas = group.get("betas")
                if betas is None:
                    continue
                if isinstance(betas, Sequence) and not isinstance(betas, (str, bytes)):
                    beta_values = list(betas)
                else:
                    beta_values = [betas]
                for beta_idx, beta_value in enumerate(beta_values, start=1):
                    if isinstance(beta_value, torch.Tensor):
                        beta_scalar = float(beta_value.detach().item())
                    else:
                        beta_scalar = float(beta_value)
                    metrics[f"beta{beta_idx}-{name}/group{idx}"] = beta_scalar

        if metrics:
            context.logger.log(metrics, context.step)


class VSMonitor(Callback):
    """Logs quasi-hyperbolic v parameters if available."""

    def __init__(self, *, interval: int = 0, enabled: bool = False) -> None:
        self.interval = interval
        self.enabled = enabled

    def on_step_end(self, context: CallbackStepContext) -> None:
        if not self.enabled or self.interval <= 0:
            return
        if context.optimizers is None:
            return
        if context.step % self.interval != 0:
            return

        metrics: dict[str, float] = {}
        for optimizer in context.optimizers:
            name = optimizer.__class__.__name__
            for idx, group in enumerate(optimizer.param_groups):
                vs = group.get("vs")
                if vs is None:
                    continue
                if isinstance(vs, Sequence) and not isinstance(vs, (str, bytes)):
                    v_values = list(vs)
                else:
                    v_values = [vs]
                for v_idx, v_value in enumerate(v_values):
                    if isinstance(v_value, torch.Tensor):
                        v_scalar = float(v_value.detach().item())
                    else:
                        v_scalar = float(v_value)
                    metrics[f"v{v_idx}-{name}/group{idx}"] = v_scalar

        if metrics:
            context.logger.log(metrics, context.step)


class HyperparameterSwitchCallback(Callback):
    """Switch optimizer betas/vs at configured steps and optionally reset momenta."""

    def __init__(
        self,
        *,
        enabled: bool,
        steps: Sequence[int],
        new_vs: Sequence[float] | None,
        new_betas: Sequence[float] | None,
        reset_momenta: Sequence[str],
        log_metrics: bool,
    ) -> None:
        self.enabled = enabled and bool(steps)
        self.steps = {int(step) for step in steps if step >= 0}
        self.new_vs = tuple(float(v) for v in new_vs) if new_vs is not None else None
        self.new_betas = (
            tuple(float(b) for b in new_betas) if new_betas is not None else None
        )
        self.reset_momenta = tuple(reset_momenta)
        self.log_metrics = log_metrics
        self._applied_steps: set[int] = set()

    def on_step_end(self, context: CallbackStepContext) -> None:
        if not self.enabled:
            return
        step = context.step
        if step not in self.steps or step in self._applied_steps:
            return
        optimizers = context.optimizers
        if optimizers is None:
            return

        for optimizer in optimizers:
            if self.new_vs is not None:
                self._update_group_values(optimizer.param_groups, "vs", self.new_vs)
            if self.new_betas is not None:
                self._update_group_values(
                    optimizer.param_groups, "betas", self.new_betas
                )
            if self.reset_momenta:
                self._reset_momenta(optimizer.state)

        if self.log_metrics:
            payload: dict[str, float] = {}
            if self.new_vs is not None:
                for idx, value in enumerate(self.new_vs):
                    payload[f"hyper_switch/v{idx}"] = value
            if self.new_betas is not None:
                for idx, value in enumerate(self.new_betas, start=1):
                    payload[f"hyper_switch/beta{idx}"] = value
            if payload:
                context.logger.log(payload, step)

        self._applied_steps.add(step)

    def _update_group_values(
        self, param_groups: list[dict[str, Any]], key: str, values: tuple[float, ...]
    ) -> None:
        for group in param_groups:
            if key not in group:
                continue
            current_value = group[key]
            if isinstance(current_value, torch.Tensor):
                target = torch.tensor(
                    values, device=current_value.device, dtype=current_value.dtype
                )
                if current_value.shape == target.shape:
                    current_value.copy_(target)
                else:
                    group[key] = target
            elif isinstance(current_value, Sequence) and not isinstance(
                current_value, (str, bytes)
            ):
                group[key] = tuple(values)
            elif isinstance(current_value, (float, int)):
                group[key] = float(values[0])
            else:
                group[key] = tuple(values)

    def _reset_momenta(self, optimizer_state: dict[Any, dict[str, Any]]) -> None:
        for state in optimizer_state.values():
            for name in self.reset_momenta:
                if name not in state:
                    continue
                self._zero_state_value(state[name])

    def _zero_state_value(self, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value.zero_()
        elif isinstance(value, dict):
            for inner in value.values():
                self._zero_state_value(inner)
        elif isinstance(value, (list, tuple)):
            for inner in value:
                self._zero_state_value(inner)


class FLMetricsProcessor(MetricsProcessor):
    """Extension of MetricsProcessor that wires the FL callback stack."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Get metrics config from fl_metrics field
        fl_metrics_config = self.job_config.fl_metrics.unwrap()  # type: ignore[attr-defined]

        optimizer_config = fl_metrics_config.optimizer_monitor
        activation_config = fl_metrics_config.activation_monitor
        activation_enabled = activation_config.enabled or (
            activation_config.interval > 0
        )
        activation_interval = activation_config.interval
        ignore_types = activation_config.ignore_module_types
        lr_config = fl_metrics_config.lr_monitor
        betas_config = fl_metrics_config.betas_monitor
        vs_config = fl_metrics_config.vs_monitor
        hyper_switch_config = fl_metrics_config.hyperparameter_switch

        self.callbacks: list[Callback] = []

        if optimizer_config.interval > 0:
            self.optimizer_monitor: OptimizerMonitor | None = OptimizerMonitor(
                interval=optimizer_config.interval,
                only_global=optimizer_config.only_global,
                log_optimizer_metrics=optimizer_config.log_metrics,
            )
            self.callbacks.append(self.optimizer_monitor)
        else:
            self.optimizer_monitor = None

        if activation_enabled:
            self.activation_monitor: ActivationMonitor | None = ActivationMonitor(
                interval=activation_interval,
                ignore_module_types=ignore_types if ignore_types else (),
                gradient_accumulation_steps=activation_config.gradient_accumulation_steps,
                enabled_metrics=activation_config.enabled_metrics,
            )
            self.callbacks.append(self.activation_monitor)
        else:
            self.activation_monitor = None

        if lr_config.enabled and lr_config.interval > 0:
            self.lr_monitor: LRMonitor | None = LRMonitor(
                interval=lr_config.interval,
                enabled=lr_config.enabled,
            )
            self.callbacks.append(self.lr_monitor)
        else:
            self.lr_monitor = None

        if betas_config.enabled and betas_config.interval > 0:
            self.betas_monitor: BetasMonitor | None = BetasMonitor(
                interval=betas_config.interval,
                enabled=betas_config.enabled,
            )
            self.callbacks.append(self.betas_monitor)
        else:
            self.betas_monitor = None

        if vs_config.enabled and vs_config.interval > 0:
            self.vs_monitor: VSMonitor | None = VSMonitor(
                interval=vs_config.interval,
                enabled=vs_config.enabled,
            )
            self.callbacks.append(self.vs_monitor)
        else:
            self.vs_monitor = None

        if hyper_switch_config.enabled and hyper_switch_config.steps:
            steps = tuple(int(step) for step in hyper_switch_config.steps)
            new_vs = (
                tuple(hyper_switch_config.new_vs)
                if hyper_switch_config.new_vs is not None
                else None
            )
            new_betas = (
                tuple(hyper_switch_config.new_betas)
                if hyper_switch_config.new_betas is not None
                else None
            )
            reset_momenta = tuple(hyper_switch_config.reset_momenta)
            self.hyperparameter_switch: HyperparameterSwitchCallback | None = (
                HyperparameterSwitchCallback(
                    enabled=hyper_switch_config.enabled,
                    steps=steps,
                    new_vs=new_vs,
                    new_betas=new_betas,
                    reset_momenta=reset_momenta,
                    log_metrics=hyper_switch_config.log_metrics,
                )
            )
            self.callbacks.append(self.hyperparameter_switch)
        else:
            self.hyperparameter_switch = None

        self._callbacks_setup_done = False

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

    def _build_unigram_payload(self, mesh: DeviceMesh | None) -> dict[str, float]:
        local_loss, local_items = collect_unigram_metrics(reset=False)
        if local_items == 0:
            reset_unigram_metrics()
            return {"pure_unigram_cross_entropy": 0.0}

        device = torch.device("cpu")
        if self.model_parts:
            try:
                device = next(self.model_parts[0].parameters()).device
            except StopIteration:
                pass

        loss_tensor = torch.tensor(
            float(local_loss), device=device, dtype=torch.float64
        )
        items_tensor = torch.tensor(
            float(local_items), device=device, dtype=torch.float64
        )

        if mesh is not None:
            loss_tensor = dist_utils.dist_sum(loss_tensor, mesh)
            items_tensor = dist_utils.dist_sum(items_tensor, mesh)

        global_loss = float(loss_tensor)
        global_items = float(items_tensor)
        reset_unigram_metrics()

        if global_items <= 0:
            return {"pure_unigram_cross_entropy": 0.0}

        return {
            "pure_unigram_cross_entropy": global_loss / global_items,
            "pure_unigram_cross_entropy/token_count": global_items,
        }

    def _ensure_callbacks_setup(self) -> None:
        if self._callbacks_setup_done:
            return
        if not self.callbacks:
            self._callbacks_setup_done = True
            return
        if not self.model_parts or self.optimizers is None:
            return

        setup_context = CallbackSetupContext(
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            logger=self.logger,
            parallel_dims=self.parallel_dims,
            job_config=self.job_config,
        )
        for callback in self.callbacks:
            callback.setup(setup_context)
        self._callbacks_setup_done = True

    def _run_step_callbacks(self, step: int, mesh: DeviceMesh | None) -> None:
        if not self.callbacks:
            return
        if not self.model_parts or self.optimizers is None:
            return

        context = CallbackStepContext(
            step=step,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            logger=self.logger,
            mesh=mesh,
        )
        for callback in self.callbacks:
            callback.on_step_end(context)

    def _run_validation_callbacks(self, loss: float, step: int) -> None:
        if not self.callbacks:
            return
        context = CallbackValidationContext(
            step=step,
            loss=loss,
            logger=self.logger,
        )
        for callback in self.callbacks:
            callback.on_validation_end(context)

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
        mesh = (
            self.parallel_dims.world_mesh["dp_cp"]
            if self.parallel_dims.dp_cp_enabled
            else None
        )
        unigram_payload = self._build_unigram_payload(mesh)
        combined_metrics = dict(extra_metrics) if extra_metrics else {}
        if unigram_payload:
            combined_metrics.update(unigram_payload)

        super().log(
            step,
            global_avg_loss,
            global_max_loss,
            grad_norm,
            extra_metrics=combined_metrics or None,
        )

        self._ensure_callbacks_setup()
        self._run_step_callbacks(step, mesh)

    def log_validation(self, loss: float, step: int) -> None:
        super().log_validation(loss, step)
        mesh = (
            self.parallel_dims.world_mesh["dp_cp"]
            if self.parallel_dims.dp_cp_enabled
            else None
        )
        unigram_payload = self._build_unigram_payload(mesh)
        if unigram_payload:
            self.logger.log(unigram_payload, step)
        self._ensure_callbacks_setup()
        self._run_validation_callbacks(loss, step)

    def close(self) -> None:
        for callback in self.callbacks:
            callback.close()
        super().close()
