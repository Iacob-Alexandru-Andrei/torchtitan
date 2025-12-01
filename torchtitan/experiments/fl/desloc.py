# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DES-LOC integration utilities for the FL experiments."""

from __future__ import annotations

import logging
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import ModuleType
from typing import Any, TYPE_CHECKING, Literal, Sequence
from collections.abc import Callable, Iterable
from fnmatch import fnmatch

import torch
from torch import nn
from torch.distributed.distributed_c10d import Work
from torch.utils.hooks import RemovableHandle
from torch.optim import Optimizer

try:  # pragma: no cover - optional dependency in some environments
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - DTensor is optional
    DTensor = None  # type: ignore[assignment]

from torchtitan.components.optimizer import FTOptimizersContainer

_MODULE_PROXY = sys.modules.get(__name__)
if _MODULE_PROXY is None:
    _MODULE_PROXY = ModuleType(__name__)
    sys.modules[__name__] = _MODULE_PROXY
_MODULE_PROXY.__dict__.update(globals())

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from torchtitan.components.ft.manager import FTManager
from torchtitan.experiments.fl.configs.optimizers import (
    DesLocConfig,
    DesLocOuterOptimizerConfig,
    DesLocStreamingConfig,
)

logger = logging.getLogger(__name__)
USE_BUCKETIZATION_ENV = "TORCHFT_USE_BUCKETIZATION"


@dataclass(frozen=True)
class ParameterFragmentConfig:
    """Configuration for synchronizing model parameters via DES-LOC."""

    manager: Any
    model: nn.Module
    param_entries: list[tuple[str, nn.Parameter]] | None
    sync_every: int
    backup_device: torch.device | None
    pin_memory: bool
    name_prefix: str
    outer_optimizer: DesLocOuterOptimizerConfig | Optimizer | list[Optimizer] | None = None
    log_outer_metrics: bool = False
    metrics_logger: Callable[[dict[str, float]], None] | None = None
    checkpoint_outer_optimizer: bool = True


@dataclass(frozen=True)
class OptimizerFragmentConfig:
    """Configuration for synchronizing optimizer state tensors."""

    manager: Any
    model: nn.Module
    param_entries: list[tuple[str, nn.Parameter]] | None
    optimizer: Optimizer
    state_key: str
    sync_every: int
    backup_device: torch.device | None
    name_prefix: str


@dataclass(frozen=True)
class StreamingOptimizerFragmentConfig:
    """Configuration for streaming optimizer state synchronization."""

    manager: Any
    fragment_id: int
    name_prefix: str
    param_entries: list[tuple[str, nn.Parameter]]
    optimizer: Optimizer
    state_key: str
    sync_every: int
    backup_device: torch.device | None
    pin_memory: bool
    use_bucketization: bool
    bucket_cap_mb: float | None
    should_quantize: bool


@dataclass(frozen=True)
class DesLocControllerConfig:
    """Configuration payload for :class:`DesLocController`."""

    manager: Any
    model: nn.Module
    optimizer: Optimizer
    param_sync_every: int
    optimizer_sync_every: int | list[int] | dict[str, int] | None
    backup_device: torch.device | None
    pin_memory: bool
    name_prefix: str
    quorum_timeout_seconds: int
    param_entries: list[tuple[str, nn.Parameter]] | None = None
    outer_optimizer: DesLocOuterOptimizerConfig | Optimizer | None = None
    log_outer_metrics: bool = False
    metrics_logger: Callable[[dict[str, float]], None] | None = None
    checkpoint_outer_optimizer: bool = True
    disable_optimizer_state_sync: bool = False


@dataclass(frozen=True)
class DesLocFTOptimizersConfig:
    """Configuration for constructing :class:`DesLocFTOptimizersContainer`."""

    model_parts: list[nn.Module]
    optimizer_cls: type[torch.optim.Optimizer]
    optimizer_kwargs: dict[str, Any]
    ft_manager: Any
    desloc_config: DesLocConfig
    use_ft_optimizer: bool = True
    param_groups: list[dict[str, Any]] | None = None
    outer_optimizer: (
        DesLocOuterOptimizerConfig | Optimizer | list[Optimizer] | None
    ) = None
    streaming: "DesLocStreamingConfig | None" = None


OptimizerStateGroup = Literal["first_moment", "second_moment", None]
_FIRST_MOMENT_ALIASES: tuple[str, ...] = ("exp_avg", "momentum", "momentum_buffer", "first_moment")
_SECOND_MOMENT_ALIASES: tuple[str, ...] = ("exp_avg_sq", "second_moment")


def _classify_optimizer_state_key(state_key: str) -> OptimizerStateGroup:
    """Return the moment category for a discovered optimizer state key."""
    lowered = state_key.lower()
    if "exp_avg_sq" in lowered or "second_moment" in lowered:
        return "second_moment"
    if "exp_avg" in lowered or "momentum" in lowered:
        return "first_moment"
    return None


def _broadcast_moment_intervals(intervals: list[int], state_keys: list[str]) -> list[int] | None:
    """Map two sync cadences onto first- and second-moment optimizer states.

    Args:
        intervals: Two-element list of sync cadences in the order [first_moment, second_moment].
        state_keys: Discovered optimizer state tensor keys.

    Returns:
        A list of sync cadences aligned with ``state_keys`` or ``None`` if any key
        cannot be classified as a first- or second-moment state.
    """
    if len(intervals) != 2:
        return None

    first_interval, second_interval = intervals
    resolved: list[int] = []
    for key in state_keys:
        category = _classify_optimizer_state_key(key)
        if category == "first_moment":
            resolved.append(first_interval)
        elif category == "second_moment":
            resolved.append(second_interval)
        else:
            return None
    return resolved


def _resolve_interval_from_mapping(state_key: str, mapping: dict[str, int]) -> int | None:
    """Resolve a sync cadence for ``state_key`` using explicit and alias mappings.

    Args:
        state_key: Optimizer state tensor key (e.g., ``exp_avg`` or ``momentum_buffer``).
        mapping: User-provided sync cadences keyed by state names or aliases.

    Returns:
        The resolved sync cadence or ``None`` if no mapping matches.
    """
    candidates: list[str] = [state_key, state_key.lower()]
    category = _classify_optimizer_state_key(state_key)
    if category == "first_moment":
        candidates.extend(_FIRST_MOMENT_ALIASES)
        candidates.extend(alias.lower() for alias in _FIRST_MOMENT_ALIASES)
    elif category == "second_moment":
        candidates.extend(_SECOND_MOMENT_ALIASES)
        candidates.extend(alias.lower() for alias in _SECOND_MOMENT_ALIASES)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in mapping:
            return int(mapping[candidate])
    return None


def _extract_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached clone of ``tensor`` on its local device."""
    local = tensor.to_local() if DTensor is not None and isinstance(tensor, DTensor) else tensor
    return local.detach().clone()


def _copy_into_tensor(param: torch.Tensor, value: torch.Tensor) -> None:
    """Copy ``value`` into ``param`` handling ``DTensor`` transparently."""
    if DTensor is not None and isinstance(param, DTensor):  # pragma: no cover - DTensor
        param.copy_(
            DTensor.from_local(
                value,
                param.device_mesh,
                param.placements,
                shape=param.shape,
            )
        )
    else:
        param.copy_(value)


def _zero_optimizer_grads(optimizer: Optimizer | None) -> None:
    """Zero gradients on the provided optimizer, preferring ``set_to_none=True``."""
    if optimizer is None:
        return
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:  # pragma: no cover - optimizer signature variance
        optimizer.zero_grad()


def _resolve_param_owner(optimizer: Optimizer, param: nn.Parameter) -> Optimizer:
    """Return the concrete optimizer responsible for ``param``."""
    param_owner: dict[nn.Parameter, Optimizer] | None = getattr(getattr(optimizer, "state", None), "_param_owner", None)
    if isinstance(param_owner, dict):
        owner = param_owner.get(param)
        if isinstance(owner, Optimizer):
            return owner

    inner_opts = getattr(optimizer, "optimizers", None)
    if isinstance(inner_opts, list):
        for opt in inner_opts:
            if param in getattr(opt, "state", {}):
                return opt
            for group in getattr(opt, "param_groups", []):
                if param in group.get("params", []):
                    return opt

    return optimizer


def _partition_named_parameters(
    model: nn.Module,
    fragments: int,
    *,
    allowed_params: set[nn.Parameter] | None = None,
    strategy: str = "strided",
    custom_fragments: Sequence[Sequence[str]] | None = None,
) -> list[list[tuple[str, nn.Parameter]]]:
    """Partition model parameters into ``fragments`` buckets."""
    if fragments <= 0:
        msg = "desloc.streaming.fragments must be a positive integer."
        raise ValueError(msg)

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if allowed_params is None or param in allowed_params
    ]
    if not named_params:
        return []

    fragments = min(max(1, fragments), len(named_params))
    if custom_fragments is not None:
        return _partition_from_custom_spec(named_params, fragments, custom_fragments)

    strategy = strategy.lower()
    if strategy == "strided":
        return _partition_strided(named_params, fragments)
    if strategy == "sequential":
        return _partition_sequential(named_params, fragments)
    if strategy == "balanced":
        return _partition_balanced(named_params, fragments)
    msg = f"Unknown DES-LOC streaming fragment strategy '{strategy}'."
    raise ValueError(msg)


_GroupedParams = list[list[tuple[str, nn.Parameter]]]


def _partition_strided(
    named_params: list[tuple[str, nn.Parameter]],
    fragments: int,
) -> _GroupedParams:
    groups = _group_parameters_for_striding(named_params)
    non_layer_groups: _GroupedParams = []
    layer_groups: _GroupedParams = []
    for group in groups:
        name = group[0][0]
        if _extract_layer_index(name) is None:
            non_layer_groups.append(group)
        else:
            layer_groups.append(group)

    layer_fragment_count = max(1, fragments)
    buckets: _GroupedParams = [[]]
    for group in non_layer_groups:
        buckets[0].extend(group)

    layer_buckets: _GroupedParams = [[] for _ in range(layer_fragment_count)]
    for idx, group in enumerate(layer_groups):
        slot = idx % layer_fragment_count
        layer_buckets[slot].extend(group)

    for bucket in layer_buckets:
        if bucket:
            buckets.append(bucket)

    return [bucket for bucket in buckets if bucket]


def _partition_sequential(
    named_params: list[tuple[str, nn.Parameter]],
    fragments: int,
) -> list[list[tuple[str, nn.Parameter]]]:
    bucket_size = math.ceil(len(named_params) / fragments)
    ordered = [
        named_params[idx : idx + bucket_size]
        for idx in range(0, len(named_params), bucket_size)
    ]
    return [bucket for bucket in ordered if bucket]


def _partition_balanced(
    named_params: list[tuple[str, nn.Parameter]],
    fragments: int,
) -> list[list[tuple[str, nn.Parameter]]]:
    if fragments == 1:
        return [named_params]

    buckets: list[list[tuple[int, str, nn.Parameter]]] = [[] for _ in range(fragments)]
    bucket_sizes = [0 for _ in range(fragments)]

    indexed = [
        (idx, name, param) for idx, (name, param) in enumerate(named_params)
    ]
    indexed.sort(key=lambda item: item[2].numel(), reverse=True)

    for original_idx, name, param in indexed:
        slot = min(range(fragments), key=lambda i: bucket_sizes[i])
        buckets[slot].append((original_idx, name, param))
        bucket_sizes[slot] += int(param.numel())

    ordered: list[list[tuple[str, nn.Parameter]]] = []
    for bucket in buckets:
        if not bucket:
            continue
        bucket.sort(key=lambda item: item[0])
        ordered.append([(name, param) for _, name, param in bucket])

    return ordered


def _partition_from_custom_spec(
    named_params: list[tuple[str, nn.Parameter]],
    fragments: int,
    custom_fragments: Sequence[Sequence[str]],
) -> list[list[tuple[str, nn.Parameter]]]:
    buckets_spec = [tuple(fragment) for fragment in custom_fragments if fragment]
    if not buckets_spec:
        msg = "desloc.streaming.custom_fragments must contain at least one selector."
        raise ValueError(msg)
    if len(buckets_spec) != fragments:
        msg = "desloc.streaming.custom_fragments must match desloc.streaming.fragments."
        raise ValueError(msg)

    param_map = dict(named_params)
    remaining = set(param_map.keys())

    partitions: list[list[tuple[str, nn.Parameter]]] = []
    for bucket_idx, selectors in enumerate(buckets_spec):
        bucket: list[tuple[str, nn.Parameter]] = []
        for selector in selectors:
            matches = [
                name for name in list(remaining) if fnmatch(name, selector)
            ]
            if not matches:
                msg = (
                    f"DES-LOC custom fragment {bucket_idx} selector '{selector}' "
                    "did not match any parameter."
                )
                raise ValueError(msg)
            for name in sorted(matches):
                bucket.append((name, param_map[name]))
                remaining.remove(name)
        if bucket:
            partitions.append(bucket)

    if remaining:
        unused = ", ".join(list(sorted(remaining))[:3])
        msg = (
            "DES-LOC custom fragments must cover every parameter; "
            f"remaining parameters include: {unused}..."
        )
        raise ValueError(msg)
    return partitions


def _group_parameters_for_striding(
    named_params: list[tuple[str, nn.Parameter]],
) -> _GroupedParams:
    groups: list[list[tuple[str, nn.Parameter]]] = []
    current_group: list[tuple[str, nn.Parameter]] = []
    current_layer: int | None = None

    for name, param in named_params:
        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_layer = None
            groups.append([(name, param)])
            continue

        if current_layer is None:
            current_layer = layer_idx
        if layer_idx != current_layer:
            groups.append(current_group)
            current_group = [(name, param)]
            current_layer = layer_idx
        else:
            current_group.append((name, param))

    if current_group:
        groups.append(current_group)

    return groups


def _extract_layer_index(param_name: str) -> int | None:
    token = "layers."
    idx = param_name.find(token)
    if idx == -1:
        return None
    remainder = param_name[idx + len(token) :]
    digits: list[str] = []
    for char in remainder:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:  # pragma: no cover - defensive
        return None


def _contains_layer_params(partition: list[tuple[str, nn.Parameter]]) -> bool:
    return any(name.startswith("layers.") for name, _ in partition)


def _merge_non_layer_partition(
    partitions: list[list[tuple[str, nn.Parameter]]],
) -> list[list[tuple[str, nn.Parameter]]]:
    non_layer_idx = next(
        (
            idx
            for idx, partition in enumerate(partitions)
            if partition and not _contains_layer_params(partition)
        ),
        None,
    )
    if non_layer_idx is None:
        return partitions

    target_idx = next(
        (
            idx
            for idx, partition in enumerate(partitions)
            if idx != non_layer_idx and _contains_layer_params(partition)
        ),
        None,
    )
    if target_idx is None:
        return partitions

    partitions[target_idx] = partitions[non_layer_idx] + partitions[target_idx]
    del partitions[non_layer_idx]
    return partitions


def _component_key_from_name(param_name: str) -> str:
    if param_name.startswith("layers."):
        parts = param_name.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
        return ".".join(parts[: len(parts)])
    return param_name.split(".")[0]


def _format_fragment_membership(names: Sequence[str], limit: int = 8) -> str:
    """Return a short string describing which tensors belong to a fragment."""
    if not names:
        return "none"
    if len(names) <= limit:
        return ", ".join(names)
    remaining = len(names) - limit
    return f"{', '.join(names[:limit])}, ... (+{remaining} more)"


def _get_global_step(manager: Any) -> int | None:
    """Best-effort retrieval of the TorchFT global step."""
    step_attr = getattr(manager, "current_step", None)
    if step_attr is None:
        return None
    try:
        return int(step_attr())
    except TypeError:
        pass
    try:
        # Handle property-like attributes or tensor scalars.
        if hasattr(step_attr, "item"):
            return int(step_attr.item())
        return int(step_attr)
    except Exception:
        return None


class _BaseFragment:
    def __init__(self, sync_every: int) -> None:
        if sync_every <= 0:
            message = "sync_every must be a positive integer"
            raise ValueError(message)
        self.sync_every = sync_every
        self._local_step = 0

    def tick(self) -> bool:
        """Advance the local fragment clock and report readiness."""
        self._local_step += 1
        return self._local_step >= self.sync_every

    def reset(self) -> None:
        self._local_step = 0

    def prepare_sync(self) -> list[Any]:
        raise NotImplementedError

    def perform_sync(self) -> None:
        raise NotImplementedError

    def save_state(self) -> None:
        raise NotImplementedError

    def restore_state(self) -> None:
        raise NotImplementedError


class _ParameterFragment(_BaseFragment):
    """Handles parameter state replication and synchronization."""

    def __init__(self, config: ParameterFragmentConfig) -> None:
        super().__init__(config.sync_every)
        self._manager = config.manager
        self._model = config.model
        self._param_entries = config.param_entries
        self._backup_device = config.backup_device
        self._pin_memory = config.pin_memory
        self._name_prefix = config.name_prefix

        entries = (
            self._param_entries
            if self._param_entries is not None
            else list(self._model.named_parameters())
        )
        self._param_map = dict(entries)
        self._original_parameters: dict[str, torch.Tensor] = {}
        self._averaged_parameters: list[tuple[str, torch.Tensor]] = []

        outer_spec = config.outer_optimizer
        self._outer_optimizer: Optimizer | None = None
        self._reference_synced = outer_spec is None
        self._reference_pending: list[tuple[str, torch.Tensor]] = []
        self._log_outer_metrics = config.log_outer_metrics
        self._metrics_logger = config.metrics_logger
        self._checkpoint_outer_optimizer = config.checkpoint_outer_optimizer
        if isinstance(outer_spec, Optimizer):
            self._outer_optimizer = outer_spec
        elif isinstance(outer_spec, DesLocOuterOptimizerConfig):
            optimizer_cls = outer_spec.resolve_optimizer_cls()
            params = [p for p in self._model.parameters() if p.requires_grad]
            if not params:
                msg = "DES-LOC outer optimizer requires at least one trainable parameter."
                raise ValueError(msg)
            self._outer_optimizer = optimizer_cls(params, **outer_spec.kwargs)
        elif outer_spec is not None:
            msg = "outer_optimizer must be an Optimizer, DesLocOuterOptimizerConfig, or None."
            raise TypeError(msg)

        self._init_backup_storage(entries)
        self.save_state()
        if self._outer_optimizer is not None:
            self._reference_synced = True

    def set_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        self._metrics_logger = logger_fn

    def _iter_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if self._param_entries is not None:
            return self._param_entries
        return list(self._model.named_parameters())

    def _refresh_param_map(self) -> None:
        self._param_map = dict(self._iter_named_parameters())

    def _init_backup_storage(self, entries: list[tuple[str, nn.Parameter]]) -> None:
        for name, param in entries:
            local_tensor = _extract_local_tensor(param.data)
            device = self._backup_device if self._backup_device is not None else local_tensor.device
            backup = torch.empty_like(local_tensor, device=device)
            if self._pin_memory and backup.device.type == "cpu" and torch.cuda.is_available():
                backup = backup.pin_memory()
            self._original_parameters[name] = backup

    def save_state(self) -> None:
        with torch.no_grad():
            self._refresh_param_map()
            for name, param in self._iter_named_parameters():
                self._original_parameters[name].copy_(_extract_local_tensor(param.data), non_blocking=True)

    def restore_state(self) -> None:
        with torch.no_grad():
            self._refresh_param_map()
            for name, param in self._iter_named_parameters():
                _copy_into_tensor(param.data, self._original_parameters[name])

    def prepare_sync(self) -> list[Any]:
        if self._outer_optimizer is not None and not self._reference_synced:
            # Ensure backups reflect the current model weights (e.g. after checkpoint load).
            self.save_state()
        self._averaged_parameters.clear()
        work_items: list[Any] = []
        self._refresh_param_map()
        for name, param in self._iter_named_parameters():
            avg_param = _extract_local_tensor(param.data)
            work_items.append(self._manager.allreduce(avg_param))
            self._averaged_parameters.append((name, avg_param))

        if self._outer_optimizer is not None and not self._reference_synced:
            self._reference_pending.clear()
            for name, avg_param in self._averaged_parameters:
                param = self._param_map[name]
                if not param.requires_grad:
                    continue
                reference = self._original_parameters[name].to(
                    device=avg_param.device,
                    dtype=avg_param.dtype,
                    copy=True,
                )
                work_items.append(self._manager.allreduce(reference))
                self._reference_pending.append((name, reference))

        return work_items

    def perform_sync(self) -> None:
        if self._outer_optimizer is not None and not self._reference_synced:
            for name, reference in self._reference_pending:
                backup = self._original_parameters[name]
                backup.copy_(reference.to(backup.device, dtype=backup.dtype))
            self._reference_pending.clear()
            self._reference_synced = True

        if self._outer_optimizer is None:
            with torch.no_grad():
                for name, avg_param in self._averaged_parameters:
                    param = self._param_map[name]
                    _copy_into_tensor(param.data, avg_param)
            return

        pseudo_norm_sq = 0.0
        grads_assigned = False
        with torch.no_grad():
            for name, avg_param in self._averaged_parameters:
                param = self._param_map[name]
                if not param.requires_grad:
                    _copy_into_tensor(param.data, avg_param)
                    continue

                reference_native = self._original_parameters[name].to(
                    device=avg_param.device,
                    dtype=avg_param.dtype,
                )
                averaged_native = avg_param

                # Ensure every replica applies gradients starting from the shared reference.
                _copy_into_tensor(param.data, reference_native)

                grad_native = reference_native - averaged_native
                pseudo_norm_sq += grad_native.pow(2).sum().item()
                param.grad = grad_native
                grads_assigned = True

        if not grads_assigned:
            with torch.no_grad():
                for name, avg_param in self._averaged_parameters:
                    param = self._param_map[name]
                    _copy_into_tensor(param.data, avg_param)
            return

        self._outer_optimizer.step()
        _zero_optimizer_grads(self._outer_optimizer)

        if self._log_outer_metrics and self._metrics_logger is not None:
            metrics: dict[str, float] = {"desloc_outer/pseudo_grad_l2": math.sqrt(max(pseudo_norm_sq, 0.0))}
            momentum_norm_sq = 0.0
            has_momentum = False
            if isinstance(self._outer_optimizer, torch.optim.SGD):
                for state in self._outer_optimizer.state.values():
                    buffer = state.get("momentum_buffer")
                    if isinstance(buffer, torch.Tensor):
                        has_momentum = True
                        momentum_norm_sq += buffer.pow(2).sum().item()
            if has_momentum:
                metrics["desloc_outer/momentum_l2"] = math.sqrt(max(momentum_norm_sq, 0.0))
            try:
                self._metrics_logger(metrics)
            except Exception:  # pragma: no cover - diagnostics only
                logger.exception("DES-LOC failed to log outer optimizer metrics; continuing.")

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            if state_dict:
                for name, tensor in state_dict.items():
                    if name in self._original_parameters:
                        self._original_parameters[name].copy_(tensor)
            else:
                # Older checkpoints might not have stored the DES-LOC state; fall back to fresh capture.
                self.save_state()
            self._reference_synced = False
            self._reference_pending.clear()

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_parameters

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_params",
            load_fn,
            save_fn,
        )

        if self._outer_optimizer is not None and self._checkpoint_outer_optimizer:

            def load_outer(state_dict: dict[str, Any]) -> None:
                self._outer_optimizer.load_state_dict(state_dict)

            def save_outer() -> dict[str, Any]:
                return self._outer_optimizer.state_dict()

            self._manager.register_state_dict_fn(
                f"{self._name_prefix}_outer_optimizer",
                load_outer,
                save_outer,
            )


class _OuterOptimizingParameterFragment(_ParameterFragment):
    """Marker subclass instantiated when an outer optimizer is configured."""

    pass


class _OptimizerStateFragment(_BaseFragment):
    """Synchronize a specific optimizer state tensor across replicas."""

    def __init__(self, config: OptimizerFragmentConfig) -> None:
        super().__init__(config.sync_every)
        self._manager = config.manager
        self._model = config.model
        self._param_entries = config.param_entries
        self._optimizer = config.optimizer
        self.state_key = config.state_key
        self._backup_device = config.backup_device
        self._name_prefix = config.name_prefix

        entries = (
            self._param_entries
            if self._param_entries is not None
            else list(self._model.named_parameters())
        )
        self._param_map = dict(entries)
        self._state_owner: dict[str, Optimizer] = {}
        self._original_state_tensors: dict[str, torch.Tensor] = {}
        self._averaged_state_tensors: list[tuple[str, Optimizer, torch.Tensor]] = []

        self._init_backup_storage(entries)
        self.save_state()

    def _iter_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        if self._param_entries is not None:
            return self._param_entries
        return list(self._model.named_parameters())

    def _refresh_param_map(self) -> None:
        self._param_map = dict(self._iter_named_parameters())

    def _init_backup_storage(self, entries: list[tuple[str, nn.Parameter]]) -> None:
        for name, param in entries:
            owner = _resolve_param_owner(self._optimizer, param)
            state = owner.state.get(param, {})
            tensor = state.get(self.state_key)
            if tensor is None:
                print(
                    f"[DESLOC DEBUG] skipping state_key={self.state_key} param={name} "
                    f"owner={type(owner).__name__} reason=missing_state"
                )
                continue
            device = self._backup_device if self._backup_device is not None else tensor.device
            self._original_state_tensors[name] = torch.empty_like(tensor, device=device)
            self._state_owner[name] = owner
            print(
                f"[DESLOC DEBUG] tracking state_key={self.state_key} param={name} "
                f"owner={type(owner).__name__} shape={tuple(tensor.shape)}"
            )

    def save_state(self) -> None:
        with torch.no_grad():
            self._refresh_param_map()
            for name, backup in self._original_state_tensors.items():
                param = self._param_map[name]
                owner = self._state_owner.get(name) or _resolve_param_owner(self._optimizer, param)
                self._state_owner[name] = owner
                tensor = owner.state[param][self.state_key]
                backup.copy_(tensor, non_blocking=True)

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, backup in self._original_state_tensors.items():
                param = self._param_map[name]
                owner = self._state_owner.get(name) or _resolve_param_owner(self._optimizer, param)
                self._state_owner[name] = owner
                state = owner.state.get(param, {})
                tensor = state.get(self.state_key)
                if tensor is not None:
                    tensor.copy_(backup)

    def prepare_sync(self) -> list[Any]:
        self._averaged_state_tensors.clear()
        work_items: list[Any] = []
        print(
            f"[DESLOC DEBUG] preparing sync for state_key={self.state_key} "
            f"params={list(self._original_state_tensors.keys())} "
            f"optimizer={type(self._optimizer).__name__}"
        )
        for name in self._original_state_tensors:
            param = self._param_map[name]
            owner = _resolve_param_owner(self._optimizer, param)
            self._state_owner[name] = owner
            state_tensor = owner.state[param][self.state_key]
            avg_state = state_tensor.detach().clone()
            work_items.append(self._manager.allreduce(avg_state))
            self._averaged_state_tensors.append((name, owner, avg_state))
        return work_items

    def perform_sync(self) -> None:
        with torch.no_grad():
            print(
                f"[DESLOC DEBUG] applying averaged state_key={self.state_key} "
                f"params={list(self._original_state_tensors.keys())} "
                f"optimizer={type(self._optimizer).__name__} "
                f"step={self._manager.current_step() if hasattr(self, '_manager') else 'unknown'}"
            )
            for name, owner, averaged in self._averaged_state_tensors:
                param = self._param_map[name]
                state = owner.state.setdefault(param, {})
                target = state.get(self.state_key)
                if target is None:
                    state[self.state_key] = averaged.clone()
                    target = state[self.state_key]
                target.copy_(averaged)

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            for name, tensor in state_dict.items():
                if name in self._original_state_tensors:
                    self._original_state_tensors[name].copy_(tensor)

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_state_tensors

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_state_{self.state_key}",
            load_fn,
            save_fn,
        )


class _StreamingOptimizerStateFragment(_BaseFragment):
    """Streaming-aware optimizer state fragment."""

    bucket_cap_mb: int = 1 * 1024 * 1024 * 1024
    use_bucketization: bool = False

    def __init__(self, config: StreamingOptimizerFragmentConfig) -> None:
        super().__init__(config.sync_every)
        self._manager = config.manager
        self._fragment_id = config.fragment_id
        self._name_prefix = config.name_prefix
        self._param_entries = config.param_entries
        self._param_map = {name: param for name, param in self._param_entries}
        self._optimizer = config.optimizer
        self.state_key = config.state_key
        self._backup_device = config.backup_device
        self._pin_memory = config.pin_memory
        self._should_quantize = config.should_quantize
        self._current_sync_step: int | None = None

        self._state_owner: dict[str, Optimizer] = {}
        self._original_state_tensors: dict[str, torch.Tensor] = {}
        self._averaged_state_tensors: list[tuple[str, Optimizer, torch.Tensor]] = []
        self._allreduce_work: list[Work] = []
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._stop_event: torch.cuda.Event | None = None

        if config.bucket_cap_mb is not None:
            self.bucket_cap_mb = int(config.bucket_cap_mb * 1024 * 1024)

        if os.getenv(USE_BUCKETIZATION_ENV, "False") == "True":
            self.use_bucketization = True
        else:
            self.use_bucketization = config.use_bucketization

        self._init_backup_storage()
        self.save_state()

    def set_step_context(self, step: int) -> None:
        self._current_sync_step = step

    @property
    def fragment_id(self) -> int:
        return self._fragment_id

    @property
    def parameter_names(self) -> list[str]:
        return [name for name, _ in self._param_entries]

    def _init_backup_storage(self) -> None:
        for name, param in self._param_entries:
            owner = _resolve_param_owner(self._optimizer, param)
            state = owner.state.get(param, {})
            tensor = state.get(self.state_key)
            if tensor is None:
                print(
                    f"[DESLOC DEBUG] streaming skipping state_key={self.state_key} param={name} "
                    f"owner={type(owner).__name__} reason=missing_state"
                )
                continue
            device = self._backup_device if self._backup_device is not None else tensor.device
            backup = torch.empty_like(tensor, device=device)
            if (
                self._pin_memory
                and backup.device.type == "cpu"
                and torch.cuda.is_available()
                and not backup.is_pinned()
            ):
                backup = backup.pin_memory()
            self._original_state_tensors[name] = backup
            self._state_owner[name] = owner
            print(
                f"[DESLOC DEBUG] streaming tracking state_key={self.state_key} param={name} "
                f"owner={type(owner).__name__} shape={tuple(tensor.shape)}"
            )

    def save_state(self) -> None:
        with torch.no_grad():
            for name, backup in self._original_state_tensors.items():
                param = self._param_map[name]
                owner = self._state_owner.get(name) or _resolve_param_owner(self._optimizer, param)
                self._state_owner[name] = owner
                tensor = owner.state[param][self.state_key]
                backup.copy_(tensor, non_blocking=True)

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, backup in self._original_state_tensors.items():
                param = self._param_map[name]
                owner = self._state_owner.get(name) or _resolve_param_owner(self._optimizer, param)
                self._state_owner[name] = owner
                state = owner.state.get(param, {})
                tensor = state.get(self.state_key)
                if tensor is not None:
                    tensor.copy_(backup)

    def prepare_sync(self) -> None:
        if not self._original_state_tensors:
            return
        assert not self._allreduce_work
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())

        logger.info(
            "DES-LOC streaming optimizer state '%s' fragment=%s sync starting (step=%s, manager_step=%s)",
            self.state_key,
            self._fragment_id,
            self._current_sync_step if self._current_sync_step is not None else "unknown",
            self._manager.current_step(),
        )
        print(
            f"[DESLOC DEBUG] streaming prepare state_key={self.state_key} fragment={self._fragment_id} "
            f"params={list(self._original_state_tensors.keys())} "
            f"optimizer={type(self._optimizer).__name__}"
        )

        context = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
        with context:
            self._capture_states()
            self._allreduce_states()
        print(
            f"[DESLOC DEBUG] streaming allreduce queued for state_key={self.state_key} "
            f"fragment={self._fragment_id} work_items={len(self._allreduce_work)}"
        )

    def _capture_states(self) -> None:
        self._averaged_state_tensors.clear()
        with torch.no_grad():
            for name in self._original_state_tensors:
                param = self._param_map[name]
                owner = _resolve_param_owner(self._optimizer, param)
                self._state_owner[name] = owner
                tensor = owner.state[param][self.state_key]
                clone = tensor.detach().clone()
                self._averaged_state_tensors.append((name, owner, clone))
        print(
            f"[DESLOC DEBUG] streaming captured {len(self._averaged_state_tensors)} tensors for state_key={self.state_key} "
            f"fragment={self._fragment_id}"
        )

    def _allreduce_states(self) -> None:
        tensors = [tensor for _, _, tensor in self._averaged_state_tensors]
        if not tensors:
            return
        if self.use_bucketization:
            self._bucketize_and_allreduce(tensors)
            return
        for tensor in tensors:
            work = self._manager.allreduce(
                tensor,
                should_quantize=self._should_quantize,
            )
            self._allreduce_work.append(work)

    def _bucketize_and_allreduce(self, tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return

        bucket_size_bytes = self.bucket_cap_mb
        offset = 0
        flat_index = 0
        total_elems = sum(t.numel() for t in tensors)
        dtype = tensors[0].dtype
        device = tensors[0].device

        while offset < total_elems:
            chunk_elems = min(bucket_size_bytes // tensors[0].element_size(), total_elems - offset)
            flat_buffer = torch.zeros(chunk_elems, dtype=dtype, device=device)

            pack_offset = 0
            bucket_tensors: list[tuple[torch.Tensor, int, int]] = []
            for tensor in tensors[flat_index:]:
                numel = tensor.numel()
                if pack_offset + numel > chunk_elems:
                    break
                flat_buffer[pack_offset : pack_offset + numel].copy_(tensor.view(-1))
                bucket_tensors.append((tensor, pack_offset, numel))
                pack_offset += numel
                flat_index += 1

            work = self._manager.allreduce(
                flat_buffer,
                should_quantize=self._should_quantize,
            )

            def callback(
                fut: torch.futures.Future[list[torch.Tensor]],
            ) -> list[torch.Tensor]:
                for tensor, tensor_offset, numel in bucket_tensors:
                    tensor.copy_(flat_buffer[tensor_offset : tensor_offset + numel].view_as(tensor))
                return []

            work.get_future().then(callback)
            self._allreduce_work.append(work)
            offset += chunk_elems

    def wait(self) -> None:
        if not self._allreduce_work:
            return
        if self._stream is not None and self._stop_event is not None:
            self._stop_event.synchronize()
            self._stop_event = None
        self._allreduce_work = []

    def perform_sync(self) -> None:
        if not self._averaged_state_tensors:
            return
        context = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
        with context:
            for work in self._allreduce_work:
                work.wait()
            if self._stream is not None:
                self._stop_event = torch.cuda.Event()
                self._stop_event.record()
        self.wait()

        should_commit = self._manager.should_commit()
        if should_commit:
            self._apply_states()
            self.save_state()
        else:
            self.restore_state()
        self._averaged_state_tensors.clear()
        logger.info(
            "DES-LOC streaming optimizer state '%s' fragment=%s sync complete (commit=%s, step=%s, manager_step=%s)",
            self.state_key,
            self._fragment_id,
            should_commit,
            self._current_sync_step if self._current_sync_step is not None else "unknown",
            self._manager.current_step(),
        )
        print(
            f"[DESLOC DEBUG] streaming perform_sync state_key={self.state_key} fragment={self._fragment_id} "
            f"commit={should_commit}"
        )
        self._current_sync_step = None

    def _apply_states(self) -> None:
        with torch.no_grad():
            for name, owner, averaged in self._averaged_state_tensors:
                param = self._param_map[name]
                state = owner.state.setdefault(param, {})
                target = state.get(self.state_key)
                if target is None:
                    state[self.state_key] = averaged.clone()
                    target = state[self.state_key]
                target.copy_(averaged)

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            for name, tensor in state_dict.items():
                if name in self._original_state_tensors:
                    self._original_state_tensors[name].copy_(tensor)

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_state_tensors

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_state_{self.state_key}",
            load_fn,
            save_fn,
        )


class _StreamingParameterFragment:
    """Streaming-enabled parameter fragment with asynchronous allreduce."""

    bucket_cap_mb: int = 1 * 1024 * 1024 * 1024
    use_bucketization: bool = False

    def __init__(
        self,
        *,
        manager,
        fragment_id: int,
        name_prefix: str,
        param_entries: list[tuple[str, nn.Parameter]],
        backup_device: torch.device | None,
        pin_memory: bool,
        outer_optimizer: Optimizer | None,
        inner_optimizer: Optimizer,
        fragment_sync_offset: int,
        fragment_sync_delay: int,
        sync_window: int,
        fragment_update_alpha: float,
        use_bucketization: bool,
        bucket_cap_mb: float | None,
        should_quantize: bool,
        log_outer_metrics: bool,
        metrics_logger: Callable[[dict[str, float]], None] | None,
        checkpoint_outer_optimizer: bool,
    ) -> None:
        self._manager = manager
        self._fragment_id = fragment_id
        self._name_prefix = name_prefix
        self._param_entries = param_entries
        self._param_map = {name: param for name, param in param_entries}
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        self._outer_optimizer = outer_optimizer
        self._inner_optimizer = inner_optimizer
        self._fragment_sync_offset = fragment_sync_offset
        self._fragment_sync_delay = fragment_sync_delay
        self._sync_window = sync_window
        self._fragment_update_alpha = fragment_update_alpha
        self._log_outer_metrics = log_outer_metrics
        self._metrics_logger = metrics_logger
        self._averaging_only = outer_optimizer is None
        self._should_quantize = should_quantize
        self._checkpoint_outer_optimizer = checkpoint_outer_optimizer
        self._current_sync_step: int | None = None

        self._grads: dict[str, torch.Tensor] = {}
        self._averaged_parameters: list[tuple[str, torch.Tensor]] = []
        self._local_parameters: dict[str, torch.Tensor] = {}
        self.original_parameters: dict[str, torch.Tensor] = {}

        self._allreduce_work: list[Work] = []
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._stop_event: torch.cuda.Event | None = None

        if bucket_cap_mb is not None:
            self.bucket_cap_mb = int(bucket_cap_mb * 1024 * 1024)

        if os.getenv(USE_BUCKETIZATION_ENV, "False") == "True":
            self.use_bucketization = True
        else:
            self.use_bucketization = use_bucketization

        self._init_backup_storage()
        self.save_parameters()

    def set_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        self._metrics_logger = logger_fn

    def set_step_context(self, step: int) -> None:
        self._current_sync_step = step

    @property
    def parameter_names(self) -> list[str]:
        return [name for name, _ in self._param_entries]

    @property
    def fragment_id(self) -> int:
        return self._fragment_id

    @property
    def fragment_sync_offset(self) -> int:
        return self._fragment_sync_offset

    @property
    def fragment_sync_delay(self) -> int:
        return self._fragment_sync_delay

    def _named_parameters(self):
        for name, param in self._param_entries:
            yield name, param

    def _init_backup_storage(self) -> None:
        for name, param in self._named_parameters():
            local_tensor = _extract_local_tensor(param.data)
            device = self._backup_device if self._backup_device is not None else local_tensor.device
            backup = torch.empty_like(local_tensor, device=device)
            if self._pin_memory and backup.device.type == "cpu" and torch.cuda.is_available():
                backup = backup.pin_memory()
            self.original_parameters[name] = backup

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, Any]) -> None:
            if not state_dict:
                self.save_parameters()
                return
            params_state = state_dict.get("original_parameters")
            if params_state is None:
                params_state = state_dict

            for name, tensor in params_state.items():
                if name in self.original_parameters:
                    self.original_parameters[name].copy_(tensor)

            if (
                self._outer_optimizer is not None
                and self._checkpoint_outer_optimizer
                and "outer_optimizer" in state_dict
            ):
                self._outer_optimizer.load_state_dict(state_dict["outer_optimizer"])

        def save_fn() -> dict[str, Any]:
            payload: dict[str, Any] = {
                "original_parameters": {
                    name: _extract_local_tensor(param)
                    for name, param in self.original_parameters.items()
                }
            }
            if self._outer_optimizer is not None and self._checkpoint_outer_optimizer:
                payload["outer_optimizer"] = self._outer_optimizer.state_dict()
            return payload

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_params",
            load_fn,
            save_fn,
        )

    def save_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self._named_parameters():
                self.original_parameters[name].copy_(_extract_local_tensor(param.data), non_blocking=True)

    def restore_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self._named_parameters():
                _copy_into_tensor(param.data, self.original_parameters[name])

    def _save_local_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self._named_parameters():
                self._local_parameters[name] = _extract_local_tensor(param.data)

    def _clear_local_parameters(self) -> None:
        self._local_parameters.clear()

    def _merge_parameters(self) -> None:
        if self._fragment_update_alpha <= 0 or not self._local_parameters:
            return
        with torch.no_grad():
            for name, param in self._named_parameters():
                local = self._local_parameters[name]
                if isinstance(param, DTensor):
                    param.data.lerp_(
                        DTensor.from_local(
                            local,
                            param.device_mesh,
                            param.placements,
                            shape=param.shape,
                            stride=param.stride(),
                        ),
                        self._fragment_update_alpha,
                    )
                else:
                    param.data.lerp_(local, self._fragment_update_alpha)

    def _save_grads(self) -> None:
        with torch.no_grad():
            for name, param in self._named_parameters():
                tensor = param.to_local() if DTensor is not None and isinstance(param, DTensor) else param
                pseudo = self.original_parameters[name].to(tensor.device) - tensor
                self._grads[name] = pseudo

    def _save_averaged_parameters(self) -> None:
        self._averaged_parameters.clear()
        with torch.no_grad():
            for name, param in self._named_parameters():
                self._averaged_parameters.append((name, _extract_local_tensor(param.data)))

    def _set_grads(self) -> None:
        with torch.no_grad():
            for name, param in self._named_parameters():
                grad = self._grads.pop(name, None)
                if grad is None:
                    continue
                if isinstance(param, DTensor):
                    param.grad = DTensor.from_local(
                        grad,
                        param.device_mesh,
                        param.placements,
                        shape=param.shape,
                        stride=param.stride(),
                    )
                else:
                    param.grad = grad

    def _apply_averaged_parameters(self) -> None:
        with torch.no_grad():
            for name, averaged in self._averaged_parameters:
                param = self._param_map[name]
                _copy_into_tensor(param.data, averaged)
        self._averaged_parameters.clear()

    def wait(self) -> None:
        if not self._allreduce_work:
            return
        if self._stream is not None and self._stop_event is not None:
            self._stop_event.synchronize()
            self._stop_event = None
        self._allreduce_work = []

    def _bucketize_and_allreduce(self, tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return

        bucket_size_bytes = self.bucket_cap_mb
        offset = 0
        flat_index = 0
        total_elems = sum(t.numel() for t in tensors)

        dtype = tensors[0].dtype
        device = tensors[0].device

        while offset < total_elems:
            chunk_elems = min(bucket_size_bytes // tensors[0].element_size(), total_elems - offset)
            flat_buffer = torch.zeros(chunk_elems, dtype=dtype, device=device)

            pack_offset = 0
            bucket_tensors: list[tuple[torch.Tensor, int, int]] = []
            for tensor in tensors[flat_index:]:
                numel = tensor.numel()
                if pack_offset + numel > chunk_elems:
                    break
                flat_buffer[pack_offset : pack_offset + numel].copy_(tensor.view(-1))
                bucket_tensors.append((tensor, pack_offset, numel))
                pack_offset += numel
                flat_index += 1

            work = self._manager.allreduce(
                flat_buffer,
                should_quantize=self._should_quantize,
            )

            def callback(
                fut: torch.futures.Future[list[torch.Tensor]],
            ) -> list[torch.Tensor]:
                for tensor, tensor_offset, numel in bucket_tensors:
                    tensor.copy_(flat_buffer[tensor_offset : tensor_offset + numel].view_as(tensor))
                return []

            work.get_future().then(callback)
            self._allreduce_work.append(work)
            offset += chunk_elems

    def _allreduce_grads(self) -> None:
        tensors = list(self._grads.values())
        if not tensors:
            return
        if self.use_bucketization:
            self._bucketize_and_allreduce(tensors)
            return
        for tensor in tensors:
            work = self._manager.allreduce(
                tensor,
                should_quantize=self._should_quantize,
            )
            self._allreduce_work.append(work)

    def _allreduce_parameters(self) -> None:
        tensors = [tensor for _, tensor in self._averaged_parameters]
        if not tensors:
            return
        if self.use_bucketization:
            self._bucketize_and_allreduce(tensors)
            return
        for tensor in tensors:
            work = self._manager.allreduce(
                tensor,
                should_quantize=self._should_quantize,
            )
            self._allreduce_work.append(work)

    def prepare_sync(self) -> None:
        assert not self._allreduce_work
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())

        logger.info(
            "DES-LOC streaming parameter fragment=%s sync starting (step=%s, manager_step=%s)",
            self._fragment_id,
            self._current_sync_step if self._current_sync_step is not None else "unknown",
            self._manager.current_step(),
        )

        context = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
        with context:
            if self._averaging_only:
                self._save_averaged_parameters()
                self._allreduce_parameters()
            else:
                self._save_grads()
                self._allreduce_grads()

    def _zero_outer_optimizer_grads(self) -> None:
        _zero_optimizer_grads(self._outer_optimizer)

    def _emit_outer_metrics(self, pseudo_norm_sq: float, momentum_norm_sq: float, has_momentum: bool) -> None:
        if not self._log_outer_metrics or self._metrics_logger is None:
            return
        metrics: dict[str, float] = {}
        metrics["desloc_outer/pseudo_grad_l2"] = math.sqrt(max(pseudo_norm_sq, 0.0))
        if has_momentum:
            metrics["desloc_outer/momentum_l2"] = math.sqrt(max(momentum_norm_sq, 0.0))
        try:
            self._metrics_logger(metrics)
        except Exception:  # pragma: no cover - diagnostics only
            logger.exception("DES-LOC streaming metrics logger failed.")

    def perform_sync(self) -> bool:
        assert self._allreduce_work
        context = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
        with context:
            for work in self._allreduce_work:
                work.wait()
            if self._stream is not None:
                self._stop_event = torch.cuda.Event()
                self._stop_event.record()

        self.wait()

        if not self._averaging_only:
            self._save_local_parameters()

        self.restore_parameters()
        should_commit = self._manager.should_commit()

        if should_commit:
            if self._averaging_only:
                self._apply_averaged_parameters()
                self.save_parameters()
            else:
                self._set_grads()
                self._outer_optimizer.step()
                self.save_parameters()
                self._merge_parameters()
                pseudo_norm_sq = 0.0
                momentum_norm_sq = 0.0
                has_momentum = False
                if self._log_outer_metrics:
                    for name, param in self._named_parameters():
                        grad = param.grad
                        if grad is not None:
                            pseudo_norm_sq += grad.pow(2).sum().item()
                    if isinstance(self._outer_optimizer, torch.optim.SGD):
                        for state in self._outer_optimizer.state.values():
                            buffer = state.get("momentum_buffer")
                            if isinstance(buffer, torch.Tensor):
                                has_momentum = True
                                momentum_norm_sq += buffer.pow(2).sum().item()
                self._emit_outer_metrics(pseudo_norm_sq, momentum_norm_sq, has_momentum)

            self._zero_outer_optimizer_grads()
        else:
            self.restore_parameters()

        self._clear_local_parameters()
        self._grads.clear()
        self._averaged_parameters.clear()

        logger.info(
            "DES-LOC streaming parameter fragment=%s sync complete (commit=%s, step=%s, manager_step=%s)",
            self._fragment_id,
            should_commit,
            self._current_sync_step if self._current_sync_step is not None else "unknown",
            self._manager.current_step(),
        )
        self._current_sync_step = None

        return should_commit


@dataclass
class _StreamingFragmentSchedule:
    fragment: _StreamingParameterFragment
    next_prepare_step: int
    next_sync_step: int
    pending: bool = False

    def advance(self, sync_window: int) -> None:
        self.next_prepare_step += sync_window
        self.next_sync_step += sync_window


class DesLocController:
    """Attach DES-LOC synchronization hooks to a PyTorch optimizer."""

    def __init__(self, config: DesLocControllerConfig) -> None:
        self._manager = config.manager
        self._model = config.model
        self._optimizer = config.optimizer
        self._backup_device = config.backup_device
        self._pin_memory = config.pin_memory
        self._name_prefix = config.name_prefix
        self._raw_optimizer_sync_config = config.optimizer_sync_every
        self._quorum_timeout = timedelta(seconds=max(1, config.quorum_timeout_seconds))
        self._optimizer_state_sync_enabled = not config.disable_optimizer_state_sync

        if config.param_entries is not None:
            opt_param_entries = list(config.param_entries)
        else:
            opt_params = {
                param
                for group in getattr(self._optimizer, "param_groups", [])
                for param in group["params"]
                if isinstance(param, nn.Parameter)
            }
            if not opt_params:
                msg = "DES-LOC streaming requires the optimizer to own at least one parameter."
                raise ValueError(msg)
            opt_param_entries = [
                (name, param) for name, param in self._model.named_parameters() if param in opt_params
            ]
            if not opt_param_entries:
                msg = "DES-LOC requires the optimizer to own at least one parameter."
                raise ValueError(msg)
        self._opt_param_entries = opt_param_entries

        param_fragment_cfg = ParameterFragmentConfig(
            manager=config.manager,
            model=config.model,
            param_entries=opt_param_entries,
            sync_every=config.param_sync_every,
            backup_device=config.backup_device,
            pin_memory=config.pin_memory,
            name_prefix=config.name_prefix,
            outer_optimizer=config.outer_optimizer,
            log_outer_metrics=config.log_outer_metrics,
            metrics_logger=config.metrics_logger,
            checkpoint_outer_optimizer=config.checkpoint_outer_optimizer,
        )
        fragment_cls = (
            _OuterOptimizingParameterFragment
            if param_fragment_cfg.outer_optimizer is not None
            else _ParameterFragment
        )
        self._param_fragment = fragment_cls(param_fragment_cfg)
        self._param_fragment.register_state_dict_fn()

        self._fragments: list[_BaseFragment] = [self._param_fragment]
        self._allreduce_work: list[Any] = []
        self._is_opt_init = not self._optimizer_state_sync_enabled

        self._hook = config.optimizer.register_step_post_hook(self._step_post_hook)
        self._warned_missing_step = False

    def close(self) -> None:
        """Detach the registered optimizer step hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def set_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        self._param_fragment.set_metrics_logger(logger_fn)

    def _iter_optimizer_states(self) -> Iterable[dict[str, Any]]:
        """Yield state dicts for the wrapped optimizer, including composites."""
        yield from self._optimizer.state.values()

        inner_opts = getattr(self._optimizer, "optimizers", None)
        if isinstance(inner_opts, list):
            for idx, opt in enumerate(inner_opts):
                print(f"[DESLOC DEBUG] inspecting inner optimizer[{idx}] {type(opt).__name__} for state discovery")
                yield from opt.state.values()

    def _resolve_optimizer_sync_intervals(self, state_keys: Iterable[str]) -> list[int]:
        keys = list(state_keys)
        if not keys:
            return []

        spec = self._raw_optimizer_sync_config
        if spec is None:
            return [self._param_fragment.sync_every for _ in keys]
        if isinstance(spec, int):
            return self._expand_single_interval(spec, keys)
        if isinstance(spec, list):
            return self._expand_list_intervals(spec, keys)
        if isinstance(spec, dict):
            return self._expand_dict_intervals(spec, keys)

        msg = f"optimizer_sync_every must be an int, list, dict, or None; received {type(spec)!r}"
        raise TypeError(msg)

    def _expand_single_interval(self, interval: int, keys: list[str]) -> list[int]:
        self._validate_positive_interval(interval)
        return [interval for _ in keys]

    def _expand_list_intervals(self, intervals: list[int], keys: list[str]) -> list[int]:
        normalized = [int(value) for value in intervals]
        for value in normalized:
            self._validate_positive_interval(value)
        if len(normalized) == len(keys):
            return normalized

        broadcasted = _broadcast_moment_intervals(normalized, keys)
        if broadcasted is not None:
            return broadcasted

        msg = (
            "Length of optimizer_sync_every list does not match discovered optimizer states; "
            "provide one value per state or two values for [first_moment, second_moment]."
        )
        raise ValueError(msg)

    def _expand_dict_intervals(self, mapping: dict[str, int], keys: list[str]) -> list[int]:
        normalized = {str(k): int(v) for k, v in mapping.items()}
        resolved: list[int] = []
        missing: list[str] = []
        for key in keys:
            value = _resolve_interval_from_mapping(key, normalized)
            if value is None:
                missing.append(key)
                continue
            self._validate_positive_interval(value)
            resolved.append(value)

        if missing:
            missing_keys = ", ".join(sorted(missing))
            msg = f"Missing DES-LOC sync interval for optimizer state(s): {missing_keys}."
            raise ValueError(msg)
        return resolved

    def _validate_positive_interval(self, value: int) -> None:
        if value <= 0:
            msg = "optimizer_sync_every values must be positive"
            raise ValueError(msg)

    def _lazy_init_optimizer_fragments(self) -> None:
        if not self._optimizer_state_sync_enabled:
            self._is_opt_init = True
            return
        state_sets = set()
        for state in self._iter_optimizer_states():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    state_sets.add(str(key))

        state_keys = sorted(state_sets)
        print(f"[DESLOC DEBUG] controller={self._name_prefix} discovered optimizer state keys: {state_keys}")
        sync_intervals = self._resolve_optimizer_sync_intervals(state_keys)

        if not state_keys and self._raw_optimizer_sync_config is not None:
            logger.warning(
                "DES-LOC optimizer_sync_every provided but no tensor states were discovered; skipping state synchronization."
            )

        for idx, key in enumerate(state_keys):
            fragment_config = OptimizerFragmentConfig(
                manager=self._manager,
                model=self._model,
                param_entries=self._opt_param_entries,
                optimizer=self._optimizer,
                state_key=key,
                sync_every=sync_intervals[idx],
                backup_device=self._backup_device,
                name_prefix=f"{self._name_prefix}_{key}",
            )
            fragment = _OptimizerStateFragment(fragment_config)
            fragment.register_state_dict_fn()
            self._fragments.append(fragment)
            param_names = [name for name, _ in self._opt_param_entries]
            print(
                f"[DESLOC DEBUG] controller={self._name_prefix} created optimizer-state fragment "
                f"state_key={key} sync_every={sync_intervals[idx]} params={param_names}"
            )

        self._is_opt_init = True

    def _step_post_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        if not self._is_opt_init:
            self._lazy_init_optimizer_fragments()

        global_step = _get_global_step(self._manager)
        if global_step is None and not self._warned_missing_step:
            print(
                f"[DESLOC DEBUG] controller={self._name_prefix} global_step unresolved; falling back to local clocks"
            )
            self._warned_missing_step = True
        ready_fragments = [fragment for fragment in self._fragments if fragment.tick()]

        if ready_fragments:
            print(
                f"[DESLOC DEBUG] controller={self._name_prefix} global_step={global_step} "
                f"ready_fragments={[type(f).__name__ for f in ready_fragments]} "
                f"sync_every={[f.sync_every for f in ready_fragments]} "
                f"local_steps={[f._local_step for f in ready_fragments]}"
            )
            self._sync(ready_fragments)

    def _sync(self, fragments: list[_BaseFragment]) -> None:
        self._manager.disallow_state_dict_read()
        try:
            try:
                self._manager.start_quorum(
                    allow_heal=False,
                    shrink_only=False,
                    timeout=self._quorum_timeout,
                )
            except TimeoutError as err:
                logger.warning(
                    "DES-LOC quorum timed out after %.1f seconds; skipping synchronization.",
                    self._quorum_timeout.total_seconds(),
                )
                self._manager.report_error(err)
                for fragment in fragments:
                    fragment.restore_state()
                    fragment.reset()
                return

            self._prepare_sync(fragments)
            self._perform_sync(fragments)
            for fragment in fragments:
                fragment.reset()
        finally:
            self._manager.allow_state_dict_read()

    def _prepare_sync(self, fragments: list[_BaseFragment]) -> None:
        self._allreduce_work.clear()
        for fragment in fragments:
            self._allreduce_work.extend(fragment.prepare_sync())

    def _perform_sync(self, fragments: list[_BaseFragment]) -> None:
        for work in self._allreduce_work:
            work.wait()

        commit_allowed = self._manager.should_commit()

        if commit_allowed:
            for fragment in fragments:
                fragment.perform_sync()
                fragment.save_state()
        else:
            for fragment in fragments:
                fragment.restore_state()


class StreamingDesLocController:
    """Streaming DES-LOC controller which mirrors TorchFT's Streaming DiLoCo."""

    def __init__(
        self,
        config: DesLocControllerConfig,
        streaming: DesLocStreamingConfig,
    ) -> None:
        self._manager = config.manager
        self._model = config.model
        self._optimizer = config.optimizer
        self._backup_device = config.backup_device
        self._pin_memory = config.pin_memory
        self._name_prefix = config.name_prefix
        self._raw_optimizer_sync_config = config.optimizer_sync_every
        self._quorum_timeout = timedelta(seconds=max(1, config.quorum_timeout_seconds))
        self._log_outer_metrics = config.log_outer_metrics
        self._metrics_logger = config.metrics_logger
        self._checkpoint_outer_optimizer = config.checkpoint_outer_optimizer
        self._streaming_cfg = streaming
        self._optimizer_state_sync_enabled = not config.disable_optimizer_state_sync
        self._warned_missing_step = False

        if config.param_entries is not None:
            opt_params = {param for _, param in config.param_entries}
        else:
            opt_params = {
                param
                for group in getattr(self._optimizer, "param_groups", [])
                for param in group["params"]
                if isinstance(param, nn.Parameter)
            }
            if not opt_params:
                msg = "DES-LOC streaming requires the optimizer to own at least one parameter."
                raise ValueError(msg)

        fragment_strategy = getattr(streaming, "fragment_strategy", "strided")
        custom_fragments = getattr(streaming, "custom_fragments", None)
        if fragment_strategy == "custom" and not custom_fragments:
            msg = "desloc.streaming.custom_fragments must be provided when using the 'custom' strategy."
            raise ValueError(msg)

        partitions = _partition_named_parameters(
            self._model,
            streaming.fragments,
            allowed_params=opt_params,
            strategy=fragment_strategy,
            custom_fragments=custom_fragments,
        )
        if not partitions:
            msg = "DES-LOC streaming requires at least one model parameter."
            raise ValueError(msg)

        if not streaming.separate_non_layer_fragment:
            before_len = len(partitions)
            partitions = _merge_non_layer_partition(partitions)
            if len(partitions) < before_len:
                logger.info("DES-LOC streaming merged non-layer parameters into fragment 0.")

        layer_fragment_indices = list(range(len(partitions)))

        layer_fragment_count = len(layer_fragment_indices)
        num_fragments = len(partitions)

        if config.param_sync_every < layer_fragment_count:
            msg = "desloc.param_sync_every must be >= the number of streaming fragments."
            raise ValueError(msg)
        if config.param_sync_every % layer_fragment_count != 0:
            msg = "desloc.param_sync_every must be divisible by the number of streaming fragments."
            raise ValueError(msg)

        self._sync_window = config.param_sync_every
        self._fragment_stride = self._sync_window // layer_fragment_count
        if streaming.sync_delay >= self._fragment_stride:
            msg = "desloc.streaming.sync_delay must be smaller than param_sync_every / fragments."
            raise ValueError(msg)
        if not (0.0 <= streaming.update_alpha <= 1.0):
            msg = "desloc.streaming.update_alpha must be between 0 and 1."
            raise ValueError(msg)

        outer_handles = self._build_outer_optimizer_handles(config.outer_optimizer, partitions)
        self._partitions = partitions
        self._fragment_sync_delay = streaming.sync_delay
        layer_offsets = self._resolve_fragment_offsets(layer_fragment_count, streaming)
        fragment_offsets = self._assign_fragment_offsets(
            num_fragments, layer_fragment_indices, layer_offsets
        )
        outer_checkpoint_flags = self._build_outer_checkpoint_flags(outer_handles)
        self._schedule_entries: list[_StreamingFragmentSchedule] = []
        self._fragments: list[_StreamingParameterFragment] = []
        for idx, (params, offset) in enumerate(zip(partitions, fragment_offsets, strict=True)):
            fragment = _StreamingParameterFragment(
                manager=self._manager,
                fragment_id=idx,
                name_prefix=f"{self._name_prefix}_fragment_{idx}",
                param_entries=params,
                backup_device=self._backup_device,
                pin_memory=self._pin_memory,
                outer_optimizer=outer_handles[idx],
                inner_optimizer=self._optimizer,
                fragment_sync_offset=offset,
                fragment_sync_delay=self._fragment_sync_delay,
                sync_window=self._sync_window,
                fragment_update_alpha=streaming.update_alpha,
                use_bucketization=streaming.use_bucketization,
                bucket_cap_mb=streaming.bucket_cap_mb,
                should_quantize=streaming.should_quantize,
                log_outer_metrics=self._log_outer_metrics,
                metrics_logger=self._metrics_logger,
                checkpoint_outer_optimizer=(
                    self._checkpoint_outer_optimizer and outer_checkpoint_flags[idx]
                ),
            )
            param_names = fragment.parameter_names
            logger.info(
                "DES-LOC streaming parameter fragment=%s initialized with %d parameters: %s",
                idx,
                len(param_names),
                _format_fragment_membership(param_names),
            )
            prepare_step = max(offset - self._fragment_sync_delay, 0)
            schedule_entry = _StreamingFragmentSchedule(
                fragment=fragment,
                next_prepare_step=prepare_step,
                next_sync_step=offset,
            )
            self._schedule_entries.append(schedule_entry)
            self._fragments.append(fragment)
        self._hooks: list[RemovableHandle] = []
        self._hooks.append(self._optimizer.register_step_pre_hook(self._step_pre_hook))
        self._hooks.append(self._optimizer.register_step_post_hook(self._step_post_hook))

        self._inner_step = 0
        self._state_cursor = 0
        self._optimizer_state_log_emitted = False
        self._optimizer_state_schedule = streaming.optimizer_state_schedule

        self._state_fragments_per_fragment: list[list[_StreamingOptimizerStateFragment]] = []
        self._is_opt_init = not self._optimizer_state_sync_enabled
        self._fragments_synced_this_step: set[int] = set()
        self._pending_aligned_state_frags: dict[int, list[tuple[_StreamingOptimizerStateFragment, int]]] = {}

        self._register_state_dict_functions()
        self._log_parameter_fragment_assignments()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def set_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        self._metrics_logger = logger_fn
        for fragment in self._fragments:
            fragment.set_metrics_logger(logger_fn if self._log_outer_metrics else None)

    def _register_state_dict_functions(self) -> None:
        for fragment in self._fragments:
            fragment.register_state_dict_fn()

    def _log_parameter_fragment_assignments(self) -> None:
        mapping: dict[str, int] = {}
        for fragment in self._fragments:
            for name in fragment.parameter_names:
                key = _component_key_from_name(name)
                existing = mapping.get(key)
                if existing is not None and existing != fragment.fragment_id:
                    logger.warning(
                        "DES-LOC streaming parameter component %s mapped to multiple fragments (%s, %s).",
                        key,
                        existing,
                        fragment.fragment_id,
                    )
                mapping[key] = fragment.fragment_id

        if not mapping:
            logger.info("DES-LOC streaming parameter fragments: none discovered.")
            return

        formatted = "; ".join(
            f"{component}->frag{fragment_id}"
            for component, fragment_id in sorted(mapping.items())
        )
        logger.info("DES-LOC streaming parameter fragments: %s", formatted)

    def _build_outer_optimizer_handles(
        self,
        outer_spec: DesLocOuterOptimizerConfig | Optimizer | list[Optimizer] | None,
        partitions: list[list[tuple[str, nn.Parameter]]],
    ) -> list[Optimizer | None]:
        if outer_spec is None:
            return [None for _ in partitions]
        if isinstance(outer_spec, list):
            if len(outer_spec) != len(partitions):
                msg = "When providing a list of outer optimizers, its length must match desloc.streaming.fragments."
                raise ValueError(msg)
            return outer_spec
        if isinstance(outer_spec, Optimizer):
            return [outer_spec for _ in partitions]
        if isinstance(outer_spec, DesLocOuterOptimizerConfig):
            handles: list[Optimizer] = []
            optimizer_cls = outer_spec.resolve_optimizer_cls()
            for params in partitions:
                trainable = [param for _, param in params if param.requires_grad]
                if not trainable:
                    msg = "DES-LOC outer optimizer requires at least one trainable parameter per fragment."
                    raise ValueError(msg)
                handles.append(optimizer_cls(trainable, **outer_spec.kwargs))
            return handles
        msg = "desloc.outer_optimizer must be a config, Optimizer, list of Optimizers, or None."
        raise TypeError(msg)

    def _build_outer_checkpoint_flags(self, outer_handles: list[Optimizer | None]) -> list[bool]:
        if not self._checkpoint_outer_optimizer:
            return [False for _ in outer_handles]
        seen: set[int] = set()
        flags: list[bool] = []
        for optimizer in outer_handles:
            if optimizer is None:
                flags.append(False)
                continue
            ident = id(optimizer)
            if ident in seen:
                flags.append(False)
                continue
            seen.add(ident)
            flags.append(True)
        return flags

    def _resolve_fragment_offsets(
        self,
        num_fragments: int,
        streaming: DesLocStreamingConfig,
    ) -> list[int]:
        fragment_sync_offsets = getattr(streaming, "fragment_sync_offsets", None)
        if fragment_sync_offsets is None:
            stride = self._sync_window / num_fragments
            offsets = [max(1, int(math.floor(stride * (idx + 1)))) for idx in range(num_fragments)]
            offsets[-1] = self._sync_window
        else:
            offsets = [int(value) for value in fragment_sync_offsets]
            if len(offsets) != num_fragments:
                msg = "desloc.streaming.fragment_sync_offsets must match the fragment count."
                raise ValueError(msg)

        if offsets != sorted(offsets):
            msg = "desloc.streaming.fragment_sync_offsets must be strictly increasing."
            raise ValueError(msg)
        if offsets[0] <= 0 or offsets[-1] > self._sync_window:
            msg = "desloc.streaming.fragment_sync_offsets must lie within the sync window."
            raise ValueError(msg)
        for offset in offsets:
            if offset <= self._fragment_sync_delay:
                msg = "Each fragment sync offset must exceed desloc.streaming.sync_delay."
                raise ValueError(msg)
        return offsets

    @staticmethod
    def _assign_fragment_offsets(
        total_fragments: int,
        layer_fragment_indices: list[int],
        layer_offsets: list[int],
    ) -> list[int]:
        offset_map: dict[int, int] = {}
        for slot, fragment_idx in enumerate(layer_fragment_indices):
            offset_map[fragment_idx] = layer_offsets[slot]
        default_offset = layer_offsets[0]
        for fragment_idx in range(total_fragments):
            offset_map.setdefault(fragment_idx, default_offset)
        return [offset_map[idx] for idx in range(total_fragments)]

    def _drive_fragment_schedule(self) -> None:
        if not self._schedule_entries:
            return
        for entry in self._schedule_entries:
            if not entry.pending and self._inner_step == entry.next_prepare_step:
                self._attempt_prepare_fragment(entry)
            if entry.pending and self._inner_step == entry.next_sync_step:
                self._complete_fragment_sync(entry)

    def _attempt_prepare_fragment(self, entry: _StreamingFragmentSchedule) -> None:
        fragment = entry.fragment
        try:
            self._manager.start_quorum(
                allow_heal=False,
                shrink_only=False,
                timeout=self._quorum_timeout,
            )
        except TimeoutError as err:
            logger.warning(
                "DES-LOC streaming quorum timed out after %.1f seconds; skipping synchronization.",
                self._quorum_timeout.total_seconds(),
            )
            self._manager.report_error(err)
            fragment.restore_parameters()
            entry.advance(self._sync_window)
            return

        logger.info(
            "Preparing fragment=%s step=%s",
            fragment.fragment_id,
            self._inner_step,
        )
        fragment.set_step_context(self._inner_step)
        fragment.prepare_sync()
        self._maybe_prepare_aligned_state_sync(fragment.fragment_id)
        entry.pending = True

    def _complete_fragment_sync(self, entry: _StreamingFragmentSchedule) -> None:
        fragment = entry.fragment
        logger.info(
            "Syncing fragment=%s step=%s manager_step=%s",
            fragment.fragment_id,
            self._inner_step,
            self._manager.current_step(),
        )
        fragment.perform_sync()
        entry.pending = False
        self._fragments_synced_this_step.add(fragment.fragment_id)
        entry.advance(self._sync_window)

    def _maybe_prepare_aligned_state_sync(self, fragment_idx: int) -> None:
        if not self._optimizer_state_sync_enabled:
            return
        if self._optimizer_state_schedule != "aligned":
            return
        commit_step = self._inner_step + self._fragment_sync_delay
        ready = self._resolve_aligned_state_candidates(
            fragment_idx,
            commit_step=commit_step,
        )
        if not ready:
            self._pending_aligned_state_frags.pop(fragment_idx, None)
            return
        entries: list[tuple[_StreamingOptimizerStateFragment, int]] = []
        for state_fragment in ready:
            state_fragment.set_step_context(self._inner_step)
            state_fragment.prepare_sync()
            entries.append((state_fragment, commit_step))
        self._pending_aligned_state_frags[fragment_idx] = entries

    def _drive_aligned_state_completion(self) -> None:
        if not self._optimizer_state_sync_enabled:
            return
        if self._optimizer_state_schedule != "aligned":
            return
        if not self._pending_aligned_state_frags:
            return
        current_step = self._inner_step
        for fragment_idx in list(self._pending_aligned_state_frags.keys()):
            entries = self._pending_aligned_state_frags.get(fragment_idx)
            if not entries:
                self._pending_aligned_state_frags.pop(fragment_idx, None)
                continue
            completed: list[tuple[_StreamingOptimizerStateFragment, int]] = []
            remaining: list[tuple[_StreamingOptimizerStateFragment, int]] = []
            for state_fragment, commit_step in entries:
                if current_step >= commit_step:
                    completed.append((state_fragment, commit_step))
                else:
                    remaining.append((state_fragment, commit_step))
            for state_fragment, _commit in completed:
                state_fragment.perform_sync()
                state_fragment.reset()
            if remaining:
                self._pending_aligned_state_frags[fragment_idx] = remaining
            else:
                self._pending_aligned_state_frags.pop(fragment_idx, None)

    def _resolve_aligned_state_candidates(
        self,
        fragment_idx: int,
        *,
        commit_step: int,
    ) -> list[_StreamingOptimizerStateFragment]:
        if not self._state_fragments_per_fragment:
            return []
        if fragment_idx >= len(self._state_fragments_per_fragment):
            return []
        states = self._state_fragments_per_fragment[fragment_idx]
        if not states:
            return []

        fragment = self._fragments[fragment_idx]
        offset = fragment.fragment_sync_offset
        if commit_step < offset:
            return []

        ready: list[_StreamingOptimizerStateFragment] = []
        for state_fragment in states:
            interval = max(1, state_fragment.sync_every)
            if (commit_step - offset) % interval == 0:
                ready.append(state_fragment)
        return ready

    def _step_pre_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._manager.disallow_state_dict_read()

    def _step_post_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._manager.allow_state_dict_read()
        if not self._is_opt_init:
            self._lazy_init_optimizer_fragments()
        self._inner_step += 1
        self._drive_fragment_schedule()

        if not self._fragments or not self._state_fragments_per_fragment:
            self._fragments_synced_this_step.clear()
            self._pending_aligned_state_frags.clear()
            return

        if self._optimizer_state_schedule == "aligned":
            self._drive_aligned_state_completion()
            self._fragments_synced_this_step.clear()
            return

        synced_fragments = tuple(self._fragments_synced_this_step)
        self._fragments_synced_this_step.clear()

        if not synced_fragments:
            self._drive_staggered_state_schedule()

    def _resolve_optimizer_sync_intervals(self, state_keys: Iterable[str]) -> list[int]:
        keys = list(state_keys)
        if not keys:
            return []

        spec = self._raw_optimizer_sync_config
        if spec is None:
            return [self._fragment_stride for _ in keys]
        if isinstance(spec, int):
            return self._expand_single_interval(spec, keys)
        if isinstance(spec, list):
            return self._expand_list_intervals(spec, keys)
        if isinstance(spec, dict):
            return self._expand_dict_intervals(spec, keys)

        msg = f"optimizer_sync_every must be an int, list, dict, or None; received {type(spec)!r}"
        raise TypeError(msg)

    def _expand_single_interval(self, interval: int, keys: list[str]) -> list[int]:
        self._validate_positive_interval(interval)
        return [interval for _ in keys]

    def _expand_list_intervals(self, intervals: list[int], keys: list[str]) -> list[int]:
        normalized = [int(value) for value in intervals]
        for value in normalized:
            self._validate_positive_interval(value)
        if len(normalized) == len(keys):
            return normalized

        broadcasted = _broadcast_moment_intervals(normalized, keys)
        if broadcasted is not None:
            return broadcasted

        msg = (
            "Length of optimizer_sync_every list does not match discovered optimizer states; "
            "provide one value per state or two values for [first_moment, second_moment]."
        )
        raise ValueError(msg)

    def _expand_dict_intervals(self, mapping: dict[str, int], keys: list[str]) -> list[int]:
        normalized = {str(k): int(v) for k, v in mapping.items()}
        resolved: list[int] = []
        missing: list[str] = []
        for key in keys:
            value = _resolve_interval_from_mapping(key, normalized)
            if value is None:
                missing.append(key)
                continue
            self._validate_positive_interval(value)
            resolved.append(value)

        if missing:
            missing_keys = ", ".join(sorted(missing))
            msg = f"Missing DES-LOC sync interval for optimizer state(s): {missing_keys}."
            raise ValueError(msg)
        return resolved

    def _validate_positive_interval(self, value: int) -> None:
        if value <= 0:
            msg = "optimizer_sync_every values must be positive"
            raise ValueError(msg)

    def _iter_optimizer_states(self) -> Iterable[dict[str, Any]]:
        """Yield state dicts for the wrapped optimizer, including composites."""
        yield from self._optimizer.state.values()

        inner_opts = getattr(self._optimizer, "optimizers", None)
        if isinstance(inner_opts, list):
            for idx, opt in enumerate(inner_opts):
                print(f"[DESLOC DEBUG] streaming: inspecting inner optimizer[{idx}] {type(opt).__name__} for state discovery")
                yield from opt.state.values()

    def _lazy_init_optimizer_fragments(self) -> None:
        if not self._optimizer_state_sync_enabled:
            self._state_fragments_per_fragment = [[] for _ in self._fragments]
            self._is_opt_init = True
            return
        state_sets: set[str] = set()
        for state in self._iter_optimizer_states():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    state_sets.add(str(key))

        state_keys = sorted(state_sets)
        print(f"[DESLOC DEBUG] streaming controller={self._name_prefix} discovered optimizer state keys: {state_keys}")
        sync_intervals = self._resolve_optimizer_sync_intervals(state_keys)

        if not state_keys and self._raw_optimizer_sync_config is not None:
            logger.warning(
                "DES-LOC optimizer_sync_every provided but no tensor states were discovered; skipping state synchronization."
            )

        if not state_keys:
            self._state_fragments_per_fragment = [[] for _ in self._fragments]
            self._is_opt_init = True
            return

        self._state_fragments_per_fragment = [[] for _ in self._fragments]
        for idx, key in enumerate(state_keys):
            sync_every = sync_intervals[idx]
            for fragment_idx, params in enumerate(self._partitions):
                fragment_config = StreamingOptimizerFragmentConfig(
                    manager=self._manager,
                    fragment_id=fragment_idx,
                    name_prefix=f"{self._name_prefix}_{key}_fragment_{fragment_idx}",
                    param_entries=params,
                    optimizer=self._optimizer,
                    state_key=key,
                    sync_every=sync_every,
                    backup_device=self._backup_device,
                    pin_memory=self._pin_memory,
                    use_bucketization=self._streaming_cfg.use_bucketization,
                    bucket_cap_mb=self._streaming_cfg.bucket_cap_mb,
                    should_quantize=self._streaming_cfg.should_quantize,
                )
                fragment = _StreamingOptimizerStateFragment(fragment_config)
                param_names = fragment.parameter_names
                logger.info(
                    "DES-LOC streaming optimizer state '%s' fragment=%s initialized with %d parameters: %s",
                    key,
                    fragment_idx,
                    len(param_names),
                    _format_fragment_membership(param_names),
                )
                print(
                    f"[DESLOC DEBUG] streaming controller={self._name_prefix} state_key={key} "
                    f"fragment={fragment_idx} sync_every={sync_every} params={param_names}"
                )
                fragment.register_state_dict_fn()
                self._state_fragments_per_fragment[fragment_idx].append(fragment)

        self._is_opt_init = True
        self._log_optimizer_state_fragment_assignments()

    def _sync_state_fragments(self, fragment_idx: int, *, limit_one: bool = False) -> None:
        if not self._optimizer_state_sync_enabled:
            return
        if not self._state_fragments_per_fragment:
            return
        if fragment_idx >= len(self._state_fragments_per_fragment):
            return

        candidates = self._state_fragments_per_fragment[fragment_idx]
        ready: list[_StreamingOptimizerStateFragment] = []
        for fragment in candidates:
            ready_flag = fragment.tick()
            if ready_flag:
                ready.append(fragment)
                if limit_one:
                    break

        if ready:
            print(
                f"[DESLOC DEBUG] streaming controller={self._name_prefix} fragment_idx={fragment_idx} "
                f"global_step={_get_global_step(self._manager)} "
                f"ready_fragments={[type(f).__name__ for f in ready]} "
                f"sync_every={[f.sync_every for f in ready]} "
                f"local_steps={[f._local_step for f in ready]}"
            )
        self._execute_state_sync_batch(ready)

    def _execute_state_sync_batch(self, fragments: list[_StreamingOptimizerStateFragment]) -> None:
        if not fragments:
            return
        try:
            self._manager.start_quorum(
                allow_heal=False,
                shrink_only=False,
                timeout=self._quorum_timeout,
            )
        except TimeoutError as err:
            logger.warning(
                "DES-LOC optimizer state quorum timed out after %.1f seconds; skipping synchronization.",
                self._quorum_timeout.total_seconds(),
            )
            self._manager.report_error(err)
            for fragment in fragments:
                fragment.restore_state()
                fragment.reset()
            return

        for fragment in fragments:
            fragment.set_step_context(self._inner_step)
            fragment.prepare_sync()
        for fragment in fragments:
            fragment.perform_sync()
            fragment.reset()

    def _drive_staggered_state_schedule(self) -> None:
        if not self._optimizer_state_sync_enabled:
            return
        if not self._state_fragments_per_fragment or not self._fragments:
            return
        fragment_idx = self._state_cursor
        self._sync_state_fragments(fragment_idx, limit_one=True)
        self._state_cursor = (self._state_cursor + 1) % len(self._fragments)

    def _log_optimizer_state_fragment_assignments(self) -> None:
        if self._optimizer_state_log_emitted:
            return

        if not any(self._state_fragments_per_fragment):
            logger.info("DES-LOC streaming optimizer state fragments: none discovered.")
            self._optimizer_state_log_emitted = True
            return

        per_state: dict[str, dict[str, int]] = defaultdict(dict)
        for fragments in self._state_fragments_per_fragment:
            for fragment in fragments:
                state_map = per_state[fragment.state_key]
                for name in fragment.parameter_names:
                    key = _component_key_from_name(name)
                    existing = state_map.get(key)
                    if existing is not None and existing != fragment.fragment_id:
                        logger.warning(
                            "DES-LOC streaming optimizer state '%s' component %s mapped to multiple fragments (%s, %s).",
                            fragment.state_key,
                            key,
                            existing,
                            fragment.fragment_id,
                        )
                    state_map[key] = fragment.fragment_id

        for state_key, mapping in sorted(per_state.items()):
            formatted = "; ".join(
                f"{component}->frag{fragment_id}"
                for component, fragment_id in sorted(mapping.items())
            )
            logger.info(
                "DES-LOC streaming optimizer state '%s' fragments: %s",
                state_key,
                formatted or "none",
            )
        self._optimizer_state_log_emitted = True
class DesLocFTOptimizersContainer(FTOptimizersContainer):
    """FT optimizer container augmented with DES-LOC synchronization."""

    def __init__(self, config: DesLocFTOptimizersConfig) -> None:
        desloc_config = config.desloc_config
        if desloc_config.param_sync_every <= 0:
            msg = "desloc.param_sync_every must be a positive integer."
            raise ValueError(msg)

        streaming_cfg = config.streaming or desloc_config.resolved_streaming()

        super().__init__(
            config.model_parts,
            config.optimizer_cls,
            config.optimizer_kwargs,
            config.ft_manager,
            use_ft_optimizer=config.use_ft_optimizer,
            param_groups=config.param_groups,
        )

        backup_device = desloc_config.resolved_backup_device()
        optimizer_sync = desloc_config.normalized_optimizer_sync()
        outer_optimizer_spec = config.outer_optimizer or desloc_config.normalized_outer_optimizer()

        self._desloc_controllers: list[DesLocController | StreamingDesLocController] = []
        for idx, (model, optimizer) in enumerate(zip(self.model_parts, self.optimizers, strict=True)):
            controller_config = DesLocControllerConfig(
                manager=config.ft_manager,
                model=model,
                optimizer=optimizer,
                param_sync_every=desloc_config.param_sync_every,
                optimizer_sync_every=optimizer_sync,
                backup_device=backup_device,
                pin_memory=desloc_config.pin_memory,
                name_prefix=f"desloc_{idx}",
                quorum_timeout_seconds=desloc_config.quorum_timeout_seconds,
                outer_optimizer=outer_optimizer_spec,
                log_outer_metrics=desloc_config.log_outer_metrics,
                metrics_logger=None,
                checkpoint_outer_optimizer=desloc_config.checkpoint_outer_optimizer,
                disable_optimizer_state_sync=desloc_config.disable_optimizer_state_sync,
            )
            if streaming_cfg is not None:
                controller = StreamingDesLocController(controller_config, streaming_cfg)
            else:
                controller = DesLocController(controller_config)
            self._desloc_controllers.append(controller)

    def close_desloc(self) -> None:
        """Detach any registered DES-LOC hooks from the wrapped optimizers."""
        for controller in self._desloc_controllers:
            controller.close()
        self._desloc_controllers.clear()

    def set_desloc_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        for controller in self._desloc_controllers:
            controller.set_metrics_logger(logger_fn)


@contextmanager
def desloc_semi_sync_context(_ft_manager: FTManager, optimizer: torch.optim.Optimizer) -> Iterator[None]:
    """Context manager wiring DES-LOC into TorchFT semi-sync execution."""
    try:
        yield
    finally:
        close_hook = getattr(optimizer, "close_desloc", None)
        if callable(close_hook):
            close_hook()


_MODULE_PROXY.__dict__.update(globals())
