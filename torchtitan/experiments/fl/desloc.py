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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from types import ModuleType
from typing import Any, TYPE_CHECKING

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
    optimizer: Optimizer
    state_key: str
    sync_every: int
    backup_device: torch.device | None
    name_prefix: str


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
    outer_optimizer: DesLocOuterOptimizerConfig | Optimizer | None = None
    log_outer_metrics: bool = False
    metrics_logger: Callable[[dict[str, float]], None] | None = None
    checkpoint_outer_optimizer: bool = True


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


def _partition_named_parameters(
    model: nn.Module,
    fragments: int,
) -> list[list[tuple[str, nn.Parameter]]]:
    """Partition model parameters into ``fragments`` balanced buckets."""
    if fragments <= 0:
        msg = "desloc.streaming.fragments must be a positive integer."
        raise ValueError(msg)

    named_params = list(model.named_parameters())
    if not named_params:
        return []

    fragments = min(max(1, fragments), len(named_params))
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


class _BaseFragment:
    def __init__(self, sync_every: int) -> None:
        if sync_every <= 0:
            message = "sync_every must be a positive integer"
            raise ValueError(message)
        self.sync_every = sync_every
        self._local_step = 0

    def tick(self) -> bool:
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
        self._backup_device = config.backup_device
        self._pin_memory = config.pin_memory
        self._name_prefix = config.name_prefix

        self._param_map = dict(self._model.named_parameters())
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

        self._init_backup_storage()
        self.save_state()
        if self._outer_optimizer is not None:
            self._reference_synced = True

    def set_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        self._metrics_logger = logger_fn

    def _init_backup_storage(self) -> None:
        for name, param in self._model.named_parameters():
            local_tensor = _extract_local_tensor(param.data)
            device = self._backup_device if self._backup_device is not None else local_tensor.device
            backup = torch.empty_like(local_tensor, device=device)
            if self._pin_memory and backup.device.type == "cpu" and torch.cuda.is_available():
                backup = backup.pin_memory()
            self._original_parameters[name] = backup

    def save_state(self) -> None:
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                self._original_parameters[name].copy_(_extract_local_tensor(param.data), non_blocking=True)

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                _copy_into_tensor(param.data, self._original_parameters[name])

    def prepare_sync(self) -> list[Any]:
        if self._outer_optimizer is not None and not self._reference_synced:
            # Ensure backups reflect the current model weights (e.g. after checkpoint load).
            self.save_state()
        self._averaged_parameters.clear()
        work_items: list[Any] = []
        for name, param in self._model.named_parameters():
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
        self._optimizer = config.optimizer
        self.state_key = config.state_key
        self._backup_device = config.backup_device
        self._name_prefix = config.name_prefix

        self._param_map = dict(self._model.named_parameters())
        self._original_state_tensors: dict[str, torch.Tensor] = {}
        self._averaged_state_tensors: list[torch.Tensor] = []

        self._init_backup_storage()
        self.save_state()

    def _init_backup_storage(self) -> None:
        for name, param in self._model.named_parameters():
            state = self._optimizer.state.get(param, {})
            tensor = state.get(self.state_key)
            if isinstance(tensor, torch.Tensor):
                device = self._backup_device if self._backup_device is not None else tensor.device
                self._original_state_tensors[name] = torch.empty_like(tensor, device=device)

    def save_state(self) -> None:
        with torch.no_grad():
            for name, backup in self._original_state_tensors.items():
                param = self._param_map[name]
                tensor = self._optimizer.state[param][self.state_key]
                backup.copy_(tensor, non_blocking=True)

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, backup in self._original_state_tensors.items():
                param = self._param_map[name]
                if param in self._optimizer.state and self.state_key in self._optimizer.state[param]:
                    self._optimizer.state[param][self.state_key].copy_(backup)

    def prepare_sync(self) -> list[Any]:
        self._averaged_state_tensors.clear()
        work_items: list[Any] = []
        for name in self._original_state_tensors:
            param = self._param_map[name]
            state_tensor = self._optimizer.state[param][self.state_key]
            avg_state = state_tensor.detach().clone()
            work_items.append(self._manager.allreduce(avg_state))
            self._averaged_state_tensors.append(avg_state)
        return work_items

    def perform_sync(self) -> None:
        with torch.no_grad():
            for name, averaged in zip(
                self._original_state_tensors.keys(),
                self._averaged_state_tensors,
                strict=True,
            ):
                param = self._param_map[name]
                self._optimizer.state[param][self.state_key].copy_(averaged)

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
        fragment_update_alpha: float,
        use_bucketization: bool,
        bucket_cap_mb: float | None,
        should_quantize: bool,
        log_outer_metrics: bool,
        metrics_logger: Callable[[dict[str, float]], None] | None,
    ) -> None:
        self._manager = manager
        self._fragment_id = fragment_id
        self._name_prefix = name_prefix
        self._param_entries = param_entries
        self._param_map = {name: param for name, param in param_entries}
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        self._outer_optimizer = outer_optimizer
        self._fragment_update_alpha = fragment_update_alpha
        self._log_outer_metrics = log_outer_metrics
        self._metrics_logger = metrics_logger
        self._averaging_only = outer_optimizer is None
        self._should_quantize = should_quantize

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
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            if state_dict:
                for name, tensor in state_dict.items():
                    if name in self.original_parameters:
                        self.original_parameters[name].copy_(tensor)
            else:
                self.save_parameters()

        def save_fn() -> dict[str, torch.Tensor]:
            return {
                name: _extract_local_tensor(param)
                for name, param in self.original_parameters.items()
            }

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

        return should_commit


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

        param_fragment_cfg = ParameterFragmentConfig(
            manager=config.manager,
            model=config.model,
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
        self._is_opt_init = False

        self._hook = config.optimizer.register_step_post_hook(self._step_post_hook)

    def close(self) -> None:
        """Detach the registered optimizer step hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def set_metrics_logger(self, logger_fn: Callable[[dict[str, float]], None] | None) -> None:
        self._param_fragment.set_metrics_logger(logger_fn)

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
        if len(intervals) != len(keys):
            msg = "Length of optimizer_sync_every list does not match discovered optimizer states."
            raise ValueError(msg)
        normalized = [int(value) for value in intervals]
        for value in normalized:
            self._validate_positive_interval(value)
        return normalized

    def _expand_dict_intervals(self, mapping: dict[str, int], keys: list[str]) -> list[int]:
        resolved: list[int] = []
        for key in keys:
            if key not in mapping:
                msg = f"Missing DES-LOC sync interval for optimizer state '{key}'."
                raise ValueError(msg)
            value = int(mapping[key])
            self._validate_positive_interval(value)
            resolved.append(value)
        return resolved

    def _validate_positive_interval(self, value: int) -> None:
        if value <= 0:
            msg = "optimizer_sync_every values must be positive"
            raise ValueError(msg)

    def _lazy_init_optimizer_fragments(self) -> None:
        state_sets = set()
        for state in self._optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    state_sets.add(str(key))

        state_keys = sorted(state_sets)
        sync_intervals = self._resolve_optimizer_sync_intervals(state_keys)

        if not state_keys and self._raw_optimizer_sync_config is not None:
            logger.warning(
                "DES-LOC optimizer_sync_every provided but no tensor states were discovered; skipping state synchronization."
            )

        for idx, key in enumerate(state_keys):
            fragment_config = OptimizerFragmentConfig(
                manager=self._manager,
                model=self._model,
                optimizer=self._optimizer,
                state_key=key,
                sync_every=sync_intervals[idx],
                backup_device=self._backup_device,
                name_prefix=f"{self._name_prefix}_{key}",
            )
            fragment = _OptimizerStateFragment(fragment_config)
            fragment.register_state_dict_fn()
            self._fragments.append(fragment)

        self._is_opt_init = True

    def _step_post_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        if not self._is_opt_init:
            self._lazy_init_optimizer_fragments()

        ready_fragments = [fragment for fragment in self._fragments if fragment.tick()]

        if ready_fragments:
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

        partitions = _partition_named_parameters(self._model, streaming.fragments)
        if not partitions:
            msg = "DES-LOC streaming requires at least one model parameter."
            raise ValueError(msg)

        num_fragments = len(partitions)
        if config.param_sync_every < num_fragments:
            msg = "desloc.param_sync_every must be >= the number of streaming fragments."
            raise ValueError(msg)
        if config.param_sync_every % num_fragments != 0:
            msg = "desloc.param_sync_every must be divisible by the number of streaming fragments."
            raise ValueError(msg)

        self._sync_every = config.param_sync_every // num_fragments
        if streaming.sync_delay >= self._sync_every:
            msg = "desloc.streaming.sync_delay must be smaller than param_sync_every / fragments."
            raise ValueError(msg)
        if not (0.0 <= streaming.update_alpha <= 1.0):
            msg = "desloc.streaming.update_alpha must be between 0 and 1."
            raise ValueError(msg)

        outer_handles = self._build_outer_optimizer_handles(config.outer_optimizer, partitions)

        self._fragments: list[_StreamingParameterFragment] = []
        for idx, params in enumerate(partitions):
            fragment = _StreamingParameterFragment(
                manager=self._manager,
                fragment_id=idx,
                name_prefix=f"{self._name_prefix}_fragment_{idx}",
                param_entries=params,
                backup_device=self._backup_device,
                pin_memory=self._pin_memory,
                outer_optimizer=outer_handles[idx],
                fragment_update_alpha=streaming.update_alpha,
                use_bucketization=streaming.use_bucketization,
                bucket_cap_mb=streaming.bucket_cap_mb,
                should_quantize=streaming.should_quantize,
                log_outer_metrics=self._log_outer_metrics,
                metrics_logger=self._metrics_logger,
            )
            self._fragments.append(fragment)

        self._fragment_sync_delay = streaming.sync_delay
        self._hooks: list[RemovableHandle] = []
        self._hooks.append(self._optimizer.register_step_pre_hook(self._step_pre_hook))
        self._hooks.append(self._optimizer.register_step_post_hook(self._step_post_hook))

        self._local_step = 0

        self._state_fragments: list[_BaseFragment] = []
        self._state_allreduce_work: list[Any] = []
        self._is_opt_init = False

        self._register_state_dict_functions()
        self._register_outer_optimizer_state(outer_handles)

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

    def _register_outer_optimizer_state(self, outer_handles: list[Optimizer | None]) -> None:
        if not self._checkpoint_outer_optimizer:
            return
        seen: set[int] = set()
        for idx, optimizer in enumerate(outer_handles):
            if optimizer is None:
                continue
            ident = id(optimizer)
            if ident in seen:
                continue
            seen.add(ident)

            def load_fn(state_dict: dict[str, Any], opt=optimizer) -> None:
                opt.load_state_dict(state_dict)

            def save_fn(opt=optimizer) -> dict[str, Any]:
                return opt.state_dict()

            self._manager.register_state_dict_fn(
                f"{self._name_prefix}_stream_outer_{idx}",
                load_fn,
                save_fn,
            )

    def _step_pre_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._manager.disallow_state_dict_read()

    def _current_fragment(self) -> int:
        step = self._manager.current_step()
        return step % len(self._fragments)

    def _step_post_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        self._manager.allow_state_dict_read()
        if not self._is_opt_init:
            self._lazy_init_optimizer_fragments()

        self._local_step += 1

        if self._local_step == self._sync_every - self._fragment_sync_delay:
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
                fragment = self._fragments[self._current_fragment()]
                fragment.restore_parameters()
                self._local_step = 0
                self._sync_state_fragments([])
                return

            fragment = self._fragments[self._current_fragment()]
            fragment.prepare_sync()

        if self._local_step < self._sync_every:
            self._sync_state_fragments([])
            return

        fragment = self._fragments[self._current_fragment()]
        fragment.perform_sync()
        self._local_step = 0

        self._sync_state_fragments([])

    def _resolve_optimizer_sync_intervals(self, state_keys: Iterable[str]) -> list[int]:
        keys = list(state_keys)
        if not keys:
            return []

        spec = self._raw_optimizer_sync_config
        if spec is None:
            return [self._sync_every for _ in keys]
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
        if len(intervals) != len(keys):
            msg = "Length of optimizer_sync_every list does not match discovered optimizer states."
            raise ValueError(msg)
        normalized = [int(value) for value in intervals]
        for value in normalized:
            self._validate_positive_interval(value)
        return normalized

    def _expand_dict_intervals(self, mapping: dict[str, int], keys: list[str]) -> list[int]:
        resolved: list[int] = []
        for key in keys:
            if key not in mapping:
                msg = f"Missing DES-LOC sync interval for optimizer state '{key}'."
                raise ValueError(msg)
            value = int(mapping[key])
            self._validate_positive_interval(value)
            resolved.append(value)
        return resolved

    def _validate_positive_interval(self, value: int) -> None:
        if value <= 0:
            msg = "optimizer_sync_every values must be positive"
            raise ValueError(msg)

    def _lazy_init_optimizer_fragments(self) -> None:
        state_sets = set()
        for state in self._optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    state_sets.add(str(key))

        state_keys = sorted(state_sets)
        sync_intervals = self._resolve_optimizer_sync_intervals(state_keys)

        if not state_keys and self._raw_optimizer_sync_config is not None:
            logger.warning(
                "DES-LOC optimizer_sync_every provided but no tensor states were discovered; skipping state synchronization."
            )

        for idx, key in enumerate(state_keys):
            fragment_config = OptimizerFragmentConfig(
                manager=self._manager,
                model=self._model,
                optimizer=self._optimizer,
                state_key=key,
                sync_every=sync_intervals[idx],
                backup_device=self._backup_device,
                name_prefix=f"{self._name_prefix}_{key}",
            )
            fragment = _OptimizerStateFragment(fragment_config)
            fragment.register_state_dict_fn()
            self._state_fragments.append(fragment)

        self._is_opt_init = True

    def _sync_state_fragments(self, fragments: list[_BaseFragment]) -> None:
        ready = fragments or [fragment for fragment in self._state_fragments if fragment.tick()]
        if not ready:
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
            for fragment in ready:
                fragment.restore_state()
                fragment.reset()
            return

        self._prepare_state_sync(ready)
        self._perform_state_sync(ready)
        for fragment in ready:
            fragment.reset()

    def _prepare_state_sync(self, fragments: list[_BaseFragment]) -> None:
        self._state_allreduce_work.clear()
        for fragment in fragments:
            self._state_allreduce_work.extend(fragment.prepare_sync())

    def _perform_state_sync(self, fragments: list[_BaseFragment]) -> None:
        for work in self._state_allreduce_work:
            work.wait()

        commit_allowed = self._manager.should_commit()

        if commit_allowed:
            for fragment in fragments:
                fragment.perform_sync()
                fragment.save_state()
        else:
            for fragment in fragments:
                fragment.restore_state()
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
