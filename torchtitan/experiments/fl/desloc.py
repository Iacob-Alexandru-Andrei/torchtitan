# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DES-LOC integration utilities for the FL experiments."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from types import ModuleType
from typing import Any, TYPE_CHECKING
import math

import torch
from torch import nn

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
    from collections.abc import Iterable, Iterator

    from torch.optim import Optimizer

    from torchtitan.components.ft.manager import FTManager
    from torchtitan.experiments.fl.configs.optimizers import (
        DesLocConfig,
        DesLocOuterOptimizerConfig,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParameterFragmentConfig:
    """Configuration for synchronizing model parameters via DES-LOC."""

    manager: Any
    model: nn.Module
    sync_every: int
    backup_device: torch.device | None
    pin_memory: bool
    name_prefix: str


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
    outer_optimizer: "DesLocOuterOptimizerConfig | None" = None


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
    outer_optimizer: "DesLocOuterOptimizerConfig | None" = None


def _extract_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached clone of ``tensor`` on its local device."""
    local = (
        tensor.to_local()
        if DTensor is not None and isinstance(tensor, DTensor)
        else tensor
    )
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

        self._original_parameters: dict[str, torch.Tensor] = {}
        self._averaged_parameters: list[torch.Tensor] = []
        self._await_snapshot_completion = False
        self._snapshot_events: dict[str, torch.cuda.Event | None] = {}

        self._init_backup_storage()
        self.save_state()

    @property
    def name_prefix(self) -> str:
        return self._name_prefix

    @property
    def backup_device(self) -> torch.device | None:
        return self._backup_device

    def _init_backup_storage(self) -> None:
        for name, param in self._model.named_parameters():
            local_tensor = _extract_local_tensor(param.data)
            device = (
                self._backup_device
                if self._backup_device is not None
                else local_tensor.device
            )
            backup = torch.empty_like(local_tensor, device=device)
            if (
                self._pin_memory
                and backup.device.type == "cpu"
                and torch.cuda.is_available()
            ):
                backup = backup.pin_memory()
            self._original_parameters[name] = backup

    def save_state(self) -> None:
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                local_tensor = _extract_local_tensor(param.data)
                async_copy = (
                    self._pin_memory
                    and torch.cuda.is_available()
                    and isinstance(local_tensor, torch.Tensor)
                    and local_tensor.device.type == "cuda"
                    and not self._await_snapshot_completion
                )
                self._original_parameters[name].copy_(
                    local_tensor, non_blocking=async_copy
                )
                self._record_snapshot_event(name, param, async_copy)

    def _record_snapshot_event(
        self, name: str, param: torch.Tensor, async_copy: bool
    ) -> None:
        if not self._await_snapshot_completion or not async_copy:
            self._snapshot_events.pop(name, None)
            return
        if param.device.type != "cuda" or not torch.cuda.is_available():
            self._snapshot_events.pop(name, None)
            return
        event = torch.cuda.Event(device=param.device)
        event.record(torch.cuda.current_stream(param.device))
        self._snapshot_events[name] = event

    def _wait_for_snapshot_completion(self, name: str) -> None:
        if not self._await_snapshot_completion:
            return
        event = self._snapshot_events.get(name)
        if event is not None:
            event.synchronize()
            self._snapshot_events.pop(name, None)

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                _copy_into_tensor(param.data, self._original_parameters[name])

    def prepare_sync(self) -> list[Any]:
        self._averaged_parameters.clear()
        work_items: list[Any] = []
        for param in self._model.parameters():
            avg_param = _extract_local_tensor(param.data)
            work_items.append(self._manager.allreduce(avg_param))
            self._averaged_parameters.append(avg_param)
        return work_items

    def perform_sync(self) -> None:
        with torch.no_grad():
            for param, avg_param in zip(
                self._model.parameters(), self._averaged_parameters, strict=True
            ):
                _copy_into_tensor(param.data, avg_param)

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            for name, tensor in state_dict.items():
                if name in self._original_parameters:
                    self._original_parameters[name].copy_(tensor)

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_parameters

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_params",
            load_fn,
            save_fn,
        )


class _OuterOptimizingParameterFragment(_ParameterFragment):
    """Parameter synchronizer that applies an outer optimizer to averaged pseudo-gradients."""

    def __init__(
        self,
        config: ParameterFragmentConfig,
        outer_spec: "DesLocOuterOptimizerConfig",
    ) -> None:
        self._outer_spec = outer_spec
        if config.pin_memory:
            config = replace(config, pin_memory=False)
        super().__init__(config)
        optimizer_cls = self._outer_spec.resolve_optimizer_cls()
        params = [p for p in self._model.parameters() if p.requires_grad]
        if not params:
            msg = "DES-LOC outer optimizer requires at least one trainable parameter."
            raise ValueError(msg)
        self._outer_optimizer = optimizer_cls(params, **self._outer_spec.kwargs)
        self._sync_entries: list[tuple[str, nn.Parameter, torch.Tensor]] = []
        self._await_snapshot_completion = True
        # Re-capture state with snapshot completion tracking enabled
        self.save_state()
        self._sync_stats: dict[str, float] = {}
        self._reference_synchronized = False
        self._pending_reference_sync: list[tuple[Any, str, torch.Tensor]] | None = None
        self._pending_reference_sync: list[tuple[Any, str, torch.Tensor]] | None = None

    @property
    def outer_optimizer(self) -> Optimizer:
        return self._outer_optimizer

    def register_state_dict_fn(self) -> None:
        super().register_state_dict_fn()

        def load_outer(state_dict: dict[str, Any]) -> None:
            self._outer_optimizer.load_state_dict(state_dict)

        def save_outer() -> dict[str, Any]:
            return self._outer_optimizer.state_dict()

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_outer_optimizer",
            load_outer,
            save_outer,
        )

    def _synchronize_reference_buffers(self) -> None:
        """Ensure all replicas agree on the captured reference parameters."""
        # Reference synchronization is performed lazily during the first sync.
        return

    def prepare_sync(self) -> list[Any]:
        self._sync_entries.clear()
        work_items: list[Any] = []

        if not self._reference_synchronized:
            self._pending_reference_sync = []
            for name, param in self._model.named_parameters():
                if not param.requires_grad:
                    continue
                self._wait_for_snapshot_completion(name)
                reference = self._original_parameters[name].to(
                    device=param.device,
                    dtype=param.dtype,
                    copy=True,
                )
                work = self._manager.allreduce(reference)
                self._pending_reference_sync.append((work, name, reference))
                work_items.append(work)
        else:
            self._pending_reference_sync = None

        work_items.extend(super().prepare_sync())
        return work_items

    def perform_sync(self) -> None:
        ref_norm_sq = 0.0
        local_norm_sq = 0.0
        pseudo_norm_sq = 0.0

        with torch.no_grad():
            if self._pending_reference_sync is not None:
                for _work, name, reduced in self._pending_reference_sync:
                    self._original_parameters[name].copy_(
                        reduced.to(self._original_parameters[name].device)
                    )
                self._pending_reference_sync = None
                self._reference_synchronized = True

            super(_OuterOptimizingParameterFragment, self).restore_state()

            averaged_iter = iter(self._averaged_parameters)
            self._sync_entries.clear()
            for name, param in self._model.named_parameters():
                avg_param = next(averaged_iter)
                avg_param = avg_param.to(device=param.device, dtype=param.dtype)
                reference = self._original_parameters[name].to(
                    device=avg_param.device, dtype=avg_param.dtype
                )
                grad = reference - avg_param
                ref_norm_sq += reference.pow(2).sum().item()
                local_norm_sq += avg_param.pow(2).sum().item()
                pseudo_norm_sq += grad.pow(2).sum().item()
                if param.requires_grad:
                    param.grad = grad
                    self._sync_entries.append((name, param, grad))

        self._outer_optimizer.step()
        self._outer_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            for _name, param, grad in self._sync_entries:
                param.grad = grad

        post_norm_sq = sum(
            grad.pow(2).sum().item() for _name, _param, grad in self._sync_entries
        )

        self._sync_stats = {
            "ref_pre": math.sqrt(ref_norm_sq),
            "local_pre": math.sqrt(local_norm_sq),
            "pseudo_pre": math.sqrt(pseudo_norm_sq),
        }

        logger.debug(
            (
                "DES-LOC outer sync stats: ref_norm(pre)=%.6f local_norm(pre)=%.6f "
                "pseudo_norm(pre)=%.6f pseudo_norm(post)=%.6f"
            ),
            self._sync_stats["ref_pre"],
            self._sync_stats["local_pre"],
            self._sync_stats["pseudo_pre"],
            math.sqrt(post_norm_sq),
        )

        self._sync_entries.clear()

    def restore_state(self) -> None:
        super().restore_state()
        self._outer_optimizer.zero_grad(set_to_none=True)
        self._sync_entries.clear()


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
                device = (
                    self._backup_device
                    if self._backup_device is not None
                    else tensor.device
                )
                self._original_state_tensors[name] = torch.empty_like(
                    tensor, device=device
                )

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
                if (
                    param in self._optimizer.state
                    and self.state_key in self._optimizer.state[param]
                ):
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
        )
        if config.outer_optimizer is not None:
            self._param_fragment = _OuterOptimizingParameterFragment(
                param_fragment_cfg, config.outer_optimizer
            )
        else:
            self._param_fragment = _ParameterFragment(param_fragment_cfg)
        self._param_fragment.register_state_dict_fn()

        self._fragments: list[_BaseFragment] = [self._param_fragment]
        self._allreduce_work: list[Any] = []
        self._is_opt_init = False
        self._initial_sync_done = False
        self._outer_optimizer_state_keys: set[str] = set()

        def _load_initial_flag(state: dict[str, Any]) -> None:
            self._initial_sync_done = bool(state.get("done", False))

        def _save_initial_flag() -> dict[str, Any]:
            return {"done": int(self._initial_sync_done)}

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_initial_sync",
            _load_initial_flag,
            _save_initial_flag,
        )

        self._hook = config.optimizer.register_step_post_hook(self._step_post_hook)

    def close(self) -> None:
        """Detach the registered optimizer step hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

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

        msg = (
            "optimizer_sync_every must be an int, list, dict, or None; "
            f"received {type(spec)!r}"
        )
        raise TypeError(msg)

    def _expand_single_interval(self, interval: int, keys: list[str]) -> list[int]:
        self._validate_positive_interval(interval)
        return [interval for _ in keys]

    def _expand_list_intervals(
        self, intervals: list[int], keys: list[str]
    ) -> list[int]:
        if len(intervals) != len(keys):
            msg = "Length of optimizer_sync_every list does not match discovered optimizer states."
            raise ValueError(msg)
        normalized = [int(value) for value in intervals]
        for value in normalized:
            self._validate_positive_interval(value)
        return normalized

    def _expand_dict_intervals(
        self, mapping: dict[str, int], keys: list[str]
    ) -> list[int]:
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
        self._maybe_init_outer_optimizer_fragments()

    def _maybe_init_outer_optimizer_fragments(self) -> None:
        if not isinstance(self._param_fragment, _OuterOptimizingParameterFragment):
            return

        outer_optimizer = self._param_fragment.outer_optimizer
        discovered_keys: set[str] = set()
        for state in outer_optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    discovered_keys.add(str(key))

        new_keys = sorted(discovered_keys - self._outer_optimizer_state_keys)
        if not new_keys:
            return

        for key in new_keys:
            fragment_config = OptimizerFragmentConfig(
                manager=self._manager,
                model=self._model,
                optimizer=outer_optimizer,
                state_key=key,
                sync_every=self._param_fragment.sync_every,
                backup_device=self._backup_device,
                name_prefix=f"{self._param_fragment.name_prefix}_outer_{key}",
            )
            fragment = _OptimizerStateFragment(fragment_config)
            fragment.register_state_dict_fn()
            self._fragments.append(fragment)
            self._outer_optimizer_state_keys.add(key)

    def _step_post_hook(
        self,
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        if not self._is_opt_init:
            self._lazy_init_optimizer_fragments()
        else:
            self._maybe_init_outer_optimizer_fragments()

        if not self._initial_sync_done:
            ready_fragments = list(self._fragments)
            self._initial_sync_done = True
        else:
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


class DesLocFTOptimizersContainer(FTOptimizersContainer):
    """FT optimizer container augmented with DES-LOC synchronization."""

    def __init__(self, config: DesLocFTOptimizersConfig) -> None:
        desloc_config = config.desloc_config
        if desloc_config.param_sync_every <= 0:
            msg = "desloc.param_sync_every must be a positive integer."
            raise ValueError(msg)

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
        outer_optimizer_spec = (
            config.outer_optimizer or desloc_config.normalized_outer_optimizer()
        )

        self._desloc_controllers: list[DesLocController] = []
        for idx, (model, optimizer) in enumerate(
            zip(self.model_parts, self.optimizers, strict=True)
        ):
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
            )
            controller = DesLocController(controller_config)
            self._desloc_controllers.append(controller)

    def close_desloc(self) -> None:
        """Detach any registered DES-LOC hooks from the wrapped optimizers."""
        for controller in self._desloc_controllers:
            controller.close()
        self._desloc_controllers.clear()


@contextmanager
def desloc_semi_sync_context(
    _ft_manager: FTManager, optimizer: torch.optim.Optimizer
) -> Iterator[None]:
    """Context manager wiring DES-LOC into TorchFT semi-sync execution."""
    try:
        yield
    finally:
        close_hook = getattr(optimizer, "close_desloc", None)
        if callable(close_hook):
            close_hook()


_MODULE_PROXY.__dict__.update(globals())
