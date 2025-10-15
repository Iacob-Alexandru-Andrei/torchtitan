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
from dataclasses import dataclass
from datetime import timedelta
from types import ModuleType
from typing import Any, TYPE_CHECKING

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
    from torchtitan.experiments.fl.configs.optimizers import DesLocConfig

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

        self._init_backup_storage()
        self.save_state()

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
                self._original_parameters[name].copy_(
                    _extract_local_tensor(param.data), non_blocking=True
                )

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
        self._param_fragment = _ParameterFragment(param_fragment_cfg)
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
