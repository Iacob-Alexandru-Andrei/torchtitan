# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""DES-LOC integration utilities for the FL experiments."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import torch
from torch import nn

try:  # pragma: no cover - optional dependency in some environments
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - DTensor is optional
    DTensor = None  # type: ignore[assignment]

from torchtitan.components.optimizer import FTOptimizersContainer
from torchtitan.experiments.fl.configs.optimizers import DesLocConfig

if TYPE_CHECKING:
    from collections.abc import Iterable
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


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
                stride=param.stride(),
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

    def __init__(
        self,
        manager: Any,
        model: nn.Module,
        sync_every: int,
        backup_device: torch.device | None,
        pin_memory: bool,
        name_prefix: str,
    ) -> None:  # noqa: PLR0913
        super().__init__(sync_every)
        self._manager = manager
        self._model = model
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        self._name_prefix = name_prefix

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

    def __init__(
        self,
        manager: Any,
        model: nn.Module,
        optimizer: Optimizer,
        state_key: str,
        sync_every: int,
        backup_device: torch.device | None,
        name_prefix: str,
    ) -> None:  # noqa: PLR0913
        super().__init__(sync_every)
        self._manager = manager
        self._model = model
        self._optimizer = optimizer
        self.state_key = state_key
        self._backup_device = backup_device
        self._name_prefix = name_prefix

        self._param_map = {name: p for name, p in self._model.named_parameters()}
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
                strict=False,
            ):
                param = self._param_map[name]
                self._optimizer.state[param][self.state_key].copy_(averaged)

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            self._original_state_tensors = state_dict

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_state_tensors

        self._manager.register_state_dict_fn(
            f"{self._name_prefix}_state_{self.state_key}",
            load_fn,
            save_fn,
        )


class DesLocController:
    """Attach DES-LOC synchronization hooks to a PyTorch optimizer."""

    def __init__(
        self,
        *,
        manager: Any,
        model: nn.Module,
        optimizer: Optimizer,
        param_sync_every: int,
        optimizer_sync_every: int | list[int] | dict[str, int] | None,
        backup_device: torch.device | None,
        pin_memory: bool,
        name_prefix: str,
    ) -> None:  # noqa: PLR0913
        if getattr(manager, "_use_async_quorum", False):
            msg = "DES-LOC requires synchronous TorchFT quorum management."
            raise ValueError(msg)

        self._manager = manager
        self._model = model
        self._optimizer = optimizer
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        self._name_prefix = name_prefix
        self._raw_optimizer_sync_config = optimizer_sync_every

        self._param_fragment = _ParameterFragment(
            manager,
            model,
            param_sync_every,
            backup_device,
            pin_memory,
            name_prefix,
        )
        self._param_fragment.register_state_dict_fn()

        self._fragments: list[_BaseFragment] = [self._param_fragment]
        self._allreduce_work: list[Any] = []
        self._is_opt_init = False

        self._hook = optimizer.register_step_post_hook(self._step_post_hook)

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _resolve_optimizer_sync_intervals(self, state_keys: "Iterable[str]") -> list[int]:
        spec = self._raw_optimizer_sync_config
        keys = list(state_keys)
        if not keys:
            return []

        if spec is None:
            return [self._param_fragment.sync_every for _ in keys]
        if isinstance(spec, int):
            if spec <= 0:
                msg = "optimizer_sync_every values must be positive"
                raise ValueError(msg)
            return [spec for _ in keys]
        if isinstance(spec, list):
            if len(spec) != len(keys):
                msg = (
                    "Length of optimizer_sync_every list does not match discovered optimizer states."
                )
                raise ValueError(msg)
            intervals = [int(v) for v in spec]
            if any(v <= 0 for v in intervals):
                msg = "optimizer_sync_every values must be positive"
                raise ValueError(msg)
            return intervals
        if isinstance(spec, dict):
            intervals = []
            for key in keys:
                if key not in spec:
                    msg = f"Missing DES-LOC sync interval for optimizer state '{key}'."
                    raise ValueError(msg)
                value = int(spec[key])
                if value <= 0:
                    msg = "optimizer_sync_every values must be positive"
                    raise ValueError(msg)
                intervals.append(value)
            return intervals
        msg = (
            "optimizer_sync_every must be an int, list, dict, or None; "
            f"received {type(spec)!r}"
        )
        raise TypeError(msg)

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
            fragment = _OptimizerStateFragment(
                self._manager,
                self._model,
                self._optimizer,
                key,
                sync_intervals[idx],
                self._backup_device,
                f"{self._name_prefix}_{key}",
            )
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

        ready_fragments = []
        for fragment in self._fragments:
            if fragment.tick():
                ready_fragments.append(fragment)

        if ready_fragments:
            self._sync(ready_fragments)

    def _sync(self, fragments: list[_BaseFragment]) -> None:
        self._manager.start_quorum()
        self._prepare_sync(fragments)
        self._perform_sync(fragments)
        for fragment in fragments:
            fragment.reset()

    def _prepare_sync(self, fragments: list[_BaseFragment]) -> None:
        self._allreduce_work.clear()
        for fragment in fragments:
            self._allreduce_work.extend(fragment.prepare_sync())

    def _perform_sync(self, fragments: list[_BaseFragment]) -> None:
        for work in self._allreduce_work:
            work.wait()

        if self._manager.should_commit():
            for fragment in fragments:
                fragment.perform_sync()
                fragment.save_state()
        else:
            for fragment in fragments:
                fragment.restore_state()


class DesLocFTOptimizersContainer(FTOptimizersContainer):
    """FT optimizer container augmented with DES-LOC synchronization."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[Optimizer],
        optimizer_kwargs: dict[str, Any],
        ft_manager: Any,
        desloc_config: DesLocConfig,
        *,
        use_ft_optimizer: bool = True,
        param_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        if desloc_config.param_sync_every <= 0:
            msg = "desloc.param_sync_every must be a positive integer."
            raise ValueError(msg)

        self._desloc_config = desloc_config
        super().__init__(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager,
            use_ft_optimizer=use_ft_optimizer,
            param_groups=param_groups,
        )

        backup_device = desloc_config.resolved_backup_device()
        optimizer_sync = desloc_config.normalized_optimizer_sync()

        self._desloc_controllers: list[DesLocController] = []
        for idx, (model, optimizer) in enumerate(
            zip(self.model_parts, self.optimizers, strict=False)
        ):
            controller = DesLocController(
                manager=ft_manager,
                model=model,
                optimizer=optimizer,
                param_sync_every=desloc_config.param_sync_every,
                optimizer_sync_every=optimizer_sync,
                backup_device=backup_device,
                pin_memory=desloc_config.pin_memory,
                name_prefix=f"desloc_{idx}",
            )
            self._desloc_controllers.append(controller)

    def close_desloc(self) -> None:
        """Detach any registered DES-LOC hooks from the wrapped optimizers."""

        for controller in self._desloc_controllers:
            controller.close()
        self._desloc_controllers.clear()
