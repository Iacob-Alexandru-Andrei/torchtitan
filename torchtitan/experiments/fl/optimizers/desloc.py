import logging
from types import TracebackType
from typing import Any, List

import torch
from torch import nn, optim
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


class _ParameterFragment:
    """Manages the state and synchronization for model parameters."""

    def __init__(
        self,
        manager,
        model: nn.Module,
        sync_every: int,
        backup_device: torch.device | None,
        pin_memory: bool,
    ):
        self._manager = manager
        self._model = model
        self.sync_every = sync_every
        self._backup_device = backup_device
        self._pin_memory = pin_memory

        self._local_step = 0
        self._original_parameters: dict[str, torch.Tensor] = {}
        self._averaged_parameters: List[torch.Tensor] = []

        self._init_backup_storage()
        self.save_state()

    def _init_backup_storage(self) -> None:
        for name, p in self._model.named_parameters():
            t = torch.empty_like(p.data, device=self._backup_device)
            if (
                self._pin_memory
                and t.device.type == "cpu"
                and torch.cuda.is_available()
            ):
                t = t.pin_memory()
            self._original_parameters[name] = t

    def save_state(self) -> None:
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                self._original_parameters[name].copy_(p.data, non_blocking=True)

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                p.data.copy_(self._original_parameters[name])

    def prepare_sync(self) -> list[Any]:
        self._averaged_parameters.clear()
        allreduce_work = []
        for p in self._model.parameters():
            avg_param = p.data.clone()
            allreduce_work.append(self._manager.allreduce(avg_param))
            self._averaged_parameters.append(avg_param)
        return allreduce_work

    def perform_sync(self) -> None:
        with torch.no_grad():
            for param, avg_param in zip(
                self._model.parameters(), self._averaged_parameters
            ):
                param.data.copy_(avg_param)

    def register_state_dict_fn(self) -> None:
        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            for name, param in state_dict.items():
                if name in self._original_parameters:
                    self._original_parameters[name].copy_(param)

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_parameters

        self._manager.register_state_dict_fn(
            "DesLoc_ParameterFragment", load_fn, save_fn
        )


class _OptimizerStateFragment:
    """Manages the state and synchronization for a single optimizer state (e.g., 'exp_avg')."""

    def __init__(
        self,
        manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        state_key: str,
        sync_every: int,
        backup_device: torch.device | None,
    ):
        self._manager = manager
        self._model = model
        self._optimizer = optimizer
        self.state_key = state_key
        self.sync_every = sync_every
        self._backup_device = backup_device

        self._local_step = 0
        self._original_state_tensors: dict[str, torch.Tensor] = {}
        self._averaged_state_tensors: List[torch.Tensor] = []
        self._param_map = {name: p for name, p in self._model.named_parameters()}

        self._init_backup_storage()
        self.save_state()

    def _init_backup_storage(self) -> None:
        for name, p in self._model.named_parameters():
            if (
                p in self._optimizer.state
                and self.state_key in self._optimizer.state[p]
            ):
                self._original_state_tensors[name] = torch.empty_like(
                    self._optimizer.state[p][self.state_key], device=self._backup_device
                )

    def save_state(self) -> None:
        with torch.no_grad():
            for name, backup_tensor in self._original_state_tensors.items():
                p = self._param_map[name]
                backup_tensor.copy_(self._optimizer.state[p][self.state_key])

    def restore_state(self) -> None:
        with torch.no_grad():
            for name, backup_tensor in self._original_state_tensors.items():
                p = self._param_map[name]
                if (
                    p in self._optimizer.state
                    and self.state_key in self._optimizer.state[p]
                ):
                    self._optimizer.state[p][self.state_key].copy_(backup_tensor)

    def prepare_sync(self) -> list[Any]:
        self._averaged_state_tensors.clear()
        allreduce_work = []
        for name in self._original_state_tensors:
            p = self._param_map[name]
            state_tensor = self._optimizer.state[p][self.state_key]
            avg_state = state_tensor.clone().detach()
            allreduce_work.append(self._manager.allreduce(avg_state))
            self._averaged_state_tensors.append(avg_state)
        return allreduce_work

    def perform_sync(self) -> None:
        with torch.no_grad():
            param_names_with_state = list(self._original_state_tensors.keys())
            for i, name in enumerate(param_names_with_state):
                p = self._param_map[name]
                self._optimizer.state[p][self.state_key].copy_(
                    self._averaged_state_tensors[i]
                )

    def register_state_dict_fn(self) -> None:
        key = f"DesLoc_OptimizerStateFragment_{self.state_key}"

        def load_fn(state_dict: dict[str, torch.Tensor]) -> None:
            self._original_state_tensors = state_dict

        def save_fn() -> dict[str, torch.Tensor]:
            return self._original_state_tensors

        self._manager.register_state_dict_fn(key, load_fn, save_fn)


class DesLoc:
    """
    Implements Desynchronized Local SGD (DesLoc), a fault-tolerant algorithm designed
    for distributed training. DesLoc decouples the synchronization of model
    parameters and optimizer states, allowing them to be synchronized at different
    frequencies.
    """

    def __init__(
        self,
        manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        param_sync_every: int,
        optimizer_sync_every: list[int],
        backup_device: torch.device | None = None,
        pin_memory: bool = True,
    ):
        if manager._use_async_quorum:
            raise ValueError(
                "DesLoc requires synchronous quorum. "
                "Ensure the manager is initialized with use_async_quorum=False."
            )

        self._manager = manager
        self._local_optimizer = optimizer
        self._allreduce_work: list[Any] = []
        self._fragments: list[_ParameterFragment | _OptimizerStateFragment] = []

        self._is_opt_init = False
        self._opt_sync_every_for_lazy_init = optimizer_sync_every
        self._backup_device_for_lazy_init = backup_device
        self._model_for_lazy_init = model

        param_fragment = _ParameterFragment(
            manager,
            model,
            param_sync_every,
            backup_device,
            pin_memory,
        )
        self._fragments.append(param_fragment)
        param_fragment.register_state_dict_fn()

        self._local_optimizer.register_step_post_hook(self._step_post_hook)

    def _lazy_init_optimizer_fragments(self) -> None:
        all_state_keys = set()
        for p_state in self._local_optimizer.state.values():
            for key, value in p_state.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    all_state_keys.add(key)

        state_keys = sorted(list(all_state_keys))

        if not state_keys:
            if self._opt_sync_every_for_lazy_init:
                logger.warning(
                    "optimizer_sync_every was provided, but no non-scalar "
                    "tensor states were found in the optimizer (e.g., vanilla SGD)."
                )
            self._is_opt_init = True
            return

        if len(state_keys) != len(self._opt_sync_every_for_lazy_init):
            raise ValueError(
                f"Mismatch between number of detected optimizer states ({len(state_keys)}) "
                f"and length of optimizer_sync_every ({len(self._opt_sync_every_for_lazy_init)}). "
                f"Detected states: {state_keys}"
            )

        for i, key in enumerate(state_keys):
            opt_fragment = _OptimizerStateFragment(
                self._manager,
                self._model_for_lazy_init,
                self._local_optimizer,
                key,
                self._opt_sync_every_for_lazy_init[i],
                self._backup_device_for_lazy_init,
            )
            self._fragments.append(opt_fragment)
            opt_fragment.register_state_dict_fn()

        self._is_opt_init = True
        del self._opt_sync_every_for_lazy_init
        del self._backup_device_for_lazy_init

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> None:
        if not self._is_opt_init:
            self._lazy_init_optimizer_fragments()

        ready_fragments = []
        for f in self._fragments:
            f._local_step += 1
            if f._local_step >= f.sync_every:
                ready_fragments.append(f)

        if ready_fragments:
            self.sync(ready_fragments)

    def sync(
        self, fragments_to_sync: list[_ParameterFragment | _OptimizerStateFragment]
    ) -> None:
        self._manager.start_quorum()
        self.prepare_sync(fragments_to_sync)
        self.perform_sync(fragments_to_sync)

        for f in fragments_to_sync:
            f._local_step = 0

    def prepare_sync(
        self, fragments_to_sync: list[_ParameterFragment | _OptimizerStateFragment]
    ) -> None:
        self._allreduce_work.clear()
        for fragment in fragments_to_sync:
            self._allreduce_work.extend(fragment.prepare_sync())

    def perform_sync(
        self, fragments_to_sync: list[_ParameterFragment | _OptimizerStateFragment]
    ) -> None:
        for work in self._allreduce_work:
            work.wait()

        if self._manager.should_commit():
            for fragment in fragments_to_sync:
                fragment.perform_sync()
                fragment.save_state()
        else:
            for fragment in fragments_to_sync:
                fragment.restore_state()