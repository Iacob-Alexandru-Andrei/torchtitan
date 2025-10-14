# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Extension hooks for integrating third-party functionality with TorchFT."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ContextManager, Protocol

__all__ = [
    "OptimizerExtension",
    "SemiSyncFactory",
    "register_optimizer_extension",
    "iter_optimizer_extensions",
    "apply_optimizer_extensions",
    "register_semi_sync_context",
    "get_semi_sync_context_factory",
]


class OptimizerExtension(Protocol):
    """Callable that augments an ``FTOptimizersContainer`` instance."""

    def __call__(
        self,
        container: "FTOptimizersContainer",
        ft_manager: Any,
    ) -> Callable[[], None] | None:
        """Install extension behaviour and return an optional cleanup callback."""


class SemiSyncFactory(Protocol):
    """Factory creating semi-sync TorchFT contexts for registered methods."""

    def __call__(
        self,
        *,
        ft_manager: "FTManager",
        optimizer: "Optimizer",
    ) -> ContextManager[Any]:
        """Return a context manager wrapping semi-sync execution."""


_optimizer_extensions: list[OptimizerExtension] = []
_semi_sync_contexts: dict[str, SemiSyncFactory] = {}


def register_optimizer_extension(extension: OptimizerExtension) -> Callable[[], None]:
    """Register an optimizer extension and return an unregistration callback."""

    _optimizer_extensions.append(extension)

    def unregister() -> None:
        try:
            _optimizer_extensions.remove(extension)
        except ValueError:
            # The extension was already removed (e.g., during cleanup).
            pass

    return unregister


def iter_optimizer_extensions() -> tuple[OptimizerExtension, ...]:
    """Return a snapshot of the currently registered optimizer extensions."""

    return tuple(_optimizer_extensions)


def apply_optimizer_extensions(
    container: "FTOptimizersContainer", ft_manager: Any
) -> list[Callable[[], None]]:
    """Invoke all registered optimizer extensions for ``container``."""

    cleanups: list[Callable[[], None]] = []
    for extension in tuple(_optimizer_extensions):
        cleanup = extension(container, ft_manager)
        if cleanup is not None:
            cleanups.append(cleanup)
    return cleanups


def register_semi_sync_context(
    name: str, factory: SemiSyncFactory
) -> Callable[[], None]:
    """Register a semi-sync context factory for ``name`` and return uninstaller."""

    key = name.lower()
    _semi_sync_contexts[key] = factory

    def unregister() -> None:
        if _semi_sync_contexts.get(key) is factory:
            del _semi_sync_contexts[key]

    return unregister


def get_semi_sync_context_factory(name: str) -> SemiSyncFactory | None:
    """Return the registered factory for ``name`` if available."""

    return _semi_sync_contexts.get(name.lower())


if TYPE_CHECKING:  # pragma: no cover - typing imports only
    from torch.optim import Optimizer
    from torchtitan.components.optimizer import FTOptimizersContainer
    from .manager import FTManager
