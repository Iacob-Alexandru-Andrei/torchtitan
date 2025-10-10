"""DES-LOC integration helpers for fine-tuning optimizers.

This module provides :class:`DesLocFTOptimizersContainer`, a thin wrapper
around :class:`torchtitan.components.optimizer.FTOptimizersContainer` that
ensures DES-LOC hooks are cleaned up once the container's lifecycle ends.

The cleanup is guaranteed by combining explicit ``close()`` semantics with a
``weakref.finalize`` guard so that hooks are removed even when callers forget
to invoke :meth:`close_desloc` manually.  Tests can register a custom DES-LOC
activator through :func:`register_desloc_activator` in order to validate the
behaviour without depending on the actual DES-LOC implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
import importlib
import logging
from typing import Any, Protocol, runtime_checkable
import weakref

import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.components.optimizer import FTOptimizersContainer

logger = logging.getLogger(__name__)


@runtime_checkable
class _SupportsClose(Protocol):
    def close(self) -> None:  # pragma: no cover - structural typing helper
        """A minimal protocol for objects exposing a ``close`` method."""


DesLocActivator = Callable[[list[nn.Module], list[Optimizer]], Any | None]


_DESLOC_ACTIVATOR: DesLocActivator | None = None


def register_desloc_activator(activator: DesLocActivator | None) -> None:
    """Register a factory used to attach DES-LOC hooks.

    The callable receives the list of model parts and optimizers.  It should
    return either ``None`` when DES-LOC should be disabled or an object that
    can be closed via a ``close()`` method, is callable, or implements the
    context manager protocol.  The returned object (if any) is guaranteed to be
    closed exactly once.
    """

    global _DESLOC_ACTIVATOR
    _DESLOC_ACTIVATOR = activator


def get_desloc_activator() -> DesLocActivator | None:
    """Return the currently registered DES-LOC activator, if any."""

    return _DESLOC_ACTIVATOR


def _import_default_desloc_activator() -> DesLocActivator | None:
    """Best-effort import for an optional DES-LOC backend.

    The upstream DES-LOC implementation is optional and may live in different
    namespaces depending on the PyTorch version.  We therefore try a list of
    known module names and look for a callable named ``enable`` or
    ``enable_desloc`` that accepts the model parts and optimizers.  Any import
    failure or signature mismatch is swallowed so TorchTitan continues to work
    even when DES-LOC is unavailable.
    """

    candidate_modules = [
        "torch.distributed.algorithms.desloc",
        "torch.distributed.algorithms._optimizer.desloc",
        "torch.distributed.algorithms._desloc",
        "desloc",
    ]

    def _maybe_make_activator(module_name: str) -> DesLocActivator | None:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None

        for attr in ("enable", "enable_desloc"):
            candidate = getattr(module, attr, None)
            if callable(candidate):

                def _activator(
                    model_parts: list[nn.Module], optimizers: list[Optimizer]
                ) -> Any | None:
                    try:
                        return candidate(model_parts=model_parts, optimizers=optimizers)
                    except TypeError:
                        # Some implementations may not use keyword arguments;
                        # retry with positional ones before giving up.
                        try:
                            return candidate(model_parts, optimizers)
                        except TypeError:
                            logger.debug(
                                "DES-LOC activator %s does not match expected signature",
                                candidate,
                            )
                            return None

                return _activator

        return None

    for module_name in candidate_modules:
        activator = _maybe_make_activator(module_name)
        if activator is not None:
            logger.info("Found DES-LOC activator in module %s", module_name)
            return activator

    return None


def _resource_to_closer(resource: Any) -> Callable[[], None] | None:
    """Convert a DES-LOC resource into a callable closing primitive."""

    if resource is None:
        return None

    if isinstance(resource, AbstractContextManager):
        resource.__enter__()

        def _close_ctx() -> None:
            resource.__exit__(None, None, None)

        return _close_ctx

    if isinstance(resource, _SupportsClose):
        return resource.close

    if callable(resource):  # pragma: no branch - callable resource
        return resource

    logger.debug("Ignoring unsupported DES-LOC resource of type %s", type(resource))
    return None


class DesLocFTOptimizersContainer(FTOptimizersContainer):
    """FT optimizers container with DES-LOC lifecycle management."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[Optimizer],
        optimizer_kwargs: dict[str, Any],
        ft_manager: Any,
        *,
        desloc_activator: DesLocActivator | None = None,
        use_ft_optimizer: bool = True,
        param_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager,
            use_ft_optimizer=use_ft_optimizer,
            param_groups=param_groups,
        )

        activator = (
            desloc_activator
            if desloc_activator is not None
            else (_DESLOC_ACTIVATOR or _import_default_desloc_activator())
        )

        self._desloc_closer: Callable[[], None] | None = None
        self._desloc_finalizer: weakref.Finalize | None = None

        if activator is None:
            return

        try:
            resource = activator(self.model_parts, list(self.optimizers))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to activate DES-LOC hooks; continuing without it")
            return

        closer = _resource_to_closer(resource)
        if closer is None:
            return

        self._desloc_closer = closer
        self._desloc_finalizer = weakref.finalize(self, closer)

    def close_desloc(self) -> None:
        """Explicitly tear down DES-LOC hooks if they are active."""

        if self._desloc_closer is None:
            return

        closer = self._desloc_closer
        self._desloc_closer = None

        if self._desloc_finalizer is not None and self._desloc_finalizer.alive:
            self._desloc_finalizer.detach()
        self._desloc_finalizer = None

        try:
            closer()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Error while closing DES-LOC hooks")

    def close(self) -> None:
        """Public close hook invoked by external lifecycle managers."""

        self.close_desloc()

    def __enter__(self) -> "DesLocFTOptimizersContainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - finalizer provides coverage
        # ``weakref.finalize`` handles the happy path, but ``__del__`` keeps the
        # intent clear and provides a best-effort fallback if the finalizer was
        # detached without calling :meth:`close`.
        try:
            self.close()
        except Exception:
            logger.exception("Exception raised while finalising DES-LOC hooks")


__all__ = [
    "DesLocFTOptimizersContainer",
    "DesLocActivator",
    "get_desloc_activator",
    "register_desloc_activator",
]

