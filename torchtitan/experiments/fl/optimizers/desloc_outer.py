# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Wrapper optimizer enabling DES-LOC outer updates to leverage arbitrary PyTorch optimizers."""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = ["DES_LOC_OUTER"]


def _resolve_optimizer(target: str) -> type[Optimizer]:
    module_path, _, attr = target.rpartition(".")
    if module_path:
        module = importlib.import_module(module_path)
        cls = getattr(module, attr, None)
    else:
        cls = getattr(torch.optim, attr, None)
    if cls is None or not issubclass(cls, Optimizer):
        msg = (
            f"DES_LOC_OUTER failed to resolve base optimizer '{target}'. "
            "Provide a fully-qualified class name or torch.optim member."
        )
        raise ValueError(msg)
    return cls


class DES_LOC_OUTER(Optimizer):
    """Delegating optimizer that wraps a user-specified base optimizer.

    This wrapper exists so that DES-LOC can materialize an outer optimizer referenced
    via configuration while still presenting a standard ``torch.optim.Optimizer`` API.
    The underlying optimizer class is resolved dynamically from ``base_target``.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        base_target: str,
        base_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        base_cls = _resolve_optimizer(base_target)
        self._base_target = base_target
        self._base_kwargs = dict(base_kwargs or {})
        inner = base_cls(params, **self._base_kwargs)
        defaults = getattr(inner, "defaults", {})
        super().__init__(params, defaults)
        self._inner = inner
        # Ensure state and param groups remain shared with the wrapped optimizer
        self.param_groups = self._inner.param_groups
        self.state = self._inner.state

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookups to the wrapped optimizer when possible.

        Optimizer callbacks (e.g. OptimizerMonitor) rely on helper methods such as
        ``report_per_parameter_metrics`` that live on the underlying optimizer class.
        Expose those transparently so instrumentation continues to function when
        DES-LOC wraps the base optimizer.
        """
        if name == "_inner":
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is not None:
            try:
                return getattr(inner, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def step(self, closure=None):
        return self._inner.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return self._inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner.load_state_dict(state_dict)
