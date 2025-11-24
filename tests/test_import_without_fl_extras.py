"""Smoke test that core imports succeed without FL optional deps."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable


class _BlockOptionalImports:
    """Meta path finder that blocks optional FL dependencies."""

    def __init__(self, blocked_prefixes: Iterable[str]) -> None:
        self._blocked_prefixes = tuple(blocked_prefixes)

    def find_spec(self, fullname: str, _path: object = None, _target: object = None):  # type: ignore[override]
        if fullname.startswith(self._blocked_prefixes):
            raise ModuleNotFoundError(fullname)
        return None


def _clear_torchtitan_modules() -> None:
    for name in list(sys.modules):
        if name == "torchtitan" or name.startswith("torchtitan."):
            sys.modules.pop(name)


def test_core_import_succeeds_without_fl_optional_dependencies(monkeypatch):
    """Importing the core package should not require Mosaic optional deps."""

    blocker = _BlockOptionalImports(("mosaicml", "streaming", "torchft"))
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    _clear_torchtitan_modules()

    module = importlib.import_module("torchtitan")
    assert isinstance(module, ModuleType)
