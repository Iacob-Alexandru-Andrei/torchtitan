"""Helpers for importing TorchTitan modules in tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType


def _ensure_parent_package(module_name: str, module_path: Path) -> None:
    """Ensure stub parent packages exist for ``module_name``."""

    parents = module_name.split(".")[:-1]
    for index, parent in enumerate(parents, start=1):
        qualified = ".".join(parents[:index])
        if qualified in sys.modules:
            continue
        package = types.ModuleType(qualified)
        package.__path__ = [str(module_path.parent)]
        sys.modules[qualified] = package


def load_module(module_name: str, module_path: Path) -> ModuleType:
    """Load ``module_name`` from ``module_path`` with graceful fallback."""

    try:
        return importlib.import_module(module_name)
    except Exception:
        _ensure_parent_package(module_name, module_path)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
            raise ImportError(f"Cannot load module {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


__all__ = ["load_module"]

