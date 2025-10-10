from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType


class ModuleLoadError(ImportError):
    """Raised when a module cannot be loaded from a provided path."""


def load_module_from_path(module_name: str, relative_path: str) -> tuple[ModuleType, Callable[[], None]]:
    """Load a module from ``relative_path`` without importing its package tree.

    The function returns the loaded module and a ``cleanup`` callable that restores
    ``sys.modules`` to its previous state. This keeps the calling test concise while
    ensuring that temporary stubs created for namespace packages do not leak into
    other tests.
    """

    module_path = Path(__file__).resolve().parents[2] / relative_path
    if not module_path.is_file():
        raise ModuleLoadError(f"Module path '{module_path}' does not exist")

    created_modules: dict[str, ModuleType] = {}
    original_modules: dict[str, ModuleType] = {}

    parent_name, _, _ = module_name.rpartition(".")
    if parent_name:
        _ensure_package_stub(parent_name, module_path.parent, created_modules, original_modules)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleLoadError(f"Unable to load spec for module '{module_name}' from '{module_path}'")

    previous_module = sys.modules.get(module_name)
    if previous_module is not None:
        original_modules[module_name] = previous_module

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    def cleanup() -> None:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        for name in reversed(created_modules.keys()):
            sys.modules.pop(name, None)
        for name, original in original_modules.items():
            sys.modules[name] = original

    return module, cleanup


def _ensure_package_stub(
    package_name: str,
    package_path: Path,
    created_modules: dict[str, ModuleType],
    original_modules: dict[str, ModuleType],
) -> None:
    if package_name in sys.modules:
        return

    parent_name, _, _ = package_name.rpartition(".")
    if parent_name:
        _ensure_package_stub(parent_name, package_path.parent, created_modules, original_modules)

    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module
    created_modules[package_name] = module
