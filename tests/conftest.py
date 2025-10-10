"""Test configuration helpers for TorchTitan's unit test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is importable so tests can use the torchtitan package
# without custom importlib boilerplate in each file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

