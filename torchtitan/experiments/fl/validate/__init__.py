# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FL validation utilities (wrappers around the base validate module)."""

import importlib.util
import pathlib

from torchtitan.experiments.fl.validate.flux_validator import (
    FLFluxValidator,
    build_fl_flux_validator,
)

# Load the legacy validate.py module under a different name to avoid circular imports.
_base_path = pathlib.Path(__file__).parent.parent.joinpath("validate.py")
_spec = importlib.util.spec_from_file_location(
    "torchtitan.experiments.fl._validate_base", _base_path
)
if _spec and _spec.loader:
    _base_validate = importlib.util.module_from_spec(_spec)
    import sys

    sys.modules[_spec.name] = _base_validate
    _spec.loader.exec_module(_base_validate)  # type: ignore[arg-type]
else:  # pragma: no cover - defensive
    raise ImportError("Failed to load base validate module")

MosaicValidator = _base_validate.MosaicValidator
MosaicValidatorRequest = _base_validate.MosaicValidatorRequest
build_mosaic_validator = _base_validate.build_mosaic_validator

__all__ = [
    "MosaicValidator",
    "MosaicValidatorRequest",
    "build_mosaic_validator",
    "FLFluxValidator",
    "build_fl_flux_validator",
]
