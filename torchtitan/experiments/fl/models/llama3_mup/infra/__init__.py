# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Parallelization utilities for Llama-3 MuP models."""
from .parallelize import parallelize_llama_mup

__all__ = ["parallelize_llama_mup"]
