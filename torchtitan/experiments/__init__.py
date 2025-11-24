# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import lru_cache

from torchtitan.experiments.fl.models.mosaic_adapter import MosaicTrainSpecAdapter
from torchtitan.protocols.train_spec import TrainSpec


_supported_experiments = frozenset(
    [
        "flux",
        "llama4",
        "qwen3",
        "simple_fsdp.llama3",
        "simple_fsdp.deepseek_v3",
        "vlm",
        "mosaic",
    ]
)


@lru_cache(maxsize=1)
def get_train_spec() -> TrainSpec:
    """Register and return the default Mosaic streaming TrainSpec.

    The Mosaic adapter wraps the core Llama-3 TrainSpec with streaming-aware
    builders without mutating the original configuration. Caching ensures the
    registration happens exactly once per process.
    """

    return MosaicTrainSpecAdapter("llama3").register()
