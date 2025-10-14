# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Register FL model training specifications."""

from dataclasses import replace

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .llama3_mup.train_configs import get_train_spec as get_llama3_mup_train_spec
from .mosaic_llama3 import get_train_spec as get_mosaic_llama3_train_spec
from .mosaic_llama3_mup import get_train_spec as get_mosaic_llama3_mup_train_spec


# Register the base Llama3 MuP spec (without Mosaic streaming)
def _get_llama3_mup_spec() -> TrainSpec:
    """Get the base Llama3 MuP training specification.

    This version uses standard HuggingFace datasets.
    """
    spec = get_llama3_mup_train_spec()
    return replace(spec, name="llama3_mup")


register_train_spec(_get_llama3_mup_spec())


# Register the Mosaic Llama3 spec
get_mosaic_llama3_train_spec()


# Register the Mosaic Llama3 MuP spec
get_mosaic_llama3_mup_train_spec()
