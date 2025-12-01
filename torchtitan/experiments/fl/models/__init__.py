# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Register FL model training specifications."""

from dataclasses import replace

from torchtitan.protocols.train_spec import register_train_spec

from .flux import get_train_spec as get_flux_train_spec
from .llama3_mup.train_configs import get_train_spec as get_llama3_mup_train_spec
from .llama3_mup_disco.train_configs import (
    get_train_spec as get_llama3_mup_disco_train_spec,
)


def _register_base_specs() -> None:
    """Register base (non-Mosaic) TrainSpecs."""
    register_train_spec(get_flux_train_spec())
    register_train_spec(get_llama3_mup_train_spec())
    register_train_spec(get_llama3_mup_disco_train_spec())


def _register_mosaic_specs() -> None:
    """Register Mosaic-enabled TrainSpecs after base specs exist."""
    from .mosaic_llama3 import get_train_spec as get_mosaic_llama3_train_spec
    from .mosaic_llama3_mup import get_train_spec as get_mosaic_llama3_mup_train_spec
    from .mosaic_llama3_mup_disco import (
        get_train_spec as get_mosaic_llama3_mup_disco_train_spec,
    )

    get_mosaic_llama3_train_spec()
    get_mosaic_llama3_mup_train_spec()
    disco_spec = get_mosaic_llama3_mup_disco_train_spec()
    register_train_spec(replace(disco_spec, name="mosaic_llama3_mup_scion"))


_register_base_specs()
_register_mosaic_specs()
