# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""New optimizers for FL experiments."""

from .adopt import ADOPT
from .aggmo_adamw import AggMoAdamW
from .aggmo_adopt import AggMoAdopt
from .decoupled_adamw import DecoupledAdamW
from .desloc_outer import DES_LOC_OUTER
from .aggmo_muon import AggMoMuon
from .muon import Muon
from .qhadamw import QHAdamW
from .qhadopt import QHADOPT
from .galore import GaLore
from .scion import Scion, ScionAggMo, ScionLight, QHScion

__all__ = [
    "ADOPT",
    "QHADOPT",
    "AggMoAdamW",
    "AggMoAdopt",
    "AggMoMuon",
    "DecoupledAdamW",
    "DES_LOC_OUTER",
    "QHAdamW",
    "GaLore",
    "Scion",
    "ScionAggMo",
    "QHScion",
    "ScionLight",
    "Muon",
]
