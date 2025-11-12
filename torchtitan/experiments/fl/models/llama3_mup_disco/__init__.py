# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MuP-augmented LLaMA3 model definitions for federated learning experiments."""

from .model.mup_args import TransformerModelArgs
from .model.mup_model import Transformer
from .train_configs import get_train_spec

__all__ = ["Transformer", "TransformerModelArgs", "get_train_spec"]
