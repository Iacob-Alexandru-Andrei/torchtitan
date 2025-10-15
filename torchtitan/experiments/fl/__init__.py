# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Federated learning experiment extensions built on top of TorchTitan.

Importing this package registers Mosaic-specific train specs, optimizers, and
callbacks required by the FL recipes.
"""

# Import the models module to trigger the registration of any custom Mosaic
# train specs.
from . import models

__all__ = ["models"]
