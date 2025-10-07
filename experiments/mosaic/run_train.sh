#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A simple script to launch the Mosaic streaming training job.

# Example usage:
# ./experiments/mosaic/run_train.sh

# For multi-node training, you can use torchrun directly, e.g.:
# torchrun --nproc_per_node=2 experiments/mosaic/train.py --config-path experiments/mosaic/configs/mosaic_job.toml

set -e

# Default to 2 GPUs if not specified
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# Launch the training job
torchrun --nproc_per_node=${NPROC_PER_NODE} experiments/mosaic/train.py --config-path experiments/mosaic/configs/mosaic_job.toml