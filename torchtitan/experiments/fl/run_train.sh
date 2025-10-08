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

export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'
# Default to 2 GPUs if not specified
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# Launch the training job
torchrun --nproc_per_node=${NPROC_PER_NODE} experiments/mosaic/train.py --config-path experiments/mosaic/configs/mosaic_job.toml


NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/mosaic/configs/mosaic_job.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.experiments.mosaic.train"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
uv run --no-sync torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
