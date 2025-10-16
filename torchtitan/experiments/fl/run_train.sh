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

rm -rf /dev/shm/*

set -e

export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'
# Default to 2 GPUs if not specified
NPROC_PER_NODE=${NPROC_PER_NODE:-2}


NGPU=${NGPU:-"4"}
export LOG_RANK=${LOG_RANK:-0,1,2,3}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/fl/configs/mosaic_mup_16M_test_long.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.experiments.fl.train"}

export WANDB_PROJECT="torchtitan"

# Optional: Set team/entity
export WANDB_TEAM="camlsys"

# Create unified RUN_UUID for all naming (WandB, S3, dump folder)
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
export RUN_UUID="${RUN_UUID:-16M-baseline-${TIMESTAMP}}"
export WANDB_RUN_NAME="${RUN_UUID}"

# Optional: Resume from a specific run and step
# Format: {run_uuid}/step-{N}
# Example: export RESUME_FROM_RUN_STEP="16M-baseline-20251011-122516/step-10"
# export RESUME_FROM_RUN_STEP="${RESUME_FROM_RUN_STEP:-}"
# export RESUME_FROM_RUN_STEP="16M-baseline-20251011-132852/step-10"

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
RUN_UUID="${RUN_UUID}" \
RESUME_FROM_RUN_STEP="${RESUME_FROM_RUN_STEP}" \
uv run --no-sync torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} \
"$@"
