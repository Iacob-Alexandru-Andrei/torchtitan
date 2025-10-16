#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to launch TorchFT fault-tolerant training with N replicas (one per GPU)
# Usage: ./run_train_torchft.sh [num_gpus] [config_file]
# Example: ./run_train_torchft.sh 4 ./torchtitan/experiments/fl/configs/mosaic_mup_16M.toml

rm -rf /dev/shm/*

set -ex

export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'

# Configuration
NGPU=${1:-"4"}  # Number of GPUs / replicas (default: 2)
CONFIG_FILE=${2:-"./torchtitan/experiments/fl/configs/mosaic_mup_16M_torchft_test_long.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.experiments.fl.train"}

# TorchFT lighthouse configuration
LIGHTHOUSE_HOST="localhost"
LIGHTHOUSE_PORT="29510"
LIGHTHOUSE_URL="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"

# Lighthouse settings
MIN_REPLICAS=${MIN_REPLICAS:-4}  # Minimum replicas required to start training
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}  # Quorum tick interval in milliseconds

# Log directory
LOG_DIR="./outputs/torchft_logs"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
export RUN_UUID="${RUN_UUID:-16M-baseline-${TIMESTAMP}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_UUID}}"

echo "=========================================="
echo "TorchFT Multi-Replica Training Launch"
echo "=========================================="
echo "Number of replicas: ${NGPU}"
echo "Config file: ${CONFIG_FILE}"
echo "Lighthouse URL: ${LIGHTHOUSE_URL}"
echo "Min replicas: ${MIN_REPLICAS}"
echo "Log directory: ${LOG_DIR}"
echo "=========================================="

# Function to cleanup background processes on exit
cleanup() {
    echo "Cleaning up background processes..."
    # Kill all child processes
    pkill -P $$ || true
    # Kill lighthouse if it's running
    pkill -f "torchft_lighthouse" || true
    wait
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

# Step 1: Start the TorchFT lighthouse server
echo "Starting TorchFT lighthouse server..."
uv run --no-sync torchft_lighthouse \
    --min_replicas ${MIN_REPLICAS} \
    --quorum_tick_ms ${QUORUM_TICK_MS} \
    --bind ${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT} \
    > ${LOG_DIR}/lighthouse.log 2>&1 &

LIGHTHOUSE_PID=$!
echo "Lighthouse started with PID: ${LIGHTHOUSE_PID}"

# Wait a moment for lighthouse to start
sleep 2

# Check if lighthouse is running
if ! kill -0 ${LIGHTHOUSE_PID} 2>/dev/null; then
    echo "ERROR: Lighthouse failed to start. Check ${LOG_DIR}/lighthouse.log"
    exit 1
fi

echo "Lighthouse is running successfully"

# Step 2: Launch N replicas, one per GPU
echo "Launching ${NGPU} training replicas..."

REPLICA_PIDS=()

for ((replica_id=0; replica_id<${NGPU}; replica_id++)); do
    echo "Starting replica ${replica_id} on GPU ${replica_id}..."

    # Each replica runs on a single GPU with its own replica_id
    CUDA_VISIBLE_DEVICES=${replica_id} \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${LIGHTHOUSE_URL} \
    uv run --no-sync torchrun \
        --nproc_per_node=1 \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:$((29600 + replica_id))" \
        --role rank \
        --tee 3 \
        -m ${TRAIN_FILE} \
        --job.config_file ${CONFIG_FILE} \
        --fault_tolerance.replica_id ${replica_id} \
        --fault_tolerance.group_size ${NGPU} \
        --fault_tolerance.min_replica_size ${MIN_REPLICAS} \
        > ${LOG_DIR}/replica_${replica_id}.log 2>&1 &

    REPLICA_PIDS+=($!)
    echo "Replica ${replica_id} started with PID: ${!}"

    # Small delay between launching replicas
    sleep 5
done

echo "All replicas launched. PIDs: ${REPLICA_PIDS[@]}"
echo "Lighthouse PID: ${LIGHTHOUSE_PID}"
echo ""
echo "Monitoring logs:"
echo "  Lighthouse: tail -f ${LOG_DIR}/lighthouse.log"
for ((i=0; i<${NGPU}; i++)); do
    echo "  Replica ${i}: tail -f ${LOG_DIR}/replica_${i}.log"
done
echo ""

# Wait for all replicas to complete
echo "Waiting for training to complete..."
echo "Press Ctrl+C to stop all processes"

# Monitor replica processes
failed=0
for pid in ${REPLICA_PIDS[@]}; do
    if wait ${pid}; then
        echo "Process ${pid} completed successfully"
    else
        exit_code=$?
        echo "Process ${pid} failed with exit code ${exit_code}"
        failed=1
    fi
done

# Check lighthouse
if kill -0 ${LIGHTHOUSE_PID} 2>/dev/null; then
    echo "Stopping lighthouse..."
    kill ${LIGHTHOUSE_PID}
    wait ${LIGHTHOUSE_PID} 2>/dev/null || true
fi

if [ ${failed} -eq 0 ]; then
    echo "All replicas completed successfully!"
    exit 0
else
    echo "Some replicas failed. Check logs in ${LOG_DIR}/"
    exit 1
fi
