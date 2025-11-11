#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:4
#SBATCH --job-name=tune_lr_torchft
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to launch TorchFT fault-tolerant training with N replicas (one per GPU)
# Usage: ./run_train_torchft.sh [num_gpus] [config_file]
# Example: ./run_train_torchft.sh 4 ./torchtitan/experiments/fl/configs/mosaic_mup_16M.toml

# Resolve repository root so sbatch jobs run from the project directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    # When not submitted via sbatch, keep the repository root as the directory
    # where the script was launched from (the current working directory).
    REPO_ROOT="$(pwd -P)"
fi
cd "${REPO_ROOT}"
export REPO_ROOT

# Drop leftover shared memory artifacts owned by the current user, ignoring permission errors.
find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

set -ex

export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'

# # Ensure Python can import the torchtitan package when started via sbatch.
# if [[ -z "${PYTHONPATH:-}" ]]; then
#     export PYTHONPATH="${REPO_ROOT}"
# else
#     export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
# fi
echo "PYTHONPATH=${PYTHONPATH:-}"

# Configuration
NGPU=${1:-"4"}  # Number of GPUs / replicas (default: 2)
CONFIG_FILE=${2:-"./torchtitan/experiments/fl/configs/mosaic_mup_1B_adamw_torchft.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.experiments.fl.train"}

# TorchFT lighthouse configuration
LIGHTHOUSE_HOST="localhost"
LIGHTHOUSE_PORT="29510"
LIGHTHOUSE_URL="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"

# Lighthouse settings
MIN_REPLICAS=${MIN_REPLICAS:-4}  # Minimum replicas required to start training
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}  # Quorum tick interval in milliseconds

# Print current working directory

echo "Current working directory: $(pwd)"

# Log directory
LOG_DIR="${REPO_ROOT}/outputs/torchft_logs"
mkdir -p "${LOG_DIR}"
LIGHTHOUSE_LOG_FILE="${LOG_DIR}/lighthouse.log"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
export RUN_UUID="${RUN_UUID:-fladam-1B-${TIMESTAMP}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_UUID}}"
export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX="${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}"

echo "=========================================="
echo "TorchFT Multi-Replica Training Launch"
echo "=========================================="
echo "Number of replicas: ${NGPU}"
echo "Config file: ${CONFIG_FILE}"
echo "Lighthouse URL: ${LIGHTHOUSE_URL}"
echo "Min replicas: ${MIN_REPLICAS}"
echo "Log directory: ${LOG_DIR}"
echo "=========================================="

declare -a ACTIVE_REPLICA_PIDS=()
ACTIVE_LIGHTHOUSE_PID=""
declare -A REPLICA_PID_TO_ID=()

cleanup_active_processes() {
    set +e
    for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${ACTIVE_LIGHTHOUSE_PID:-}" ]] && kill -0 "${ACTIVE_LIGHTHOUSE_PID}" 2>/dev/null; then
        kill "${ACTIVE_LIGHTHOUSE_PID}" 2>/dev/null || true
        wait "${ACTIVE_LIGHTHOUSE_PID}" 2>/dev/null || true
    fi
    ACTIVE_REPLICA_PIDS=()
    REPLICA_PID_TO_ID=()
    set -e
}
trap cleanup_active_processes EXIT INT TERM

start_lighthouse() {
    local host=$1
    local port=$2
    local log_file=$3

    uv run --no-sync torchft_lighthouse \
        --min_replicas "${MIN_REPLICAS}" \
        --quorum_tick_ms "${QUORUM_TICK_MS}" \
        --bind "${host}:${port}" \
        > "${log_file}" 2>&1 &

    local pid=$!
    sleep 2
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: Lighthouse failed to start. Check ${log_file}" >&2
        return 1
    fi
    echo "${pid}"
}

stop_lighthouse() {
    local pid=$1
    if [[ -z "${pid}" ]]; then
        return
    fi
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

launch_replica_process() {
    local replica_id=$1
    local gpu_id=$2
    local log_file=$3
    local config_file=$4

    (
        set -euo pipefail
        cd "${REPO_ROOT}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export PYTORCH_ALLOC_CONF="expandable_segments:True"
        local rdzv_port=$((29600 + replica_id))
        uv run --no-sync torchrun \
            --nproc_per_node=1 \
            --rdzv_backend c10d \
            --rdzv_endpoint="localhost:${rdzv_port}" \
            --role rank \
            --tee 3 \
            -m "${TRAIN_FILE}" \
            --job.config_file "${config_file}" \
            --fault_tolerance.replica_id "${replica_id}" \
            --fault_tolerance.group_size "${NGPU}" \
            --fault_tolerance.min_replica_size "${MIN_REPLICAS}"
    ) > "${log_file}" 2>&1 &

    echo $!
}

wait_for_replicas() {
    local status=0
    set +e
    while true; do
        local pending=false
        for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
            if [[ -n "${pid}" ]]; then
                pending=true
                break
            fi
        done

        if [[ "${pending}" == "false" ]]; then
            break
        fi

        local finished_pid=""
        wait -n -p finished_pid
        local exit_code=$?
        if (( exit_code == 127 )); then
            status=$(( status == 0 ? 127 : status ))
            break
        fi

        if [[ -z "${REPLICA_PID_TO_ID[${finished_pid}]+_}" ]]; then
            # Some other child process finished; keep waiting for replicas.
            continue
        fi

        local replica_idx="${REPLICA_PID_TO_ID[${finished_pid}]}"
        if (( exit_code == 0 )); then
            echo "Replica ${replica_idx} (pid ${finished_pid}) completed successfully."
        else
            echo "Replica ${replica_idx} (pid ${finished_pid}) exited with status ${exit_code}."
            status=${exit_code}
        fi
        unset "REPLICA_PID_TO_ID[${finished_pid}]"

        for idx in "${!ACTIVE_REPLICA_PIDS[@]}"; do
            if [[ "${ACTIVE_REPLICA_PIDS[idx]}" == "${finished_pid}" ]]; then
                ACTIVE_REPLICA_PIDS[idx]=""
                break
            fi
        done

        if (( exit_code != 0 )); then
            break
        fi
    done
    set -e

    if (( status != 0 )); then
        for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" 2>/dev/null || true
            fi
        done
        for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
            if [[ -n "${pid}" ]]; then
                wait "${pid}" 2>/dev/null || true
            fi
        done
    fi

    ACTIVE_REPLICA_PIDS=()
    REPLICA_PID_TO_ID=()
    return "${status}"
}

# Step 1: Start the TorchFT lighthouse server
echo "Starting TorchFT lighthouse server..."
echo "[Lighthouse] logging to ${LIGHTHOUSE_LOG_FILE}"
ACTIVE_LIGHTHOUSE_PID=$(start_lighthouse "${LIGHTHOUSE_HOST}" "${LIGHTHOUSE_PORT}" "${LIGHTHOUSE_LOG_FILE}") || exit 1
echo "Lighthouse started with PID: ${ACTIVE_LIGHTHOUSE_PID}"
echo "Lighthouse is running successfully"

# Step 2: Launch N replicas, one per GPU
echo "Launching ${NGPU} training replicas as background processes..."
export TORCHFT_LIGHTHOUSE=${LIGHTHOUSE_URL}

AVAILABLE_GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    for ((i=0; i<NGPU; i++)); do
        AVAILABLE_GPUS+=("$i")
    done
fi

if (( ${#AVAILABLE_GPUS[@]} < NGPU )); then
    echo "ERROR: Requested ${NGPU} replicas but only ${#AVAILABLE_GPUS[@]} GPU(s) are available via CUDA_VISIBLE_DEVICES."
    exit 1
fi

REPLICA_GPUS=("${AVAILABLE_GPUS[@]:0:NGPU}")

echo "GPU assignments for replicas:"
for ((i=0; i<NGPU; i++)); do
    echo "  Replica ${i} -> GPU ${REPLICA_GPUS[$i]}"
done

for ((replica_id=0; replica_id<NGPU; replica_id++)); do
    gpu_id="${REPLICA_GPUS[$replica_id]}"
    log_file="${LOG_DIR}/replica_${replica_id}.log"
    echo "[Replica ${replica_id}] logging to ${log_file}"

    replica_pid=$(launch_replica_process "${replica_id}" "${gpu_id}" "${log_file}" "${CONFIG_FILE}")
    ACTIVE_REPLICA_PIDS+=("${replica_pid}")
    REPLICA_PID_TO_ID["${replica_pid}"]="${replica_id}"
    sleep 1
done

echo "Lighthouse PID: ${ACTIVE_LIGHTHOUSE_PID}"
echo ""
echo "Monitoring logs:"
echo "  Lighthouse: tail -f ${LIGHTHOUSE_LOG_FILE}"
for ((i=0; i<NGPU; i++)); do
    echo "  Replica ${i}: tail -f ${LOG_DIR}/replica_${i}.log"
done
echo ""

echo "Waiting for replicas to finish..."
if wait_for_replicas; then
    REPLICA_EXIT=0
else
    REPLICA_EXIT=$?
fi

echo "Stopping lighthouse..."
stop_lighthouse "${ACTIVE_LIGHTHOUSE_PID}"
ACTIVE_LIGHTHOUSE_PID=""

if [ ${REPLICA_EXIT} -eq 0 ]; then
    echo "All replicas completed successfully!"
    exit 0
else
    echo "Some replicas failed. Check logs in ${LOG_DIR}/"
    exit ${REPLICA_EXIT}
fi
