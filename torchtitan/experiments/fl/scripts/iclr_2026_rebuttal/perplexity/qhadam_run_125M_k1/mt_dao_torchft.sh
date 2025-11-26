#!/usr/bin/env bash
# Launch the AdeMaMix TorchFT (\"mt_dao\") experiment with TorchFT replicas.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)
else
  REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../.." && pwd -P)
fi
cd "${REPO_ROOT}"
export REPO_ROOT

# Remove stale shared-memory artifacts owned by the current user.
find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi

CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base_torchft.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
MIN_REPLICAS=${MIN_REPLICAS:-${NGPU}}
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}

LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"localhost"}
LIGHTHOUSE_PORT=${LIGHTHOUSE_PORT:-29510}
LIGHTHOUSE_URL="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"

LR_SWITCH_STEP=${LR_SWITCH_STEP:-2049}
ADEMAMIX_SWITCH_SCALE=${ADEMAMIX_SWITCH_SCALE:-2.0}
ADEMAMIX_NEW_VS=${ADEMAMIX_NEW_VS:-"0.00 0.95"}
ADEMAMIX_NEW_BETAS=${ADEMAMIX_NEW_BETAS:-"0.9 0.999 0.999"}
ADEMAMIX_RESET_MOMENTA=${ADEMAMIX_RESET_MOMENTA:-"exp_avg exp_avg_2"}

read -r -a ADEMAMIX_NEW_VS_ARRAY <<< "${ADEMAMIX_NEW_VS}"
read -r -a ADEMAMIX_NEW_BETAS_ARRAY <<< "${ADEMAMIX_NEW_BETAS}"
read -r -a ADEMAMIX_RESET_MOMENTA_ARRAY <<< "${ADEMAMIX_RESET_MOMENTA}"

TRAINING_ARGS=("$@")

LOG_DIR="${REPO_ROOT}/outputs/torchft_logs"
mkdir -p "${LOG_DIR}"
LIGHTHOUSE_LOG_FILE="${LOG_DIR}/lighthouse.log"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-k1qhmtdao125M"}
export RUN_UUID=${RUN_UUID:-"${RUN_PREFIX}-${TIMESTAMP}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_tune_N"}
export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
export WANDB_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}

echo "=========================================="
echo "TorchFT AdeMaMix Launch"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Config   : ${CONFIG_FILE}"
echo "Replicas : ${NGPU}"
echo "Lighthouse: ${LIGHTHOUSE_URL}"
echo "Log dir  : ${LOG_DIR}"
echo "=========================================="

declare -a REPLICA_PIDS=()
LIGHTHOUSE_PID=""

cleanup() {
  set +e
  for pid in "${REPLICA_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${LIGHTHOUSE_PID}" ]] && kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
    kill "${LIGHTHOUSE_PID}" 2>/dev/null || true
    wait "${LIGHTHOUSE_PID}" 2>/dev/null || true
  fi
  set -e
}
trap cleanup EXIT INT TERM

echo "[Lighthouse] starting at ${LIGHTHOUSE_URL}"
uv run --no-sync torchft_lighthouse \
  --min_replicas "${MIN_REPLICAS}" \
  --quorum_tick_ms "${QUORUM_TICK_MS}" \
  --bind "${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}" \
  > "${LIGHTHOUSE_LOG_FILE}" 2>&1 &
LIGHTHOUSE_PID=$!
sleep 2
if ! kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
  echo "ERROR: torchft_lighthouse failed to start. Check ${LIGHTHOUSE_LOG_FILE}" >&2
  exit 1
fi
echo "Lighthouse PID: ${LIGHTHOUSE_PID}"

export TORCHFT_LIGHTHOUSE="${LIGHTHOUSE_URL}"

AVAILABLE_GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((i=0; i<NGPU; i++)); do
    AVAILABLE_GPUS+=("${i}")
  done
fi

if (( ${#AVAILABLE_GPUS[@]} < NGPU )); then
  echo "ERROR: Requested ${NGPU} replicas but only ${#AVAILABLE_GPUS[@]} GPU(s) available." >&2
  exit 1
fi

REPLICA_GPUS=("${AVAILABLE_GPUS[@]:0:NGPU}")
for ((i=0; i<NGPU; i++)); do
  echo "Replica ${i} -> GPU ${REPLICA_GPUS[$i]}"
done

for ((replica_id=0; replica_id<NGPU; replica_id++)); do
  gpu_id="${REPLICA_GPUS[$replica_id]}"
  log_file="${LOG_DIR}/replica_${replica_id}.log"
  echo "[Replica ${replica_id}] logging to ${log_file}"

  (
    set -euo pipefail
    cd "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    rdzv_port=$((29600 + replica_id))
    uv run --no-sync torchrun \
      --nproc_per_node=1 \
      --rdzv_backend=c10d \
      --rdzv_endpoint="localhost:${rdzv_port}" \
      --role rank \
      --tee 3 \
      -m "${TRAIN_MODULE}" \
      --job.config_file "${CONFIG_FILE}" \
      --run_uuid "${RUN_UUID}" \
      --fault_tolerance.replica_id "${replica_id}" \
      --fault_tolerance.group_size "${NGPU}" \
      --fault_tolerance.min_replica_size "${MIN_REPLICAS}" \
      --lr_scheduler.switch_step "${LR_SWITCH_STEP}" \
      --lr_scheduler.switch_scale "${ADEMAMIX_SWITCH_SCALE}" \
      --fl_metrics.hyperparameter_switch.steps "${LR_SWITCH_STEP}" \
      --fl_metrics.hyperparameter_switch.new_vs "${ADEMAMIX_NEW_VS_ARRAY[@]}" \
      --fl_metrics.hyperparameter_switch.new_betas "${ADEMAMIX_NEW_BETAS_ARRAY[@]}" \
      --fl_metrics.hyperparameter_switch.reset_momenta "${ADEMAMIX_RESET_MOMENTA_ARRAY[@]}" \
      "${TRAINING_ARGS[@]}"
  ) > "${log_file}" 2>&1 &
  REPLICA_PIDS[$replica_id]=$!
  sleep 1
done

echo ""
echo "Lighthouse log: tail -f ${LIGHTHOUSE_LOG_FILE}"
for ((i=0; i<NGPU; i++)); do
  echo "Replica ${i} log: tail -f ${LOG_DIR}/replica_${i}.log"
done
echo ""

set +e
REPLICA_EXIT=0
for ((replica_id=0; replica_id<NGPU; replica_id++)); do
  pid=${REPLICA_PIDS[$replica_id]}
  if wait "${pid}"; then
    echo "Replica ${replica_id} completed successfully."
  else
    status=$?
    echo "Replica ${replica_id} exited with status ${status}."
    REPLICA_EXIT=${status}
  fi
done
set -e

if [[ -n "${LIGHTHOUSE_PID}" ]] && kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
  echo "Stopping lighthouse..."
  kill "${LIGHTHOUSE_PID}" 2>/dev/null || true
  wait "${LIGHTHOUSE_PID}" 2>/dev/null || true
  LIGHTHOUSE_PID=""
fi

if (( REPLICA_EXIT == 0 )); then
  echo "All replicas completed successfully!"
else
  echo "Some replicas failed. Check logs in ${LOG_DIR}/"
fi

exit "${REPLICA_EXIT}"
