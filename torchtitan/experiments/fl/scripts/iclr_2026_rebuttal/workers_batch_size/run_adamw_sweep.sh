#!/usr/bin/env bash
# Baseline AdamW sweep across worker counts and global batch sizes for 16M and 350M models.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
MODEL_SIZES=${MODEL_SIZES:-"16M 350M"}
WORKER_COUNTS=${WORKER_COUNTS:-"1 2 4 8"}
GLOBAL_BATCH_SIZES=${GLOBAL_BATCH_SIZES:-"64 128 256 512"}
RUN_PREFIX=${RUN_PREFIX:-"iclr2026"}
LOG_RANK=${LOG_RANK:-0}
NPROC_PER_NODE_DEFAULT=${NPROC_PER_NODE_DEFAULT:-}

# Networking defaults (can be overridden via env vars when launching the script).
RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
RDZV_BASE_PORT=${RDZV_BASE_PORT:-43000}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-43100}
PORT_STRIDE=${PORT_STRIDE:-5}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
  exit 1
fi

# Utility: sanitize model size for env var keys (e.g., 16M -> 16M, 350M -> 350M)
san() {
  echo "$1" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9' '_'
}

warmup_step_for() {
  local size=$1
  local key=$(san "${size}")
  local var="WARMUP_RUN_STEP_${key}"
  local value=""
  if [[ -n "${!var-}" ]]; then
    value=${!var}
  elif [[ -n "${WARMUP_RUN_STEP:-}" ]]; then
    value=${WARMUP_RUN_STEP}
  fi
  if [[ -z "${value}" ]]; then
    echo "Warmup checkpoint not provided. Set WARMUP_RUN_STEP or ${var}." >&2
    exit 1
  fi
  printf '%s' "${value}"
}

ensure_divisible() {
  local numer=$1
  local denom=$2
  if (( numer % denom != 0 )); then
    echo "Global batch size ${numer} must be divisible by worker count ${denom}." >&2
    exit 1
  fi
}

declare -i iteration=0

for model_size in ${MODEL_SIZES}; do
  warmup_step=$(warmup_step_for "${model_size}")
  for workers in ${WORKER_COUNTS}; do
    for global_bs in ${GLOBAL_BATCH_SIZES}; do
      ensure_divisible ${global_bs} ${workers}
      local_bs=$((global_bs / workers))
      if (( local_bs == 0 )); then
        echo "Local batch size resolved to zero; adjust inputs." >&2
        exit 1
      fi

      iteration+=1
      timestamp=$(date +"%Y%m%d-%H%M%S")
      run_uuid="${RUN_PREFIX}-adamw-${model_size}-w${workers}-gb${global_bs}-${timestamp}-${iteration}"

      offset=$((iteration - 1))
      rdzv_port=$((RDZV_BASE_PORT + offset * PORT_STRIDE))
      lighthouse_port=$((LIGHTHOUSE_BASE_PORT + offset * PORT_STRIDE))
      rdzv_endpoint="${RDZV_HOST}:${rdzv_port}"
      lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"

      export TORCHFT_LIGHTHOUSE="${lighthouse_url}"

      export RUN_UUID="${run_uuid}"
      export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan"}
      export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
      export WANDB_RUN_NAME="${RUN_UUID}"
      export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
      export RESUME_FROM_RUN_STEP="${warmup_step}"

      nproc=${NPROC_PER_NODE_DEFAULT:-${workers}}

      echo "=== Launching AdamW run ${RUN_UUID} (model=${model_size}, workers=${workers}, global_bs=${global_bs}, local_bs=${local_bs}) ==="
      echo "    torchrun rendezvous: ${rdzv_endpoint} | TORCHFT lighthouse: ${lighthouse_url}"

      EXTRA_ARGS=(
        --model.flavor "${model_size}"
        --optimizer.builder default
        --optimizer.name AdamW
        --optimizer.lr 0.01
        --optimizer.beta1 0.9
        --optimizer.beta2 0.999
        --optimizer.weight_decay 0.0
        --optimizer.eps 1e-08
        --training.global_batch_size "${global_bs}"
        --training.local_batch_size "${local_bs}"
        --parallelism.data_parallel_replicate_degree "${workers}"
      )

      if [[ -n "${DRY_RUN:-}" ]]; then
        printf 'Command:'
        printf ' %q' uv run --no-sync torchrun \
          "--nproc_per_node=${nproc}" \
          --rdzv_backend=c10d \
          "--rdzv_endpoint=${rdzv_endpoint}" \
          "--rdzv_id=${run_uuid}" \
          "--local-ranks-filter=${LOG_RANK}" \
          --role rank --tee 3 \
          -m "${TRAIN_MODULE}" \
          --job.config_file "${CONFIG_FILE}"
        printf ' %q' "${EXTRA_ARGS[@]}"
        printf ' %q' --run_uuid "${RUN_UUID}"
        if [[ $# -gt 0 ]]; then
          for extra_arg in "$@"; do
            printf ' %q' "${extra_arg}"
          done
        fi
        printf '\n'
        continue
      fi

      uv run --no-sync torchrun \
        --nproc_per_node="${nproc}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${rdzv_endpoint}" \
        --rdzv_id "${run_uuid}" \
        --local-ranks-filter="${LOG_RANK}" \
        --role rank \
        --tee 3 \
        -m "${TRAIN_MODULE}" \
        --job.config_file "${CONFIG_FILE}" \
        "${EXTRA_ARGS[@]}" \
        --run_uuid "${RUN_UUID}" \
        "$@"
    done
  done
done
