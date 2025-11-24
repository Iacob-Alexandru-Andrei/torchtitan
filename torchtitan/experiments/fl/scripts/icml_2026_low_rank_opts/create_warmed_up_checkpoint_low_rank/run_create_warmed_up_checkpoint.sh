#!/usr/bin/env bash
# Train just the warmup phase at a configurable scale so the resulting checkpoint
# can be reused by downstream jobs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
MODEL_SIZE=${MODEL_SIZE:-"16M"}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
TARGET_STEPS=${TARGET_STEPS:-2048}
LEARNING_RATE=${LEARNING_RATE:-0.01}
OPTIMIZER_NAME=${OPTIMIZER_NAME:-"GaLore"}
OPTIMIZER_BUILDER=${OPTIMIZER_BUILDER:-"mosaic"}
OPTIMIZER_BETA1=${OPTIMIZER_BETA1:-0.9}
OPTIMIZER_BETA2=${OPTIMIZER_BETA2:-0.999}
OPTIMIZER_EPS=${OPTIMIZER_EPS:-"1e-8"}
OPTIMIZER_WEIGHT_DECAY=${OPTIMIZER_WEIGHT_DECAY:-0.0}
WORKER_COUNT=${WORKER_COUNT:-1}
LOG_RANK=${LOG_RANK:-0}
RUN_PREFIX=${RUN_PREFIX:-"warmup"}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"localhost:0"}
LIGHTHOUSE_URL=${LIGHTHOUSE_URL:-${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}}
DRY_RUN=${DRY_RUN:-false}


LOCAL_BATCH_SIZE=16

usage() {
  cat <<'USAGE'
Usage: run_create_warmed_up_checkpoint.sh [options] [-- extra trainer args]

Options:
  --model-size SIZE           Model flavor to train (default: 16M).
  --global-batch-size N       Global batch size (default: 256).
  --steps N                   Number of optimizer steps, also used for warmup (default: 2048).
  --lr VALUE                  Learning rate (default: 0.01).
  --optimizer NAME            Optimizer to use (default: GaLore with projections disabled).
  --beta1 VALUE               Optimizer beta1 (default: 0.9).
  --beta2 VALUE               Optimizer beta2 (default: 0.999).
  --eps VALUE                 Optimizer epsilon (default: 1e-8).
  --weight-decay VALUE        Optimizer weight decay (default: 0.0).
  --workers N                 Data-parallel worker count (default: 4).
  --config FILE               Base TOML config (default: base.toml in this folder).
  --train-module MODULE       Python module to launch (default: torchtitan.experiments.fl.train).
  --rdzv-endpoint HOST:PORT   torchrun rendezvous endpoint (default: localhost:0).
  --lighthouse-url URL        TORCHFT lighthouse URL (default: http://localhost:29510).
  --run-prefix PREFIX         Prefix for generated RUN_UUID (default: warmup).
  --log-rank N                Only log from rank N (default: 0).
  --dry-run                   Print the resolved command without launching.
  -h, --help                  Show this help message.
  --                          Treat remaining args as trainer overrides.
USAGE
}

normalize_bool() {
  local value=${1:-}
  case "${value,,}" in
    1|true|yes|on) echo true ;;
    0|false|no|off|"") echo false ;;
    *) echo "${value}" ;;
  esac
}

TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size) MODEL_SIZE=$2; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE=$2; shift 2 ;;
    --steps) TARGET_STEPS=$2; shift 2 ;;
    --lr) LEARNING_RATE=$2; shift 2 ;;
    --optimizer) OPTIMIZER_NAME=$2; shift 2 ;;
    --beta1) OPTIMIZER_BETA1=$2; shift 2 ;;
    --beta2) OPTIMIZER_BETA2=$2; shift 2 ;;
    --eps) OPTIMIZER_EPS=$2; shift 2 ;;
    --weight-decay) OPTIMIZER_WEIGHT_DECAY=$2; shift 2 ;;
    --workers) WORKER_COUNT=$2; shift 2 ;;
    --config) CONFIG_FILE=$2; shift 2 ;;
    --train-module) TRAIN_MODULE=$2; shift 2 ;;
    --rdzv-endpoint) RDZV_ENDPOINT=$2; shift 2 ;;
    --lighthouse-url) LIGHTHOUSE_URL=$2; shift 2 ;;
    --run-prefix) RUN_PREFIX=$2; shift 2 ;;
    --log-rank) LOG_RANK=$2; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        TRAINING_ARGS+=("$@")
      fi
      break
      ;;
    *)
      TRAINING_ARGS+=("$1")
      shift
      ;;
  esac
done

DRY_RUN=$(normalize_bool "${DRY_RUN}")

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Config file not found: $1" >&2
    exit 1
  fi
}

require_integer() {
  local name=$1
  local value=$2
  if ! [[ ${value} =~ ^[0-9]+$ ]]; then
    echo "${name} must be an integer (got '${value}')." >&2
    exit 1
  fi
}

require_integer "GLOBAL_BATCH_SIZE" "${GLOBAL_BATCH_SIZE}"
require_integer "TARGET_STEPS" "${TARGET_STEPS}"
require_integer "WORKER_COUNT" "${WORKER_COUNT}"

if (( WORKER_COUNT <= 0 )); then
  echo "WORKER_COUNT must be positive." >&2
  exit 1
fi

if (( GLOBAL_BATCH_SIZE % WORKER_COUNT != 0 )); then
  echo "Global batch size ${GLOBAL_BATCH_SIZE} must be divisible by worker count ${WORKER_COUNT}." >&2
  exit 1
fi


require_file "${CONFIG_FILE}"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_UUID=${RUN_UUID:-"${RUN_PREFIX}-${MODEL_SIZE}-bs${GLOBAL_BATCH_SIZE}-s${TARGET_STEPS}-${TIMESTAMP}"}

export RUN_UUID
export WANDB_PROJECT=${WANDB_PROJECT:-"galore-tune-lr"}
export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"${RUN_UUID}"}
export TORCHTITAN_WANDB_BASE_RUN_NAME=${TORCHTITAN_WANDB_BASE_RUN_NAME:-"${RUN_UUID}"}
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-"http://taranaki.cl.cam.ac.uk:9000"}
export TORCHFT_LIGHTHOUSE="${LIGHTHOUSE_URL}"

NPROC_PER_NODE=${NPROC_PER_NODE:-${WORKER_COUNT}}

CMD=(
  uv run --no-sync torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="${RDZV_ENDPOINT}"
  --local-ranks-filter="${LOG_RANK}"
  --role rank
  --tee 3
  -m "${TRAIN_MODULE}"
  --job.config_file "${CONFIG_FILE}"
  --model.flavor "${MODEL_SIZE}"
  --optimizer.name "${OPTIMIZER_NAME}"
  --optimizer.builder "${OPTIMIZER_BUILDER}"
  --optimizer.lr "${LEARNING_RATE}"
  --optimizer.beta1 "${OPTIMIZER_BETA1}"
  --optimizer.beta2 "${OPTIMIZER_BETA2}"
  --optimizer.eps "${OPTIMIZER_EPS}"
  --optimizer.weight_decay "${OPTIMIZER_WEIGHT_DECAY}"
  # Low-rank GaLore projection is configured in base.toml (rank 32 on attention/FFN linears).
  --training.global_batch_size "${GLOBAL_BATCH_SIZE}"
  --training.local_batch_size "${LOCAL_BATCH_SIZE}"
  --training.steps "${TARGET_STEPS}"
  --lr_scheduler.warmup_steps "${TARGET_STEPS}"
  --lr_scheduler.switch_step 9999999
  --checkpoint.interval "${TARGET_STEPS}"
  --parallelism.data_parallel_replicate_degree "${WORKER_COUNT}"
  --run_uuid "${RUN_UUID}"
)

if [[ ${#TRAINING_ARGS[@]} -gt 0 ]]; then
  CMD+=("${TRAINING_ARGS[@]}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'Dry run command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

printf 'Launching warmup checkpoint run %s (model=%s, workers=%s, global_bs=%s, steps=%s)\n' \
  "${RUN_UUID}" "${MODEL_SIZE}" "${WORKER_COUNT}" "${GLOBAL_BATCH_SIZE}" "${TARGET_STEPS}"

"${CMD[@]}"
