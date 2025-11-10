#!/usr/bin/env bash
# Launch a 1B cooldown run (4-way DDP) from a prior checkpoint with reset momentum and sqrt cooldown.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_cooldown_ddp.sh --source-run RUN_UUID [--source-step STEP]
                           [--resume-run-step RUN/step-N]
                           [--config-file PATH] [--gpu-ids 0,1,2,3]
                           [--rdzv-host HOST] [--rdzv-port PORT]
                           [--run-prefix PREFIX] [extra torchrun/tyro args...]

Required arguments:
  --source-run RUN_UUID        Base training run to resume from (used when
                               --resume-run-step is not provided).

Optional arguments:
  --source-step STEP           Step number to resume from (default: 61440).
  --resume-run-step PATH       Explicit resume path (overrides source-run/step).
  --config-file PATH           Config file to use (default: cooldown/base.toml).
  --gpu-ids A,B,C,D            Comma-separated GPU ids (default: 0,1,2,3).
  --rdzv-host HOST             Rendezvous host (default: 127.0.0.1).
  --rdzv-port PORT             Rendezvous port (default: 47000).
  --run-prefix PREFIX          Prefix for generated cooldown run UUIDs
                               (default: iclr2026-cooldown).
  --help                       Show this message.

All remaining arguments are forwarded to the underlying TorchTitan tyro CLI.
USAGE
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="${SCRIPT_DIR}/base.toml"
GPU_IDS="0,1,2,3"
RDZV_HOST="127.0.0.1"
RDZV_PORT="47000"
RUN_PREFIX="iclr2026-cooldown"
SOURCE_RUN=""
SOURCE_STEP="61440"
RESUME_OVERRIDE=""
LOG_RANK=${LOG_RANK:-0}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-run)
      SOURCE_RUN=$2; shift 2 ;;
    --source-step)
      SOURCE_STEP=$2; shift 2 ;;
    --resume-run-step)
      RESUME_OVERRIDE=$2; shift 2 ;;
    --config-file)
      CONFIG_FILE=$2; shift 2 ;;
    --gpu-ids)
      GPU_IDS=$2; shift 2 ;;
    --rdzv-host)
      RDZV_HOST=$2; shift 2 ;;
    --rdzv-port)
      RDZV_PORT=$2; shift 2 ;;
   --run-prefix)
      RUN_PREFIX=$2; shift 2 ;;
    --train-module)
      TRAIN_MODULE=$2; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done
      break ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ -z "${RESUME_OVERRIDE}" && -z "${SOURCE_RUN}" ]]; then
  echo "Missing --source-run (or provide --resume-run-step)." >&2
  usage; exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}
if [[ ${NUM_GPUS} -ne 4 ]]; then
  echo "Cooldown expects exactly 4 GPUs (received ${NUM_GPUS})." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${GPU_ARRAY[*]}")

sanitize() {
  local value=$1
  value=${value//\//-}
  value=${value//./_}
  value=${value// /}
  value=${value//:/-}
  echo "${value}"
}

if [[ -n "${RESUME_OVERRIDE}" ]]; then
  RESUME_TARGET=${RESUME_OVERRIDE}
else
  RESUME_TARGET="${SOURCE_RUN}/step-${SOURCE_STEP}"
fi

if [[ -z "${RESUME_TARGET}" ]]; then
  echo "Unable to derive resume path." >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
BASE_TAG=$(sanitize "${SOURCE_RUN:-cooldown}")
RUN_UUID="${RUN_PREFIX}-${BASE_TAG}-${TIMESTAMP}"

export CUDA_VISIBLE_DEVICES
export RESUME_FROM_RUN_STEP="${RESUME_TARGET}"
export RUN_UUID
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan"}
export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
export WANDB_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}

echo "Launching cooldown run ${RUN_UUID}" >&2
if [[ -n "${SOURCE_RUN}" ]]; then
  echo "  Source run: ${SOURCE_RUN}" >&2
fi
echo "  Resume path: ${RESUME_FROM_RUN_STEP}" >&2
echo "  GPUs: ${CUDA_VISIBLE_DEVICES} (nproc=${NUM_GPUS})" >&2
echo "  Rendezvous endpoint: ${RDZV_HOST}:${RDZV_PORT}" >&2

CMD_ARGS=(
  uv run --no-sync torchrun
  --nproc_per_node=${NUM_GPUS}
  --rdzv_backend=c10d
  --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}"
  --rdzv_id "${RUN_UUID}"
  --local-ranks-filter="${LOG_RANK}"
  --role rank
  --tee 3
  -m "${TRAIN_MODULE}"
  --job.config_file "${CONFIG_FILE}"
  --model.flavor "1B"
  --optimizer.builder default
  --optimizer.name AdamW
  --optimizer.lr 0.01
  --optimizer.beta1 0.9
  --optimizer.beta2 0.999
  --training.global_batch_size 64
  --training.local_batch_size 16
  --training.steps 2048
  --lr_scheduler.decay_type sqrt
  --lr_scheduler.warmup_steps 0
  --lr_scheduler.decay_ratio 0.0
  --parallelism.data_parallel_replicate_degree ${NUM_GPUS}
)

if ((${#EXTRA_ARGS[@]})); then
  CMD_ARGS+=("${EXTRA_ARGS[@]}")
fi

"${CMD_ARGS[@]}"
