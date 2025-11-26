#!/usr/bin/env bash
# Run a single eval-only pass for a given model size/run UUID/step triple.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if REPO_ROOT=$(cd -- "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null); then
  :
else
  REPO_ROOT="/nfs-share/aai30/projects/torchtitan"
fi
cd "${REPO_ROOT}"

CONFIG_FILE="${SCRIPT_DIR}/base_eval.toml"
TRAIN_MODULE="torchtitan.experiments.fl.train"
NGPU=${NGPU:-1}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"localhost:0"}

MODEL_SIZE=""
TARGET_RUN_UUID=""
RESUME_STEP=""
VAL_BATCH_SIZE=4
VAL_STEPS=4096

print_usage() {
  cat <<'EOF'
Usage: run_eval_once.sh --model-size <SIZE> --run-uuid <UUID> --step <N> [options]

Required arguments:
  --model-size        Model flavor to evaluate (e.g. 16M, 125M, 360M).
  --run-uuid          Existing run UUID containing checkpoints on S3.
  --step              Step number to load (must exist for the run UUID).

Optional arguments:
  --val-batch-size    Validation local batch size (default: 4).
  --val-steps         Number of validation steps to run (default: 32).
  --help              Show this help message.

Environment overrides:
  HF_ASSETS_PATH      When set, overrides model.hf_assets_path.
  WANDB_MODE          Defaults to "online" (override to "offline" if desired).
  S3_ENDPOINT_URL     Passed through when defined by the caller.
  NGPU                Override number of processes per node (default: 1).
  RDZV_ENDPOINT       Override rendezvous endpoint (default: localhost:0).
  EVAL_SWEEP_INDEX    Optional integer suffix used for deterministic eval UUIDs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size)
      MODEL_SIZE=${2:-}
      shift 2
      ;;
    --run-uuid)
      TARGET_RUN_UUID=${2:-}
      shift 2
      ;;
    --step)
      RESUME_STEP=${2:-}
      shift 2
      ;;
    --val-batch-size)
      VAL_BATCH_SIZE=${2:-}
      shift 2
      ;;
    --val-steps)
      VAL_STEPS=${2:-}
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_SIZE}" || -z "${TARGET_RUN_UUID}" || -z "${RESUME_STEP}" ]]; then
  echo "Missing required arguments." >&2
  print_usage >&2
  exit 1
fi

if [[ "${VAL_BATCH_SIZE}" -lt 1 || "${VAL_STEPS}" -lt 1 ]]; then
  echo "Validation batch size and steps must be positive integers." >&2
  exit 1
fi
if ! [[ "${NGPU}" =~ ^[0-9]+$ ]] || (( NGPU < 1 )); then
  echo "NGPU must be a positive integer (got ${NGPU})." >&2
  exit 1
fi
if [[ -z "${RDZV_ENDPOINT}" ]]; then
  RDZV_ENDPOINT="localhost:0"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-torchtitan_validation}"
export WANDB_TEAM="${WANDB_TEAM:-camlsys}"

EVAL_SWEEP_INDEX=${EVAL_SWEEP_INDEX:-}
if [[ -n "${EVAL_SWEEP_INDEX}" && ! "${EVAL_SWEEP_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "EVAL_SWEEP_INDEX must be an integer when provided." >&2
  exit 1
fi

# Derive a distinct UUID for the eval job to keep logs/checkpoints isolated.
if [[ -n "${EVAL_SWEEP_INDEX}" ]]; then
  DEFAULT_EVAL_RUN_UUID="${TARGET_RUN_UUID}_${EVAL_SWEEP_INDEX}"
else
  DEFAULT_EVAL_RUN_UUID="eval-${TARGET_RUN_UUID}-step-${RESUME_STEP}-$(date +%Y%m%d-%H%M%S)"
fi
EVAL_RUN_UUID=${EVAL_RUN_UUID:-"${DEFAULT_EVAL_RUN_UUID}"}
export RUN_UUID="${EVAL_RUN_UUID}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EVAL_RUN_UUID}}"
export TORCHTITAN_WANDB_BASE_RUN_NAME="${TORCHTITAN_WANDB_BASE_RUN_NAME:-${EVAL_RUN_UUID}}"
export RESUME_FROM_RUN_STEP="${TARGET_RUN_UUID}/step-${RESUME_STEP}"
export WANDB_MODE=${WANDB_MODE:-online}

cli_args=(
  --job.config_file "${CONFIG_FILE}"
  --run_uuid "${EVAL_RUN_UUID}"
  --eval_only
  --checkpoint.load_step "${RESUME_STEP}"
  --s3_checkpoint.resume_from_run_step "${TARGET_RUN_UUID}/step-${RESUME_STEP}"
  --validation.enable
  --validation.local_batch_size "${VAL_BATCH_SIZE}"
  --validation.steps "${VAL_STEPS}"
  --model.flavor "${MODEL_SIZE}"
)

CHECKPOINT_EXCLUDE_FROM_LOADING=${CHECKPOINT_EXCLUDE_FROM_LOADING:-"optimizer,lr_scheduler,dataloader"}
if [[ -n "${CHECKPOINT_EXCLUDE_FROM_LOADING}" ]]; then
  cli_args+=(--checkpoint.exclude_from_loading "${CHECKPOINT_EXCLUDE_FROM_LOADING}")
fi

if [[ -n "${HF_ASSETS_PATH:-}" ]]; then
  cli_args+=(--model.hf_assets_path "${HF_ASSETS_PATH}")
fi

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
uv run --no-sync torchrun \
  --nproc_per_node="${NGPU}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  --role rank \
  --tee 3 \
  -m "${TRAIN_MODULE}" \
  "${cli_args[@]}"
