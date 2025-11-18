#!/usr/bin/env bash
# Run a single eval-only pass for a given model size/run UUID/step triple.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../.." && pwd -P)
cd "${REPO_ROOT}"

CONFIG_FILE="${SCRIPT_DIR}/base_eval.toml"
TRAIN_MODULE="torchtitan.experiments.fl.train"
NGPU=1
RDZV_ENDPOINT="localhost:0"

MODEL_SIZE=""
TARGET_RUN_UUID=""
RESUME_STEP=""
VAL_BATCH_SIZE=4
VAL_STEPS=32

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
  WANDB_MODE          Defaults to "offline" to prevent accidental uploads.
  S3_ENDPOINT_URL     Passed through when defined by the caller.
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

# Derive a distinct UUID for the eval job to keep logs/checkpoints isolated.
EVAL_RUN_UUID=${EVAL_RUN_UUID:-"eval-${TARGET_RUN_UUID}-step-${RESUME_STEP}-$(date +%Y%m%d-%H%M%S)"}
export RUN_UUID="${EVAL_RUN_UUID}"
export RESUME_FROM_RUN_STEP="${TARGET_RUN_UUID}/step-${RESUME_STEP}"
export WANDB_MODE=${WANDB_MODE:-offline}

cli_args=(
  --job.config_file "${CONFIG_FILE}"
  --run_uuid "${EVAL_RUN_UUID}"
  --eval_only true
  --checkpoint.load_step "${RESUME_STEP}"
  --s3_checkpoint.resume_from_run_step "${TARGET_RUN_UUID}/step-${RESUME_STEP}"
  --validation.enable true
  --validation.local_batch_size "${VAL_BATCH_SIZE}"
  --validation.steps "${VAL_STEPS}"
  --model.flavor "${MODEL_SIZE}"
)

if [[ -n "${HF_ASSETS_PATH:-}" ]]; then
  cli_args+=(--model.hf_assets_path "${HF_ASSETS_PATH}")
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" \
uv run --no-sync torchrun \
  --nproc_per_node="${NGPU}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  --role rank \
  --tee 3 \
  -m "${TRAIN_MODULE}" \
  "${cli_args[@]}"
