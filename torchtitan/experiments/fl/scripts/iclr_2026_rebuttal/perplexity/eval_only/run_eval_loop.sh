#!/usr/bin/env bash
# Batch evaluator that submits eval-only runs either locally or via sbatch.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUN_ONCE_SCRIPT="${SCRIPT_DIR}/run_eval_once.sh"

if [[ ! -x "${RUN_ONCE_SCRIPT}" ]]; then
  echo "Missing helper script: ${RUN_ONCE_SCRIPT}" >&2
  exit 1
fi

# ============================================================================
# User configuration
# ============================================================================

# Provide entries in the format "<model_size> <run_uuid>".
# Example:
# RUN_MODELS=(
#   "16M iclr2026-adamw16m-20251118-093019"
#   "125M iclr2026-adamw125m-20251119-120001"
# )
RUN_MODELS=(
)

iclr2026-adamw16m-20251115-020141
iclr2026-ademamix16m-20251115-113214

iclr2026-ademamix125M-20251115-190012

iclr2026-ademamix360M-20251116-083236
iclr2026-agg3ademamix125M-20251118-004306

iclr2026-agg3mtdao16m-20251118-095753
iclr2026-qhademamix16m-20251117-102445
iclr2026-qhademamix16m-20251117-142610
iclr2026-qhmtdao16m-20251117-131851
iclr2026-qhmtdao125M-20251117-160027


# Override STEP_LIST (space-separated) to customize checkpoints.
STEP_LIST=${STEP_LIST:-""}
STEP_START=${STEP_START:-2048}
STEP_END=${STEP_END:-40960}
STEP_INTERVAL=${STEP_INTERVAL:-2048}

VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
VAL_STEPS=${VAL_STEPS:-32}

USE_SBATCH=${USE_SBATCH:-true}
DRY_RUN=${DRY_RUN:-false}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-4}
SBATCH_GPUS_PER_TASK=${SBATCH_GPUS_PER_TASK:-1}
SBATCH_MEM=${SBATCH_MEM:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_QOS=${SBATCH_QOS:-}
SBATCH_CONSTRAINT=${SBATCH_CONSTRAINT:-}
SBATCH_COMMENT=${SBATCH_COMMENT:-}
SBATCH_LOG_DIR=${SBATCH_LOG_DIR:-"${SCRIPT_DIR}/logs"}
SBATCH_ADDITIONAL_ARGS=${SBATCH_ADDITIONAL_ARGS:-}
SBATCH_NODE=${SBATCH_NODE:-"ruapehu"}

# ============================================================================

usage() {
  cat <<'EOF'
Usage: run_eval_loop.sh [--sbatch|--no-sbatch] [--dry-run]

Environment variables:
  RUN_MODELS            Array entries of "<model_size> <run_uuid>" (edit script).
  STEP_LIST             Space-separated steps (defaults to 2048..40960 every 2048).
  VAL_BATCH_SIZE        Validation local batch size (default: 4).
  VAL_STEPS             Validation steps (default: 32).
  USE_SBATCH            true/false (default: true).
  DRY_RUN               true/false (default: false).
  SBATCH_*              Mirrors fields from other sweep scripts (CPUs, GPUs, etc.).

Example:
  USE_SBATCH=true DRY_RUN=true ./run_eval_loop.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch)
      USE_SBATCH=true
      shift
      ;;
    --no-sbatch)
      USE_SBATCH=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

normalize_bool() {
  local value="${1:-}"
  case "${value,,}" in
    true|1|yes|on) echo "true" ;;
    false|0|no|off|"") echo "false" ;;
    *) echo "${value,,}" ;;
  esac
}

USE_SBATCH=$(normalize_bool "${USE_SBATCH}")
DRY_RUN=$(normalize_bool "${DRY_RUN}")

if [[ ${#RUN_MODELS[@]} -eq 0 ]]; then
  echo "RUN_MODELS is empty. Edit run_eval_loop.sh and add the runs to process." >&2
  exit 1
fi

declare -a STEP_ARRAY=()
if [[ -n "${STEP_LIST}" ]]; then
  read -r -a STEP_ARRAY <<< "${STEP_LIST}"
else
  if ! [[ "${STEP_START}" =~ ^[0-9]+$ && "${STEP_END}" =~ ^[0-9]+$ && "${STEP_INTERVAL}" =~ ^[0-9]+$ ]]; then
    echo "STEP_START/STEP_END/STEP_INTERVAL must be positive integers." >&2
    exit 1
  fi
  if (( STEP_START <= 0 || STEP_INTERVAL <= 0 || STEP_END < STEP_START )); then
    echo "Invalid step bounds (start=${STEP_START}, end=${STEP_END}, interval=${STEP_INTERVAL})." >&2
    exit 1
  fi
  for ((step=STEP_START; step<=STEP_END; step+=STEP_INTERVAL)); do
    STEP_ARRAY+=("${step}")
  done
fi

if (( ${#STEP_ARRAY[@]} == 0 )); then
  echo "No evaluation steps configured." >&2
  exit 1
fi

if [[ -n "${SBATCH_LOG_DIR}" ]]; then
  mkdir -p "${SBATCH_LOG_DIR}"
fi
read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_ADDITIONAL_ARGS}"
if [[ -z "${SBATCH_ADDITIONAL_ARGS}" ]]; then
  SBATCH_EXTRA_ARRAY=()
fi

submit_sbatch_job() {
  local model_size=$1
  local run_uuid=$2
  local step=$3
  local job_idx=$4
  local total_jobs=$5

  local job_name="eval-${model_size}-s${step}"
  local sbatch_opts=(--parsable "-c" "${SBATCH_CPUS_PER_TASK}" "--gres=gpu:${SBATCH_GPUS_PER_TASK}" "--job-name=${job_name}")
  [[ -n "${SBATCH_MEM}" ]] && sbatch_opts+=("--mem=${SBATCH_MEM}")
  [[ -n "${SBATCH_TIME}" ]] && sbatch_opts+=("--time=${SBATCH_TIME}")
  [[ -n "${SBATCH_PARTITION}" ]] && sbatch_opts+=("--partition=${SBATCH_PARTITION}")
  [[ -n "${SBATCH_ACCOUNT}" ]] && sbatch_opts+=("--account=${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_QOS}" ]] && sbatch_opts+=("--qos=${SBATCH_QOS}")
  [[ -n "${SBATCH_CONSTRAINT}" ]] && sbatch_opts+=("--constraint=${SBATCH_CONSTRAINT}")
  [[ -n "${SBATCH_COMMENT}" ]] && sbatch_opts+=("--comment=${SBATCH_COMMENT}")
  [[ -n "${SBATCH_NODE}" ]] && sbatch_opts+=("-w" "${SBATCH_NODE}")
  sbatch_opts+=("--output=${SBATCH_LOG_DIR}/%j-${job_name}.out" "--error=${SBATCH_LOG_DIR}/%j-${job_name}.err")
  sbatch_opts+=("${SBATCH_EXTRA_ARRAY[@]}")

  sbatch "${sbatch_opts[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "==================================================================="
echo "Eval job ${job_idx}/${total_jobs} | Model ${model_size} | Run ${run_uuid} | Step ${step}"
echo "Node: \$(hostname) | Time: \$(date)"
echo "==================================================================="
${RUN_ONCE_SCRIPT} \
  --model-size "${model_size}" \
  --run-uuid "${run_uuid}" \
  --step "${step}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --val-steps "${VAL_STEPS}"
EOF
}

total_jobs=$(( ${#RUN_MODELS[@]} * ${#STEP_ARRAY[@]} ))
dispatch_mode=$([[ "${USE_SBATCH}" == "true" ]] && echo "sbatch" || echo "local")
echo "Dispatching ${total_jobs} eval jobs (${#RUN_MODELS[@]} run(s) x ${#STEP_ARRAY[@]} step(s)) via ${dispatch_mode}."

job_counter=0
for spec in "${RUN_MODELS[@]}"; do
  read -r model_size run_uuid <<<"${spec}"
  if [[ -z "${model_size}" || -z "${run_uuid}" ]]; then
    echo "Invalid RUN_MODELS entry '${spec}'. Expected '<model_size> <run_uuid>'." >&2
    exit 1
  fi
  for step in "${STEP_ARRAY[@]}"; do
    if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
      echo "Invalid step '${step}' (must be integer)." >&2
      exit 1
    fi
    ((++job_counter))
    echo "[EvalLoop] ${job_counter}/${total_jobs}: ${model_size} | ${run_uuid} | step ${step}"
    if [[ "${DRY_RUN}" == "true" ]]; then
      continue
    fi
    if [[ "${USE_SBATCH}" == "true" ]]; then
      submit_sbatch_job "${model_size}" "${run_uuid}" "${step}" "${job_counter}" "${total_jobs}"
    else
      "${RUN_ONCE_SCRIPT}" \
        --model-size "${model_size}" \
        --run-uuid "${run_uuid}" \
        --step "${step}" \
        --val-batch-size "${VAL_BATCH_SIZE}" \
        --val-steps "${VAL_STEPS}"
    fi
  done
done
