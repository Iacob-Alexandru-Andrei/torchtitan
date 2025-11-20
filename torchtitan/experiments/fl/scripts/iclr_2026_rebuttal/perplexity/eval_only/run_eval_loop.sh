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
  # "16M iclr2026-qhmtdao16m-20251119-213808"
  # "125M iclr2026-qhmtdao125M-20251119-115225"
  # "360M iclr2026-agg3mtdao360M-20251118-121555"
  # "16M iclr2026-adamw16m-20251115-020141"
  # "125M iclr2026-ademamix125M-20251115-190012"
  # "125M iclr2026-ademamix125M-20251115-190012"
  # "16M iclr2026-ademamix16m-20251115-113214"
  # "360M iclr2026-ademamix360M-20251116-083236"
  # "125M iclr2026-agg3ademamix125M-20251118-004306"
  # "360M iclr2026-agg3ademamix360M-20251118-123143"
  # "16M iclr2026-agg3mtdao16m-20251118-095753"
  # "125M iclr2026-qhadam125M-20251119-114858"
  # "16M iclr2026-qhadam16m-20251119-213744"
  # "16M iclr2026-qhademamix16m-20251117-102445"
  # "16M iclr2026-qhademamix16m-20251117-142610"
  # "125M iclr2026-qhmtdao125M-20251117-160027"
  # "16M iclr2026-qhmtdao16m-20251117-131851"
  # "125M iclr2026-strmtdao125M-20251119-235954"
  # "16M iclr2026-agg3ademamix16m-20251118-094352"
  # "125M iclr2026-mtdao125M-20251116-174945"
  # "16M iclr2026-mtdao16m-20251116-153626"
  # "125M iclr2026-qhademamix125M-20251117-142757"
  # "360M iclr2026-qhademamix360M-20251117-152932"
  # "16M iclr2026-qhmtdao16m-20251119-181645"
  # "16M iclr2026-qhmtdao16m-20251119-184740"
  # "16M iclr2026-qhmtdao16m-20251119-204917"
  # "16M iclr2026-strdill16M-20251119-231627"
  # "125M iclr2026-strlocaladam125M-20251120-002605"
  # "16M iclr2026-strmtdao16M-20251119-224938"
  # "360M iclr2026-qhadam360M-20251119-110436"
  # "360M iclr2026-qhmtdao360M-20251119-111504"
  # "125M iclr2026-adamw125M-20251115-113315"
  # "360M iclr2026-adamw360M-20251115-215917"
  # "125M iclr2026-agg3mtdao125M-20251118-004346"
  # "360M iclr2026-strmtdao360M-20251120-063740"
  # "360M iclr2026-strdill360M-20251120-064356"
  # "125M iclr2026-strdill125M-20251120-000630"
  # "360M iclr2026-strlocaladam360M-20251120-083226"
  "125M iclr2026-noclip_qhmtdao125M-20251120-083432"
  "360M iclr2026-noclip_qhmtdao360M-20251120-083636"
)

# Route eval logging to torchtitan_validation on camlsys unless caller overrides.
export WANDB_PROJECT="${WANDB_PROJECT:-torchtitan_validation}"
export WANDB_TEAM="${WANDB_TEAM:-camlsys}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX="${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}"

# Override STEP_LIST (space-separated) to customize checkpoints.
STEP_LIST=${STEP_LIST:-""}
STEP_START=${STEP_START:-2048}
STEP_END=${STEP_END:-40960}
STEP_INTERVAL=${STEP_INTERVAL:-2048}

VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
VAL_STEPS=${VAL_STEPS:-4096}

USE_SBATCH=${USE_SBATCH:-true}
DRY_RUN=${DRY_RUN:-false}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-8}
SBATCH_GPUS_PER_TASK=${SBATCH_GPUS_PER_TASK:-1}
SBATCH_MAX_CHAINS=${SBATCH_MAX_CHAINS:-1}
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
  SBATCH_MAX_CHAINS     Number of sbatch dependency chains (default: 2).
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

if ! [[ "${SBATCH_MAX_CHAINS}" =~ ^[0-9]+$ ]] || (( SBATCH_MAX_CHAINS < 1 )); then
  echo "SBATCH_MAX_CHAINS must be a positive integer (got ${SBATCH_MAX_CHAINS})." >&2
  exit 1
fi

serialize_csv() {
  local IFS=','
  printf "%s" "$*"
}

wandb_run_exists() {
  local query_run_name="$1"
  local entity="${WANDB_ENTITY:-${WANDB_TEAM:-}}"
  local project="${WANDB_PROJECT:-}"
  if [[ -z "${entity}" || -z "${project}" || -z "${query_run_name}" ]]; then
    echo "[wandb-check] Missing entity/project/run_name; cannot search for '${query_run_name}'." >&2
    return 1
  fi
  local python_output
  python_output=$(python3 - "$entity" "$project" "$query_run_name" <<'PY'
import re
import sys

entity, project, run_name = sys.argv[1:]

try:
    import wandb  # noqa: F401
except Exception as exc:
    print(f"[wandb-check] Failed to import wandb: {exc}")
    sys.exit(2)

try:
    api = wandb.Api()
except Exception as exc:
    print(f"[wandb-check] Failed to initialize WandB API: {exc}")
    sys.exit(2)

project_path = f"{entity}/{project}"
print(f"[wandb-check] Searching '{project_path}' for run '{run_name}'")
filters_to_try = [
    ("config.run_uuid.value", {"config.run_uuid.value": run_name}),
    ("group", {"group": run_name}),
    ("display_name", {"display_name": run_name}),
    (
        "display_name regex",
        {"display_name": {"$regex": f"^{re.escape(run_name)}(-worker.*)?$"}},
    ),
]

for label, filter_query in filters_to_try:
    try:
        runs = api.runs(project_path, filters=filter_query, per_page=1)
    except Exception as exc:
        print(f"[wandb-check]   - Filter '{label}' failed ({exc})")
        continue

    match = None
    for candidate in runs:
        match = candidate
        break

    if match is None:
        print(f"[wandb-check]   - Filter '{label}': no matches")
        continue

    run_display_name = getattr(match, "display_name", None) or getattr(match, "name", None) or getattr(match, "id", "<unknown>")
    run_url = getattr(match, "url", "")
    suffix = f" ({run_url})" if run_url else ""
    print(f"[wandb-check]   - Filter '{label}': found '{run_display_name}'{suffix}")
    sys.exit(0)

print(f"[wandb-check] No matches for '{run_name}' in '{project_path}'")
sys.exit(1)
PY
)
  local status=$?
  if [[ -n "${python_output}" ]]; then
    echo "${python_output}" >&2
  fi
  [[ ${status} -eq 0 ]]
}

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

for step in "${STEP_ARRAY[@]}"; do
  if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
    echo "Invalid step '${step}' (must be integer)." >&2
    exit 1
  fi
done

TOTAL_STEPS=${#STEP_ARRAY[@]}

if [[ -n "${SBATCH_LOG_DIR}" ]]; then
  mkdir -p "${SBATCH_LOG_DIR}"
fi
read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_ADDITIONAL_ARGS}"
if [[ -z "${SBATCH_ADDITIONAL_ARGS}" ]]; then
  SBATCH_EXTRA_ARRAY=()
fi
declare -a SBATCH_CHAIN_LAST_IDS=()
declare -a SBATCH_JOB_IDS=()
if [[ "${USE_SBATCH}" == "true" ]]; then
  for ((chain_idx = 0; chain_idx < SBATCH_MAX_CHAINS; ++chain_idx)); do
    SBATCH_CHAIN_LAST_IDS[chain_idx]=''
  done
fi

run_steps_locally() {
  local model_size=$1
  local run_uuid=$2
  local job_idx=$3
  local total_jobs=$4
  local step_values_csv=$5
  local step_indices_csv=$6
  IFS=',' read -r -a local_steps <<<"${step_values_csv}"
  IFS=',' read -r -a local_indices <<<"${step_indices_csv}"
  local pending_count=${#local_steps[@]}
  echo "Running eval job ${job_idx}/${total_jobs} locally | Model ${model_size} | Run ${run_uuid} | Steps ${pending_count}"
  for idx in "${!local_steps[@]}"; do
    step="${local_steps[idx]}"
    step_index="${local_indices[idx]}"
    echo "[EvalRun] ${model_size} | ${run_uuid} | step ${step} (${idx}/${pending_count})"
    EVAL_SWEEP_INDEX="${step_index}" "${RUN_ONCE_SCRIPT}" \
      --model-size "${model_size}" \
      --run-uuid "${run_uuid}" \
      --step "${step}" \
      --val-batch-size "${VAL_BATCH_SIZE}" \
      --val-steps "${VAL_STEPS}"
  done
}

submit_sbatch_run_job() {
  local model_size=$1
  local run_uuid=$2
  local job_idx=$3
  local total_jobs=$4
  local step_values=$5
  local step_indices=$6
  local pending_steps=$7
   local dependency_job=${8:-}
   local chain_index=${9:-0}

  local job_name="eval-${model_size}-${job_idx}"
  local sbatch_opts=(--parsable "-c" "${SBATCH_CPUS_PER_TASK}" "--gres=gpu:${SBATCH_GPUS_PER_TASK}" "--job-name=${job_name}")
  [[ -n "${SBATCH_MEM}" ]] && sbatch_opts+=("--mem=${SBATCH_MEM}")
  [[ -n "${SBATCH_TIME}" ]] && sbatch_opts+=("--time=${SBATCH_TIME}")
  [[ -n "${SBATCH_PARTITION}" ]] && sbatch_opts+=("--partition=${SBATCH_PARTITION}")
  [[ -n "${SBATCH_ACCOUNT}" ]] && sbatch_opts+=("--account=${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_QOS}" ]] && sbatch_opts+=("--qos=${SBATCH_QOS}")
  [[ -n "${SBATCH_CONSTRAINT}" ]] && sbatch_opts+=("--constraint=${SBATCH_CONSTRAINT}")
  [[ -n "${SBATCH_COMMENT}" ]] && sbatch_opts+=("--comment=${SBATCH_COMMENT}")
  [[ -n "${SBATCH_NODE}" ]] && sbatch_opts+=("--nodelist=${SBATCH_NODE}")
  [[ -n "${dependency_job}" ]] && sbatch_opts+=("--dependency=afterany:${dependency_job}")
  sbatch_opts+=("--output=${SBATCH_LOG_DIR}/%j-${job_name}.out" "--error=${SBATCH_LOG_DIR}/%j-${job_name}.err")
  sbatch_opts+=("${SBATCH_EXTRA_ARRAY[@]}")

  local job_id
  if ! job_id=$(
    MODEL_SIZE="${model_size}" \
    RUN_UUID="${run_uuid}" \
    STEP_VALUES="${step_values}" \
    STEP_INDICES="${step_indices}" \
    TOTAL_STEPS="${TOTAL_STEPS}" \
    JOB_IDX="${job_idx}" \
    TOTAL_JOBS="${total_jobs}" \
    PENDING_STEPS="${pending_steps}" \
    RUN_ONCE_SCRIPT="${RUN_ONCE_SCRIPT}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    VAL_STEPS="${VAL_STEPS}" \
    CHAIN_INDEX="${chain_index}" \
    CHAIN_DEPENDENCY="${dependency_job}" \
    sbatch "${sbatch_opts[@]}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "==================================================================="
echo "Eval run ${JOB_IDX}/${TOTAL_JOBS} | Model ${MODEL_SIZE} | Run ${RUN_UUID}"
echo "Node: $(hostname) | Time: $(date)"
echo "Evaluating ${PENDING_STEPS} checkpoint(s) sequentially on this worker."
echo "Chain index: ${CHAIN_INDEX:-0} | Dependency: ${CHAIN_DEPENDENCY:-none}"
echo "==================================================================="
IFS=',' read -r -a steps <<< "${STEP_VALUES}"
IFS=',' read -r -a step_indices <<< "${STEP_INDICES}"
for idx in "${!steps[@]}"; do
  step="${steps[idx]}"
  step_index="${step_indices[idx]}"
  echo "[EvalRun] ${MODEL_SIZE} | ${RUN_UUID} | step ${step} (${idx}/${PENDING_STEPS})"
  export EVAL_SWEEP_INDEX="${step_index}"
  "${RUN_ONCE_SCRIPT}" \
    --model-size "${MODEL_SIZE}" \
    --run-uuid "${RUN_UUID}" \
    --step "${step}" \
    --val-batch-size "${VAL_BATCH_SIZE}" \
    --val-steps "${VAL_STEPS}"
done
EOF
  ); then
    echo "Failed to submit sbatch job ${job_name}." >&2
    return 1
  fi

  echo "${job_id}"
}

total_jobs=${#RUN_MODELS[@]}
if [[ "${USE_SBATCH}" == "true" ]]; then
  dispatch_mode="sbatch (per-run, ${SBATCH_MAX_CHAINS} chain(s))"
else
  dispatch_mode="local"
fi
echo "Dispatching ${total_jobs} eval run(s) (${TOTAL_STEPS} checkpoint(s) per run) via ${dispatch_mode}."

job_counter=0
sbatch_dispatched=0
for spec in "${RUN_MODELS[@]}"; do
  read -r model_size run_uuid <<<"${spec}"
  if [[ -z "${model_size}" || -z "${run_uuid}" ]]; then
    echo "Invalid RUN_MODELS entry '${spec}'. Expected '<model_size> <run_uuid>'." >&2
    exit 1
  fi
  pending_steps=()
  pending_indices=()
  for idx in "${!STEP_ARRAY[@]}"; do
    step="${STEP_ARRAY[idx]}"
    eval_run_uuid="${run_uuid}_${idx}"
    if wandb_run_exists "${eval_run_uuid}"; then
      echo "[EvalLoop] Skipping ${model_size} | ${eval_run_uuid} (already on W&B)."
      continue
    fi
    pending_steps+=("${step}")
    pending_indices+=("${idx}")
  done
  if (( ${#pending_steps[@]} == 0 )); then
    echo "[EvalLoop] ${model_size} | ${run_uuid} already evaluated for all checkpoints."
    continue
  fi
  steps_csv=$(IFS=','; printf "%s" "${pending_steps[*]}")
  indices_csv=$(IFS=','; printf "%s" "${pending_indices[*]}")
  pending_count=${#pending_steps[@]}
  ((++job_counter))
  echo "[EvalLoop] ${job_counter}/${total_jobs}: ${model_size} | ${run_uuid} (${pending_count} pending step(s))"
  if [[ "${DRY_RUN}" == "true" ]]; then
    continue
  fi
  if [[ "${USE_SBATCH}" == "true" ]]; then
    chain_index=$((sbatch_dispatched % SBATCH_MAX_CHAINS))
    dependency_job=${SBATCH_CHAIN_LAST_IDS[chain_index]:-}
    if [[ -n "${dependency_job}" ]]; then
      echo "[EvalLoop][SBATCH chain ${chain_index}] Queue ${model_size} | ${run_uuid} after job ${dependency_job}." >&2
    else
      echo "[EvalLoop][SBATCH chain ${chain_index}] Queue ${model_size} | ${run_uuid} (no dependency)." >&2
    fi
    if ! job_id=$(submit_sbatch_run_job "${model_size}" "${run_uuid}" "${job_counter}" "${total_jobs}" "${steps_csv}" "${indices_csv}" "${pending_count}" "${dependency_job}" "${chain_index}"); then
      echo "Failed to submit sbatch job for ${model_size} | ${run_uuid}. Aborting." >&2
      exit 1
    fi
    SBATCH_CHAIN_LAST_IDS[chain_index]="${job_id}"
    SBATCH_JOB_IDS+=("${job_id}")
    ((++sbatch_dispatched))
  else
    run_steps_locally "${model_size}" "${run_uuid}" "${job_counter}" "${total_jobs}" "${steps_csv}" "${indices_csv}"
  fi
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[EvalLoop] Dry run complete; reviewed ${job_counter} run(s)." >&2
elif [[ "${USE_SBATCH}" == "true" ]]; then
  submitted_jobs=${#SBATCH_JOB_IDS[@]}
  echo "[EvalLoop] Submitted ${submitted_jobs} sbatch job(s) across ${SBATCH_MAX_CHAINS} chain(s)." >&2
  for ((chain_idx = 0; chain_idx < SBATCH_MAX_CHAINS; ++chain_idx)); do
    last_id=${SBATCH_CHAIN_LAST_IDS[chain_idx]:-}
    if [[ -n "${last_id}" ]]; then
      status_msg="last job ${last_id}"
    else
      status_msg="no jobs queued"
    fi
    echo "  - Chain ${chain_idx}: ${status_msg}" >&2
  done
else
  echo "[EvalLoop] Completed ${job_counter} run(s) locally." >&2
fi
