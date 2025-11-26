#!/usr/bin/env bash
# Sweep Muon (decay_lr buckets) and AdamW (embeddings/norms/other) learning rates over a fixed grid.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-muon-adam"}
LOG_RANK=${LOG_RANK:-0}

# Grid for Muon and AdamW LRs (AdamW anchored at 1e-3 first).
MUON_LR_VALUES=${MUON_LR_VALUES:-"5e-4 1e-3 2e-3 4e-3 8e-3 1.6e-2 3.2e-2"}
ADAMW_LR_VALUES=${ADAMW_LR_VALUES:-"1e-3 5e-4 2e-3 4e-3 8e-3 1.6e-2 3.2e-2"}
# FLAVORS=${FLAVORS:-"16M_VAL 16M 16M_RMS_NO_VAL_OUT 16M_VAL 16M_RMS_NO_OUT 16M_OUT 16M_RMS_NO_VAL 16M_VAL_OUT 16M_RMS"}
FLAVORS=${FLAVORS:-"16M_VAL"}

RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}
RUN_INDEX_RANGE=${RUN_INDEX_RANGE:-}
USE_SBATCH=${USE_SBATCH:-false}
DRY_RUN=${DRY_RUN:-false}
REVERSE_RUNS=${REVERSE_RUNS:-false}
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
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}

usage() {
  cat <<'EOF'
Usage: run_lr_grid.sh [OPTIONS] [-- extra training args]

Options mirror run_adamw_sweep.sh (range filtering, sbatch toggles, etc.).
Environment overrides:
  MUON_LR_VALUES (space-separated), ADAMW_LR_VALUES (space-separated),
  RUN_INDEX, RUN_INDEX_OFFSET, RUN_INDEX_RANGE, USE_SBATCH, DRY_RUN,
  REVERSE_RUNS (true/false to flip launch order),
  SBATCH_* controls, GPU_IDS for local launches.
EOF
}

TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --range)
      [[ $# -ge 2 ]] || { echo "Missing value for --range" >&2; exit 1; }
      RUN_INDEX_RANGE=$2; shift 2 ;;
    --run-index)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-index" >&2; exit 1; }
      RUN_INDEX=$2; shift 2 ;;
    --run-index-offset)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-index-offset" >&2; exit 1; }
      RUN_INDEX_OFFSET=$2; shift 2 ;;
    --sbatch) USE_SBATCH=true; shift ;;
    --no-sbatch) USE_SBATCH=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --reverse) REVERSE_RUNS=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; if [[ $# -gt 0 ]]; then TRAINING_ARGS+=("$@"); fi; break ;;
    *) TRAINING_ARGS+=("$1"); shift ;;
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
REVERSE_RUNS=$(normalize_bool "${REVERSE_RUNS}")

if ! [[ "${SBATCH_MAX_CHAINS}" =~ ^[0-9]+$ ]] || (( SBATCH_MAX_CHAINS < 1 )); then
  echo "SBATCH_MAX_CHAINS must be a positive integer (got ${SBATCH_MAX_CHAINS})." >&2
  exit 1
fi

USER_TRAINING_ARGS=("${TRAINING_ARGS[@]}")
serialize_training_args() {
  local formatted=""
  for arg in "${USER_TRAINING_ARGS[@]}"; do
    formatted+=" $(printf '%q' "$arg")"
  done
  echo "${formatted}"
}
TRAINING_ARGS_ESCAPED=$(serialize_training_args)
read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_ADDITIONAL_ARGS}"
if [[ -z "${SBATCH_ADDITIONAL_ARGS}" ]]; then
  SBATCH_EXTRA_ARRAY=()
fi

read -r -a MUON_ARRAY <<< "${MUON_LR_VALUES}"
read -r -a ADAM_ARRAY <<< "${ADAMW_LR_VALUES}"
read -r -a FLAVOR_ARRAY <<< "${FLAVORS}"

if (( ${#MUON_ARRAY[@]} == 0 )); then
  echo "MUON_LR_VALUES must contain at least one entry." >&2
  exit 1
fi
if (( ${#ADAM_ARRAY[@]} == 0 )); then
  echo "ADAMW_LR_VALUES must contain at least one entry." >&2
  exit 1
fi
if (( ${#FLAVOR_ARRAY[@]} == 0 )); then
  echo "FLAVORS must contain at least one entry." >&2
  exit 1
fi

TOTAL_RUNS=$(( ${#MUON_ARRAY[@]} * ${#ADAM_ARRAY[@]} * ${#FLAVOR_ARRAY[@]} ))

RANGE_ENABLED=false
RANGE_START=
RANGE_END=
if [[ -n "${RUN_INDEX_RANGE}" ]]; then
  if [[ "${RUN_INDEX_RANGE}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    RANGE_ENABLED=true
    RANGE_START=$((10#${BASH_REMATCH[1]}))
    RANGE_END=$((10#${BASH_REMATCH[2]}))
    if (( RANGE_END < RANGE_START )); then
      echo "RUN_INDEX_RANGE end must be >= start (got ${RUN_INDEX_RANGE})." >&2
      exit 1
    fi
  else
    echo "RUN_INDEX_RANGE must match START-END (e.g., 0-15)." >&2
    exit 1
  fi
fi

  if [[ -n "${RUN_INDEX}" ]]; then
    if ! [[ "${RUN_INDEX}" =~ ^[0-9]+$ ]]; then
      echo "RUN_INDEX must be a positive integer between 1 and ${TOTAL_RUNS}." >&2
      exit 1
    fi
  RUN_INDEX=$((10#${RUN_INDEX}))
  if (( RUN_INDEX < 1 || RUN_INDEX > TOTAL_RUNS )); then
    echo "RUN_INDEX must be between 1 and ${TOTAL_RUNS} (got ${RUN_INDEX})." >&2
    exit 1
  fi
fi

if ! [[ "${RUN_INDEX_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX_OFFSET must be a non-negative integer (got ${RUN_INDEX_OFFSET})." >&2
  exit 1
fi
RUN_INDEX_OFFSET=$((10#${RUN_INDEX_OFFSET}))

if [[ -n "${RUN_INDEX}" && RUN_INDEX_OFFSET -ne 0 ]]; then
  echo "RUN_INDEX and RUN_INDEX_OFFSET cannot both be set." >&2
  exit 1
fi

if [[ -n "${RUN_INDEX}" && "${RANGE_ENABLED}" == "true" ]]; then
  echo "RUN_INDEX and RUN_INDEX_RANGE cannot both be set." >&2
  exit 1
fi

if (( RUN_INDEX_OFFSET >= TOTAL_RUNS )); then
  echo "RUN_INDEX_OFFSET (${RUN_INDEX_OFFSET}) skips all ${TOTAL_RUNS} runs; nothing to do." >&2
  exit 0
fi

make_temp_config() {
  local muon_lr=$1
  local adam_lr=$2
  local flavor=$3
  mkdir -p "${SCRIPT_DIR}/tmp_configs"
  local tmp_file
  tmp_file=$(mktemp "${SCRIPT_DIR}/tmp_configs/muon_adamXXXXXX.toml")
  python3 - <<PY
import tomllib, tomli_w, sys, copy
from pathlib import Path

src = Path("${CONFIG_FILE}")
dst = Path("${tmp_file}")

with src.open("rb") as f:
    data = tomllib.load(f)

optim = copy.deepcopy(data.get("optimizer", {}))
composite = optim.get("composite")
if not isinstance(composite, list) or len(composite) < 2:
    sys.exit("optimizer.composite must contain at least two entries (Muon, AdamW).")

updated = []
for entry in composite:
    entry = copy.deepcopy(entry)
    name = entry.get("name")
    overrides = entry.get("config_overrides", {}) or {}
    if name == "Muon":
        overrides["lr"] = float("${muon_lr}")
    else:
        overrides["lr"] = float("${adam_lr}")
    entry["config_overrides"] = overrides
    updated.append(entry)

optim["composite"] = updated
data["optimizer"] = optim
data.setdefault("model", {})["flavor"] = "${flavor}"

with dst.open("wb") as f:
    tomli_w.dump(data, f)
PY
  echo "${tmp_file}"
}

should_run_combination() {
  local combo_index_1=$1
  local combo_index_0=$((combo_index_1 - 1))

  if (( combo_index_1 <= RUN_INDEX_OFFSET )); then
    return 1
  fi

  if [[ -n "${RUN_INDEX}" ]]; then
    if (( combo_index_1 == RUN_INDEX )); then
      return 0
    fi
    return 1
  fi

  if [[ "${RANGE_ENABLED}" == "true" ]]; then
    if (( combo_index_0 < RANGE_START )); then
      return 1
    fi
    if (( combo_index_0 > RANGE_END )); then
      return 1
    fi
  fi

  return 0
}

count_selected_runs() {
  local idx=0
  local selected=0
  for flavor in "${FLAVOR_ARRAY[@]}"; do
    for muon_lr in "${MUON_ARRAY[@]}"; do
      for adam_lr in "${ADAM_ARRAY[@]}"; do
        ((++idx))
        if should_run_combination "${idx}"; then
          ((++selected))
        fi
      done
    done
  done
  echo "${selected}"
}

SELECTED_RUNS=$(count_selected_runs)
if (( SELECTED_RUNS == 0 )); then
  echo "No runs match the requested filters; nothing to do." >&2
  exit 0
fi

declare -a RUN_PLAN_INDICES=()
declare -a RUN_PLAN_MUON=()
declare -a RUN_PLAN_ADAM=()
declare -a RUN_PLAN_FLAVOR=()

combination_index=0
for flavor in "${FLAVOR_ARRAY[@]}"; do
  for adam_lr in "${ADAM_ARRAY[@]}"; do
    for muon_lr in "${MUON_ARRAY[@]}"; do
      ((++combination_index))
      if ! should_run_combination "${combination_index}"; then
        continue
      fi
      RUN_PLAN_INDICES+=("${combination_index}")
      RUN_PLAN_MUON+=("${muon_lr}")
      RUN_PLAN_ADAM+=("${adam_lr}")
      RUN_PLAN_FLAVOR+=("${flavor}")
    done
  done
done

if [[ "${REVERSE_RUNS}" == "true" ]]; then
  declare -a REVERSED_INDICES=()
  declare -a REVERSED_MUON=()
  declare -a REVERSED_ADAM=()
  declare -a REVERSED_FLAVOR=()
  for ((rev_idx=${#RUN_PLAN_INDICES[@]}-1; rev_idx>=0; --rev_idx)); do
    REVERSED_INDICES+=("${RUN_PLAN_INDICES[rev_idx]}")
    REVERSED_MUON+=("${RUN_PLAN_MUON[rev_idx]}")
    REVERSED_ADAM+=("${RUN_PLAN_ADAM[rev_idx]}")
    REVERSED_FLAVOR+=("${RUN_PLAN_FLAVOR[rev_idx]}")
  done
  RUN_PLAN_INDICES=("${REVERSED_INDICES[@]}")
  RUN_PLAN_MUON=("${REVERSED_MUON[@]}")
  RUN_PLAN_ADAM=("${REVERSED_ADAM[@]}")
  RUN_PLAN_FLAVOR=("${REVERSED_FLAVOR[@]}")
fi

SELECTED_RUNS=${#RUN_PLAN_INDICES[@]}

SWEEP_CONFIG_STRING="muon_lr=${MUON_LR_VALUES}|adamw_lr=${ADAMW_LR_VALUES}|flavors=${FLAVORS}|config=${CONFIG_FILE}"
if command -v sha1sum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | sha1sum | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | shasum -a 1 | awk '{print $1}')
else
  SWEEP_HASH=$(
    SWEEP_CONFIG_STRING="${SWEEP_CONFIG_STRING}" python3 - <<'PY'
import hashlib
import os
config = os.environ["SWEEP_CONFIG_STRING"]
print(hashlib.sha1(config.encode("utf-8")).hexdigest())
PY
  )
fi
SWEEP_HASH=${SWEEP_HASH:0:8}

GPU_IDS=${GPU_IDS:-""}
RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
RDZV_BASE_PORT=${RDZV_BASE_PORT:-46000}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-46100}
PORT_STRIDE=${PORT_STRIDE:-4}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-5}

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
  exit 1
fi

GPU_IDS_AUTO_SOURCE=""
GPU_IDS_DEDUPED_COUNT=0
declare -a GPU_ARRAY=()

detect_available_gpu_ids() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local normalized=${CUDA_VISIBLE_DEVICES//,/ }
    local -a ids=()
    read -r -a ids <<< "${normalized}"
    if (( ${#ids[@]} > 0 )); then
      GPU_IDS_AUTO_SOURCE="CUDA_VISIBLE_DEVICES"
      printf "%s\n" "${ids[*]}"
      return 0
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local -a ids=()
    while IFS= read -r line; do
      line=${line//$'\r'/}
      line=${line//[[:space:]]/}
      [[ -z "${line}" ]] && continue
      ids+=("${line}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    if (( ${#ids[@]} > 0 )); then
      GPU_IDS_AUTO_SOURCE="nvidia-smi"
      printf "%s\n" "${ids[*]}"
      return 0
    fi
  fi

  GPU_IDS_AUTO_SOURCE=""
  return 1
}

set_gpu_array_from_string() {
  local raw="${1:-}"
  local cleaned=${raw//,/ }
  read -r -a parsed <<< "${cleaned}"
  local -A seen=()
  local -a unique=()
  GPU_IDS_DEDUPED_COUNT=0

  for token in "${parsed[@]}"; do
    token=${token//[[:space:]]/}
    [[ -z "${token}" ]] && continue
    if [[ ! "${token}" =~ ^[0-9]+$ ]]; then
      echo "GPU id '${token}' must be a non-negative integer. Provide GPU_IDS as digits or set GPU_IDS=auto." >&2
      return 1
    fi
    if [[ -n "${seen[$token]-}" ]]; then
      ((GPU_IDS_DEDUPED_COUNT++))
      continue
    fi
    unique+=("${token}")
    seen["$token"]=1
  done

  GPU_ARRAY=("${unique[@]}")
  return 0
}

declare -a GPU_PIDS=()
declare -a GPU_LABELS=()
declare -a ALL_PIDS=()
declare -a SBATCH_JOB_IDS=()
declare -a SBATCH_CHAIN_LAST_IDS=()
RUN_COUNTER=0

if [[ "${USE_SBATCH}" == "true" && "${DRY_RUN}" != "true" ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found; falling back to local multi-GPU execution." >&2
    USE_SBATCH="false"
  fi
fi

GPU_IDS_EFFECTIVE="${GPU_IDS:-}"
if [[ -z "${GPU_IDS_EFFECTIVE}" || "${GPU_IDS_EFFECTIVE,,}" == "auto" ]]; then
  if ! GPU_IDS_EFFECTIVE=$(detect_available_gpu_ids); then
    echo "GPU_IDS not provided (or set to 'auto') and automatic GPU detection failed. Set GPU_IDS explicitly." >&2
    exit 1
  fi
else
  GPU_IDS_AUTO_SOURCE=""
fi

if ! set_gpu_array_from_string "${GPU_IDS_EFFECTIVE}"; then
  exit 1
fi

TOTAL_GPUS=${#GPU_ARRAY[@]}
if [[ "${DRY_RUN}" != "true" && ${TOTAL_GPUS} -eq 0 ]]; then
  echo "GPU_IDS resolved to zero GPUs; provide GPU ids or set GPU_IDS=auto." >&2
  exit 1
fi

if [[ -n "${GPU_IDS_AUTO_SOURCE}" ]]; then
  echo "Auto-detected ${TOTAL_GPUS} GPU id(s) via ${GPU_IDS_AUTO_SOURCE}: ${GPU_ARRAY[*]}." >&2
fi
if (( GPU_IDS_DEDUPED_COUNT > 0 )); then
  echo "Removed ${GPU_IDS_DEDUPED_COUNT} duplicate GPU id(s); using GPU set: ${GPU_ARRAY[*]}." >&2
fi

if [[ "${USE_SBATCH}" == "true" ]]; then
  for ((chain_idx = 0; chain_idx < SBATCH_MAX_CHAINS; ++chain_idx)); do
    SBATCH_CHAIN_LAST_IDS[chain_idx]=''
  done
fi

if [[ -n "${SBATCH_LOG_DIR}" ]]; then
  mkdir -p "${SBATCH_LOG_DIR}"
fi
mkdir -p "${LOG_DIR}"

for idx in "${!GPU_ARRAY[@]}"; do
  GPU_PIDS[idx]=''
  GPU_LABELS[idx]=''
done

cleanup() {
  for pid in "${ALL_PIDS[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

wait_for_gpu() {
  while true; do
    for idx in "${!GPU_ARRAY[@]}"; do
      pid=${GPU_PIDS[idx]:-}
      if [[ -z "${pid}" ]]; then
        echo "${idx}"
        return
      fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        GPU_PIDS[idx]=''
        GPU_LABELS[idx]=''
        echo "${idx}"
        return
      fi
    done
    sleep "${POLL_INTERVAL_SEC}"
  done
}

sanitize_value() {
  local value=$1
  value=${value//./p}
  value=${value//-/m}
  value=${value//+/p}
  echo "${value}"
}

print_run_plan_table() {
  local total=$1
  if (( total == 0 )); then
    return
  fi
  echo "" >&2
  echo "Selected run configurations (${total} total):" >&2
  printf "%-10s %-10s %-12s %-12s %-18s\n" "Idx(1-based)" "Idx(0-based)" "muon_lr" "adamw_lr" "flavor" >&2
  printf "%-10s %-10s %-12s %-12s %-18s\n" "----------" "----------" "--------" "--------" "------------------" >&2
  for idx in "${!RUN_PLAN_INDICES[@]}"; do
    local combo_index_1=${RUN_PLAN_INDICES[idx]}
    local combo_index_0=$((combo_index_1 - 1))
    local muon_lr=${RUN_PLAN_MUON[idx]}
    local adam_lr=${RUN_PLAN_ADAM[idx]}
    local flavor=${RUN_PLAN_FLAVOR[idx]}
    printf "%-10s %-10s %-12s %-12s %-18s\n" "${combo_index_1}" "${combo_index_0}" "${muon_lr}" "${adam_lr}" "${flavor}" >&2
  done
  echo "" >&2
}

submit_sbatch_job() {
  local run_uuid=$1
  local run_progress=$2
  local rdzv_endpoint=$3
  local lighthouse_url=$4
  local combo_index_label=$5
  local muon_lr=$6
  local adam_lr=$7
  local cfg_path=$8
  local dependency_job=$9
  local chain_index=${10}
  local flavor=${11}

  local mu_tag
  mu_tag=$(sanitize_value "${muon_lr}")
  local ad_tag
  ad_tag=$(sanitize_value "${adam_lr}")

  local job_name="${RUN_PREFIX}-${flavor}-m${mu_tag}-a${ad_tag}-idx${combo_index_label}"
  local sbatch_opts=(--parsable "-c" "${SBATCH_CPUS_PER_TASK}" "--gres=gpu:${SBATCH_GPUS_PER_TASK}" "--job-name=${job_name}")
  [[ -n "${SBATCH_MEM}" ]] && sbatch_opts+=("--mem=${SBATCH_MEM}")
  [[ -n "${SBATCH_TIME}" ]] && sbatch_opts+=("--time=${SBATCH_TIME}")
  [[ -n "${SBATCH_PARTITION}" ]] && sbatch_opts+=("--partition=${SBATCH_PARTITION}")
  [[ -n "${SBATCH_ACCOUNT}" ]] && sbatch_opts+=("--account=${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_QOS}" ]] && sbatch_opts+=("--qos=${SBATCH_QOS}")
  [[ -n "${SBATCH_CONSTRAINT}" ]] && sbatch_opts+=("--constraint=${SBATCH_CONSTRAINT}")
  [[ -n "${SBATCH_COMMENT}" ]] && sbatch_opts+=("--comment=${SBATCH_COMMENT}")
  [[ -n "${SBATCH_NODE}" ]] && sbatch_opts+=("-w" "${SBATCH_NODE}")
  [[ -n "${dependency_job}" ]] && sbatch_opts+=("--dependency=afterany:${dependency_job}")
  if [[ -n "${SBATCH_LOG_DIR}" ]]; then
    sbatch_opts+=("--output=${SBATCH_LOG_DIR}/%j-${job_name}.out" "--error=${SBATCH_LOG_DIR}/%j-${job_name}.err")
  fi
  sbatch_opts+=("${SBATCH_EXTRA_ARRAY[@]}")

  local job_id
  if ! job_id=$(sbatch "${sbatch_opts[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "==================================================================="
echo "STARTING SBATCH JOB: ${job_name} (ID: \$SLURM_JOB_ID)"
echo "Run UUID: ${run_uuid} | Progress: ${run_progress} | muon_lr=${muon_lr} | adamw_lr=${adam_lr}"
echo "Chain Index: ${chain_index} | Dependency: ${dependency_job:-none}"
echo "Node: \$(hostname) at \$(date)"
echo "==================================================================="
      export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
      export RUN_UUID="${run_uuid}"
export WANDB_PROJECT=\${WANDB_PROJECT:-"torchtitan_tune_muon"}
export WANDB_TEAM=\${WANDB_TEAM:-"camlsys"}
export WANDB_RUN_NAME="${run_uuid}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=\${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'

uv run --no-sync torchrun \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${rdzv_endpoint}" \
  --rdzv_id "${run_uuid}" \
  --local-ranks-filter="${LOG_RANK}" \
  --role rank \
  --tee 3 \
  -m "${TRAIN_MODULE}" \
  --job.config_file "${cfg_path}" \
  --model.flavor "${flavor}" \
  ${TRAINING_ARGS_ESCAPED} |& tee "${LOG_DIR}/${run_uuid}.log"
echo "JOB FINISHED: \$(date)"
EOF
); then
    echo "Failed to submit sbatch job ${job_name}." >&2
    return 1
  fi

  echo "${job_id}"
}

if [[ "${USE_SBATCH}" == "true" ]]; then
  EXECUTION_DESC="sbatch submission"
else
  EXECUTION_DESC="${#GPU_ARRAY[@]} GPU slot(s)"
fi

if [[ -n "${RUN_INDEX}" ]]; then
  FILTER_DESC="single run ${RUN_INDEX}/${TOTAL_RUNS}"
elif [[ "${RANGE_ENABLED}" == "true" ]]; then
  FILTER_DESC="range ${RANGE_START}-${RANGE_END} (0-indexed)"
elif (( RUN_INDEX_OFFSET > 0 )); then
  start_index=$((RUN_INDEX_OFFSET + 1))
  FILTER_DESC="starting at run ${start_index} (1-indexed)"
else
  FILTER_DESC="all runs"
fi

echo "Starting Muon+AdamW LR sweep hash=${SWEEP_HASH} using ${EXECUTION_DESC}: ${SELECTED_RUNS}/${TOTAL_RUNS} run(s) selected (${FILTER_DESC})." >&2
print_run_plan_table "${SELECTED_RUNS}"

timestamp_global=$(date +"%Y%m%d-%H%M%S")

dispatched_runs=0
for idx in "${!RUN_PLAN_INDICES[@]}"; do
    combination_index=${RUN_PLAN_INDICES[idx]}
    combination_index_zero=$((combination_index - 1))
    muon_lr=${RUN_PLAN_MUON[idx]}
    adam_lr=${RUN_PLAN_ADAM[idx]}
    flavor=${RUN_PLAN_FLAVOR[idx]}
    muon_tag=$(sanitize_value "${muon_lr}")
    adam_tag=$(sanitize_value "${adam_lr}")
    run_uuid="${RUN_PREFIX}-${flavor}-mu${muon_tag}-ad${adam_tag}-h${SWEEP_HASH}-t${timestamp_global}-i${combination_index_zero}"
    run_progress="run ${combination_index}/${TOTAL_RUNS}"
    cfg_path=$(make_temp_config "${muon_lr}" "${adam_lr}" "${flavor}")

    if [[ "${DRY_RUN}" == "true" ]]; then
      if [[ "${USE_SBATCH}" == "true" ]]; then
        echo "[DRY-RUN][SBATCH] ${run_uuid} (muon_lr=${muon_lr}, adamw_lr=${adam_lr}) [${run_progress}]" >&2
      else
        echo "[DRY-RUN][LOCAL] ${run_uuid} (muon_lr=${muon_lr}, adamw_lr=${adam_lr}) [${run_progress}]" >&2
      fi
      continue
    fi

    launch_index=$((RUN_COUNTER + 1))
    rdzv_port=$((RDZV_BASE_PORT + launch_index * PORT_STRIDE))
    lighthouse_port=$((LIGHTHOUSE_BASE_PORT + launch_index * PORT_STRIDE))
    rdzv_endpoint="${RDZV_HOST}:${rdzv_port}"
    lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"
    RUN_COUNTER=${launch_index}

    if [[ "${USE_SBATCH}" == "true" ]]; then
      chain_index=$((dispatched_runs % SBATCH_MAX_CHAINS))
      dependency_job=${SBATCH_CHAIN_LAST_IDS[chain_index]:-}
      if [[ -n "${dependency_job}" ]]; then
        echo "[SBATCH] Chain ${chain_index}: submitting ${run_uuid} after job ${dependency_job}." >&2
      else
        echo "[SBATCH] Chain ${chain_index}: submitting ${run_uuid} with no dependency." >&2
      fi
      job_id=$(submit_sbatch_job "${run_uuid}" "${run_progress}" "${rdzv_endpoint}" "${lighthouse_url}" "${combination_index_zero}" "${muon_lr}" "${adam_lr}" "${cfg_path}" "${dependency_job}" "${chain_index}" "${flavor}")
      if [[ -z "${job_id}" ]]; then
        echo "Aborting sweep after failed sbatch submission." >&2
        exit 1
      fi
      SBATCH_CHAIN_LAST_IDS[chain_index]="${job_id}"
      SBATCH_JOB_IDS+=("${job_id}")
      echo "[SBATCH] Submitted ${run_uuid} (${run_progress}) as job ${job_id} | muon_lr=${muon_lr}, adamw_lr=${adam_lr}" >&2
      sleep 5
    else
      gpu_slot=$(wait_for_gpu)
      gpu_id=${GPU_ARRAY[gpu_slot]}
      echo "[GPU ${gpu_id}] Launching run ${run_uuid} (muon_lr=${muon_lr}, adamw_lr=${adam_lr}) [${run_progress}]" >&2
      echo "           torchrun rendezvous: ${rdzv_endpoint}, lighthouse: ${lighthouse_url}" >&2

      (
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
        export RUN_UUID="${run_uuid}"
        export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_tune_Lr"}
        export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
        export WANDB_RUN_NAME="${run_uuid}"
        export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
        export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'

uv run --no-sync torchrun \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${rdzv_endpoint}" \
  --rdzv_id "${run_uuid}" \
  --local-ranks-filter="${LOG_RANK}" \
  --role rank \
  --tee 3 \
  -m "${TRAIN_MODULE}" \
  --job.config_file "${cfg_path}" \
  --model.flavor "${flavor}" \
          "${USER_TRAINING_ARGS[@]}"
      ) |& tee "${LOG_DIR}/${run_uuid}.log" &

      pid=$!
      GPU_PIDS[gpu_slot]=$pid
      GPU_LABELS[gpu_slot]="${run_uuid}"
      ALL_PIDS+=($pid)
    fi

    ((++dispatched_runs))
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run complete: ${SELECTED_RUNS} run(s) matched the filters (no jobs launched)." >&2
elif [[ "${USE_SBATCH}" == "true" ]]; then
  echo "Submitted ${dispatched_runs} job(s) via sbatch (hash ${SWEEP_HASH})." >&2
else
  echo "All ${dispatched_runs} job(s) queued; waiting for completion." >&2
  for pid in "${GPU_PIDS[@]}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ ${#ALL_PIDS[@]} -gt 0 ]]; then
    wait "${ALL_PIDS[@]}" 2>/dev/null || true
  fi
fi

echo "Sweep complete." >&2
