#!/usr/bin/env bash
# Sweep GaLore ranks and learning rates locally without requiring Slurm.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
RUN_PREFIX=${RUN_PREFIX:-"icml2026-720M-ddp-qhm"}
LOG_RANK=${LOG_RANK:-0}

PROJECTION_RANKS=${PROJECTION_RANKS:-"256 2048"}
LR_VALUES=${LR_VALUES:-"0.008 0.008"}
ROTATE_MOMENTS_OPTIONS=${ROTATE_MOMENTS_OPTIONS:-"true"}
SWITCH_SCALES=${SWITCH_SCALES:-"1.0 1.0"}

VS_VALUES=${VS_VALUES:-"0.91 0.94"}
QHM_OUTSIDE_OPTIONS=${QHM_OUTSIDE_OPTIONS:-"false"}
ADAM_SENTINEL_RANK=${ADAM_SENTINEL_RANK:-2048}
RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}
RUN_INDEX_RANGE=${RUN_INDEX_RANGE:-}
DRY_RUN=${DRY_RUN:-false}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-"${SCRIPT_DIR}/logs"}

usage() {
  cat <<'EOF'
Usage: run_adamw_sweep.sh [OPTIONS] [-- extra training args]

Options:
  --range START-END        Run only the 0-indexed inclusive range of sweep jobs.
  --run-index INDEX        Run only the 1-indexed job at INDEX.
  --run-index-offset N     Skip the first N jobs before applying other filters.
  --dry-run                Print matching runs without launching anything.
  -h, --help               Show this message.
  --                       Treat the remaining arguments as training args.

Environment variables (override defaults):
  RUN_INDEX, RUN_INDEX_OFFSET, RUN_INDEX_RANGE, DRY_RUN
  NPROC_PER_NODE (number of local processes to launch via torchrun)
  LOCAL_LOG_DIR (where to store per-run logs)
  PROJECTION_RANKS, LR_VALUES, ROTATE_MOMENTS_OPTIONS, VS_VALUES, QHM_OUTSIDE_OPTIONS, ADAM_SENTINEL_RANK
EOF
}

TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --range)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --range" >&2
        exit 1
      fi
      RUN_INDEX_RANGE=$2
      shift 2
      ;;
    --run-index)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --run-index" >&2
        exit 1
      fi
      RUN_INDEX=$2
      shift 2
      ;;
    --run-index-offset)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --run-index-offset" >&2
        exit 1
      fi
      RUN_INDEX_OFFSET=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
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

normalize_bool() {
  local value="${1:-}"
  case "${value,,}" in
    true|1|yes|on) echo "true" ;;
    false|0|no|off|"") echo "false" ;;
    *) echo "${value,,}" ;;
  esac
}

DRY_RUN=$(normalize_bool "${DRY_RUN}")

if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 )); then
  echo "NPROC_PER_NODE must be a positive integer (got ${NPROC_PER_NODE})." >&2
  exit 1
fi

USER_TRAINING_ARGS=("${TRAINING_ARGS[@]}")

read -r -a PROJECTION_RANK_ARRAY <<< "${PROJECTION_RANKS}"
read -r -a LR_ARRAY <<< "${LR_VALUES}"
read -r -a SWITCH_SCALE_ARRAY <<< "${SWITCH_SCALES}"
read -r -a VS_ARRAY <<< "${VS_VALUES}"

# Broadcast single switch_scale or vs value across all ranks if needed
if (( ${#SWITCH_SCALE_ARRAY[@]} == 1 )) && (( ${#PROJECTION_RANK_ARRAY[@]} > 1 )); then
  switch_broadcast=${SWITCH_SCALE_ARRAY[0]}
  SWITCH_SCALE_ARRAY=()
  for _ in "${PROJECTION_RANK_ARRAY[@]}"; do
    SWITCH_SCALE_ARRAY+=("${switch_broadcast}")
  done
fi
if (( ${#VS_ARRAY[@]} == 1 )) && (( ${#PROJECTION_RANK_ARRAY[@]} > 1 )); then
  vs_broadcast=${VS_ARRAY[0]}
  VS_ARRAY=()
  for _ in "${PROJECTION_RANK_ARRAY[@]}"; do
    VS_ARRAY+=("${vs_broadcast}")
  done
fi

# Use a fixed LR per rank: lengths must match one-to-one
if (( ${#LR_ARRAY[@]} != ${#PROJECTION_RANK_ARRAY[@]} )); then
  echo "When using fixed LR per rank, PROJECTION_RANKS and LR_VALUES must have the same number of entries." >&2
  exit 1
fi
# Enforce aligned sweeps: switch_scales and vs_values must match proj ranks length
if (( ${#SWITCH_SCALE_ARRAY[@]} != ${#PROJECTION_RANK_ARRAY[@]} )); then
  echo "SWITCH_SCALES must have the same number of entries as PROJECTION_RANKS for index-aligned runs." >&2
  exit 1
fi
if (( ${#VS_ARRAY[@]} != ${#PROJECTION_RANK_ARRAY[@]} )); then
  echo "VS_VALUES must have the same number of entries as PROJECTION_RANKS for index-aligned runs." >&2
  exit 1
fi

# Force rotate to true for all runs (ignore ROTATE_MOMENTS_OPTIONS input)
ROTATE_MOMENTS_OPTIONS="true"
ROTATE_MOMENTS_ARRAY=("true")

read -r -a QHM_OUTSIDE_ARRAY_RAW <<< "${QHM_OUTSIDE_OPTIONS}"
QHM_OUTSIDE_ARRAY=()
for qopt in "${QHM_OUTSIDE_ARRAY_RAW[@]}"; do
  normalized_q=$(normalize_bool "${qopt}")
  if [[ "${normalized_q}" != "true" && "${normalized_q}" != "false" ]]; then
    echo "QHM_OUTSIDE_OPTIONS entries must be boolean strings (got ${qopt})." >&2
    exit 1
  fi
  QHM_OUTSIDE_ARRAY+=("${normalized_q}")
done

if (( ${#PROJECTION_RANK_ARRAY[@]} == 0 )); then
  echo "PROJECTION_RANKS must contain at least one entry." >&2
  exit 1
fi
if (( ${#LR_ARRAY[@]} == 0 )); then
  echo "LR_VALUES must contain at least one entry." >&2
  exit 1
fi

if (( ${#SWITCH_SCALE_ARRAY[@]} == 0 )); then
  echo "SWITCH_SCALES must contain at least one entry." >&2
  exit 1
fi
if (( ${#VS_ARRAY[@]} == 0 )); then
  echo "VS_VALUES must contain at least one entry." >&2
  exit 1
fi
if (( ${#QHM_OUTSIDE_ARRAY[@]} == 0 )); then
  echo "QHM_OUTSIDE_OPTIONS must contain at least one entry." >&2
  exit 1
fi
# QHM outside flag is applied uniformly across all runs (no sweep)
if (( ${#QHM_OUTSIDE_ARRAY[@]} > 1 )); then
  echo "QHM_OUTSIDE_OPTIONS should provide a single value; a uniform setting is applied to all runs." >&2
  exit 1
fi
QHM_OUTSIDE_VALUE=${QHM_OUTSIDE_ARRAY[0]:-false}

ADAM_SENTINEL_RANK_INT=""
if [[ -n "${ADAM_SENTINEL_RANK}" ]]; then
  if [[ "${ADAM_SENTINEL_RANK}" =~ ^[0-9]+$ ]]; then
    ADAM_SENTINEL_RANK_INT=$((10#${ADAM_SENTINEL_RANK}))
  else
    echo "ADAM_SENTINEL_RANK must be a non-negative integer (got ${ADAM_SENTINEL_RANK})." >&2
    exit 1
  fi
fi

# TOTAL runs is product of ranks x switch_scales x vs x qhm options x rotate options
# TOTAL runs is index-aligned across rank/lr/switch_scale/vs; rotate and qhm_outside are fixed
TOTAL_RUNS=${#PROJECTION_RANK_ARRAY[@]}

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
    echo "RUN_INDEX_RANGE must match START-END (e.g., 0-63)." >&2
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
  for i in "${!PROJECTION_RANK_ARRAY[@]}"; do
    ((++idx))
    if should_run_combination "${idx}"; then
      ((++selected))
    fi
  done
  echo "${selected}"
}

SELECTED_RUNS=$(count_selected_runs)
if (( SELECTED_RUNS == 0 )); then
  echo "No runs match the requested filters; nothing to do." >&2
  exit 0
fi

declare -a RUN_PLAN_INDICES=()
declare -a RUN_PLAN_RANKS=()
declare -a RUN_PLAN_LRS=()
declare -a RUN_PLAN_ROTATES=()
declare -a RUN_PLAN_SWITCH_SCALES=()
declare -a RUN_PLAN_NEW_VS=()
declare -a RUN_PLAN_QHM_OUTSIDE=()

combination_index=0
# Build runs by index: rank/lr/switch_scale/vs are index-aligned; qhm_outside and rotate are fixed
for i in "${!PROJECTION_RANK_ARRAY[@]}"; do
  rank=${PROJECTION_RANK_ARRAY[i]}
  lr=${LR_ARRAY[i]}
  switch_scale=${SWITCH_SCALE_ARRAY[i]}
  new_v=${VS_ARRAY[i]}
  rotate_flag=${ROTATE_MOMENTS_ARRAY[0]}
  qhm=${QHM_OUTSIDE_VALUE}

  ((++combination_index))
  if ! should_run_combination "${combination_index}"; then
    continue
  fi
  RUN_PLAN_INDICES+=("${combination_index}")
  RUN_PLAN_RANKS+=("${rank}")
  RUN_PLAN_LRS+=("${lr}")
  RUN_PLAN_ROTATES+=("${rotate_flag}")
  RUN_PLAN_SWITCH_SCALES+=("${switch_scale}")
  RUN_PLAN_NEW_VS+=("${new_v}")
  RUN_PLAN_QHM_OUTSIDE+=("${qhm}")
done

SELECTED_RUNS=${#RUN_PLAN_INDICES[@]}

SWEEP_CONFIG_STRING="proj_ranks=${PROJECTION_RANKS}|lrs=${LR_VALUES}|rotate=${ROTATE_MOMENTS_OPTIONS}|train_module=${TRAIN_MODULE}|config=${CONFIG_FILE}"
SWEEP_CONFIG_STRING="proj_ranks=${PROJECTION_RANKS}|lrs=${LR_VALUES}|rotate=${ROTATE_MOMENTS_OPTIONS}|switch_scales=${SWITCH_SCALES}|vs=${VS_VALUES}|qhm=${QHM_OUTSIDE_OPTIONS}|train_module=${TRAIN_MODULE}|config=${CONFIG_FILE}"
if command -v sha1sum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | sha1sum | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | shasum -a 1 | awk '{print $1}')
elif command -v md5sum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | md5sum | awk '{print $1}')
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

RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
RDZV_BASE_PORT=${RDZV_BASE_PORT:-30000}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-29610}
PORT_STRIDE=${PORT_STRIDE:-4}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}
GALORE_REGEX_PATTERN=${GALORE_REGEX_PATTERN:-"attention\\.w[qkv]|attention\\.wo|feed_forward\\.w[12]"}
GENERATED_CONFIG_DIR=${GENERATED_CONFIG_DIR:-"${SCRIPT_DIR}/generated_configs"}
NEW_V=${NEW_V:-1.0}

generate_run_config() {
  local run_uuid=$1
  local target_rank=$2
  local rotate_flag=$3
  local switch_scale=${4:-}
  local new_v_run=${5:-}
  local qhm_outside=${6:-}
  local lr_value=${7:-}
  local output_path="${GENERATED_CONFIG_DIR}/${run_uuid}.toml"

  BASE_CONFIG_PATH="${CONFIG_FILE}" \
  OUTPUT_CONFIG_PATH="${output_path}" \
  SWEEP_REGEX_PATTERN="${GALORE_REGEX_PATTERN}" \
  TARGET_RANK="${target_rank}" \
  ROTATE_MOMENTS_FLAG="${rotate_flag}" \
  ADAM_SENTINEL_RANK="${ADAM_SENTINEL_RANK}" \
  NEW_V="${NEW_V}" \
  NEW_V_RUN="${new_v_run}" \
  SWITCH_SCALE="${switch_scale}" \
  QHM_OUTSIDE="${qhm_outside}" \
  LR_VALUE="${lr_value}" \
  uv run --no-sync python3  <<'PY'
import os
import sys
from copy import deepcopy
from pathlib import Path

try:  # Python >=3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older versions
    import tomli as tomllib  # type: ignore

import tomli_w

base = Path(os.environ["BASE_CONFIG_PATH"])
output = Path(os.environ["OUTPUT_CONFIG_PATH"])
pattern = os.environ["SWEEP_REGEX_PATTERN"]
rank = int(os.environ["TARGET_RANK"])
rotate_flag = os.environ["ROTATE_MOMENTS_FLAG"].strip().lower()
adam_rank_env = os.environ.get("ADAM_SENTINEL_RANK", "").strip()
full_rank_threshold = int(adam_rank_env) if adam_rank_env else None

true_values = {"true", "1", "yes", "on"}
false_values = {"false", "0", "no", "off", ""}
if rotate_flag in true_values:
  rotate_moments = True
elif rotate_flag in false_values:
  rotate_moments = False
else:
  raise ValueError(f"Unsupported boolean for rotate_moments_on_refresh: {rotate_flag}")

data = tomllib.loads(base.read_text(encoding="utf-8"))
optimizer = data.setdefault("optimizer", {})

is_full_rank = full_rank_threshold is not None and rank >= full_rank_threshold
if is_full_rank:
  optimizer["name"] = "GaLore"
  # Disable low-rank projections for sentinel ranks by clearing all GaLore-specific fields.
  optimizer.pop("galore_param_regexes", None)
  for key in list(optimizer.keys()):
    if key.startswith("galore_"):
      optimizer.pop(key, None)
else:
  optimizer["name"] = "GaLore"
  optimizer["galore_rotate_moments_on_refresh"] = rotate_moments
  optimizer["galore_use_error_feedback"] = True
  regex_entries = optimizer.get("galore_param_regexes")
  normalized: list[dict] = []
  if isinstance(regex_entries, (list, tuple)):
      for entry in regex_entries:
          if isinstance(entry, dict):
              normalized.append(deepcopy(entry))
          else:
              try:
                  normalized.append(dict(entry))
              except Exception:
                  continue
  elif regex_entries is None:
      normalized = []
  else:
      normalized = [dict(regex_entries)] if isinstance(regex_entries, dict) else []

  updated = False
  for entry in normalized:
    if entry.get("param_str_match") == pattern:
      entry["rank"] = rank
      updated = True
      break

  if not updated:
    normalized.append({"param_str_match": pattern, "rank": rank})

  optimizer["galore_param_regexes"] = normalized

# Insert or update a fl_metrics.hyperparameter_switch entry to trigger at step 2048
# Prefer per-run NEW_V_RUN if provided, otherwise fall back to global NEW_V
new_v_env = os.environ.get("NEW_V_RUN", os.environ.get("NEW_V", "1.0"))
try:
  new_v = float(new_v_env)
except Exception:
  raise ValueError(f"NEW_V must be numeric (got {new_v_env})")

# Optional switch_scale passed per-run
switch_scale_env = os.environ.get("SWITCH_SCALE", None)
if switch_scale_env is not None and switch_scale_env.strip() != "":
  try:
    switch_scale_val = float(switch_scale_env)
  except Exception:
    raise ValueError(f"SWITCH_SCALE must be numeric (got {switch_scale_env})")
else:
  switch_scale_val = None

# Optional qhm_outside flag passed per-run
qhm_env = os.environ.get("QHM_OUTSIDE", "").strip().lower()
if qhm_env in true_values:
  qhm_bool = True
elif qhm_env in false_values:
  qhm_bool = False
elif qhm_env == "":
  qhm_bool = None
else:
  raise ValueError(f"Unsupported boolean for QHM_OUTSIDE: {qhm_env}")

fl_metrics = data.setdefault("fl_metrics", {})
# Use the expected schema for hyperparameter_switch
# [fl_metrics.hyperparameter_switch]
# enabled = false
# steps = []
# new_vs = []
# new_betas = []
# reset_momenta = []
hp_switch = fl_metrics.get("hyperparameter_switch")
if hp_switch is None:
  fl_metrics["hyperparameter_switch"] = {
    "enabled": True,
    "steps": [2048],
    "new_vs": [new_v],
    "new_betas": [0.999, 0.999],
    "reset_momenta": ["exp_avg", "exp_avg_sq"],
  }
  hp_switch = fl_metrics["hyperparameter_switch"]
else:
  # update fields conservatively to the expected schema
  hp_switch["enabled"] = True
  hp_switch["steps"] = [2048]
  hp_switch["new_vs"] = [new_v]
  hp_switch["new_betas"] = [0.999, 0.999]
  hp_switch["reset_momenta"] = ["exp_avg", "exp_avg_sq"]

# If a per-run switch_scale was provided, store it under the hyperparameter_switch
if switch_scale_val is not None:
  # Also apply switch_scale to lr_scheduler config so scheduler sees the value
  lr_scheduler = data.setdefault("lr_scheduler", {})
  lr_scheduler["switch_scale"] = switch_scale_val

# If qhm_outside flag provided, add to optimizer config
if (not is_full_rank) and qhm_bool is not None:
  optimizer["galore_qhm_outside_projection"] = qhm_bool
# If a per-run LR value was provided, write it into the optimizer section
lr_env = os.environ.get("LR_VALUE", "")
if lr_env is not None and lr_env.strip() != "":
  try:
    optimizer["lr"] = float(lr_env)
  except Exception:
    optimizer["lr"] = lr_env
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(tomli_w.dumps(data), encoding="utf-8")
PY

  echo "${output_path}"
}



if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
  exit 1
fi

RUN_COUNTER=0

sanitize_value() {
  local value=$1
  value=${value//./p}
  value=${value//-/m}
  value=${value//+/p}
  value=${value// /}
  echo "${value}"
}

print_run_plan_table() {
  local total=$1
  if (( total == 0 )); then
    return
  fi
  echo "" >&2
  echo "Selected run configurations (${total} total):" >&2
  printf "% -10s % -10s % -8s % -10s % -8s % -12s % -8s % -12s\n" "Idx(1-based)" "Idx(0-based)" "rank" "lr" "rotate" "switch_scale" "new_v" "qhm_outside" >&2
  printf "% -10s % -10s % -8s % -10s % -8s % -12s % -8s % -12s\n" "----------" "----------" "----" "----" "------" "------------" "-----" "-----------" >&2
  for idx in "${!RUN_PLAN_INDICES[@]}"; do
    local combo_index_1=${RUN_PLAN_INDICES[idx]}
    local combo_index_0=$((combo_index_1 - 1))
    local rank=${RUN_PLAN_RANKS[idx]}
    local lr=${RUN_PLAN_LRS[idx]}
    local rotate_flag=${RUN_PLAN_ROTATES[idx]}
    local switch_scale=${RUN_PLAN_SWITCH_SCALES[idx]:-}
    local new_v=${RUN_PLAN_NEW_VS[idx]:-}
    local qhm_outside=${RUN_PLAN_QHM_OUTSIDE[idx]:-}
    printf "% -10s % -10s % -8s % -10s % -8s % -12s % -8s % -12s\n" "${combo_index_1}" "${combo_index_0}" "${rank}" "${lr}" "${rotate_flag}" "${switch_scale}" "${new_v}" "${qhm_outside}" >&2
  done
  echo "" >&2
}

run_local_job() {
  local run_uuid=$1
  local run_progress=$2
  local rdzv_endpoint=$3
  local lighthouse_url=$4
  local combo_index_label=$5
  local proj_rank=$6
  local lr_value=$7
  local rotate_flag=$8
  local run_config_path=$9

  local optimizer_name="GaLore"
  local log_file=""
  if [[ -n "${LOCAL_LOG_DIR}" ]]; then
    mkdir -p "${LOCAL_LOG_DIR}"
    log_file="${LOCAL_LOG_DIR}/${run_uuid}.log"
  fi

  echo "==================================================================="
  echo "STARTING LOCAL JOB: ${run_uuid}"
  echo "Run UUID: ${run_uuid} | Progress: ${run_progress} | rank=${proj_rank} | lr=${lr_value} | rotate=${rotate_flag}"
  echo "Local launch index: ${combo_index_label} | Logs: ${log_file:-stdout}"
  echo "Node: $(hostname) at $(date)"
  echo "==================================================================="

  export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
  export RUN_UUID="${run_uuid}"
  export WANDB_PROJECT=${WANDB_PROJECT:-"galore-tune-lr"}
  export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
  export WANDB_RUN_NAME="${run_uuid}"
  export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
  export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}

  local cmd=(
    uv run --no-sync torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --rdzv_backend=c10d
    --rdzv_endpoint="${rdzv_endpoint}"
    --rdzv_id "${run_uuid}"
    --local-ranks-filter="${LOG_RANK}"
    --role rank
    --tee 3
    -m "${TRAIN_MODULE}"
    --job.config_file "${run_config_path}"
    --optimizer.builder mosaic
    --optimizer.name "${optimizer_name}"
    --optimizer.lr "${lr_value}"
    --training.global_batch_size 256
    --training.local_batch_size 16
    --training.steps 6144
  )

  if (( ${#USER_TRAINING_ARGS[@]} )); then
    cmd+=("${USER_TRAINING_ARGS[@]}")
  fi

  if [[ -n "${log_file}" ]]; then
    # shellcheck disable=SC2068
    { set -o pipefail; "${cmd[@]}" 2>&1 | tee -a "${log_file}"; }
  else
    "${cmd[@]}"
  fi
}

EXECUTION_DESC="local execution"

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

echo "Starting GaLore sweep hash=${SWEEP_HASH} using ${EXECUTION_DESC}: ${SELECTED_RUNS}/${TOTAL_RUNS} run(s) selected (${FILTER_DESC})." >&2
print_run_plan_table "${SELECTED_RUNS}"

timestamp_global=$(date +"%Y%m%d-%H%M%S")

dispatched_runs=0
for idx in "${!RUN_PLAN_INDICES[@]}"; do
    combination_index=${RUN_PLAN_INDICES[idx]}
    combination_index_zero=$((combination_index - 1))
    proj_rank=${RUN_PLAN_RANKS[idx]}
    lr_value=${RUN_PLAN_LRS[idx]}
    rotate_flag=${RUN_PLAN_ROTATES[idx]}

    rank_label=$(sanitize_value "${proj_rank}")
    lr_label=$(sanitize_value "${lr_value}")
    rotate_label=$(sanitize_value "${rotate_flag}")
    switch_scale_val=${RUN_PLAN_SWITCH_SCALES[idx]:-}
    new_v_val=${RUN_PLAN_NEW_VS[idx]:-}
    qhm_outside_val=${RUN_PLAN_QHM_OUTSIDE[idx]:-}
    switch_label=$(sanitize_value "${switch_scale_val}")
    new_v_label=$(sanitize_value "${new_v_val}")
    qhm_label=$(sanitize_value "${qhm_outside_val}")
    run_uuid="${RUN_PREFIX}-${SWEEP_HASH}-r${rank_label}-lr${lr_label}-rot${rotate_label}-ss${switch_label}-v${new_v_label}-q${qhm_label}-${timestamp_global}-idx${combination_index_zero}"
    run_progress="run ${combination_index}/${TOTAL_RUNS}"

    if [[ "${DRY_RUN}" != "true" ]]; then
      mkdir -p "${GENERATED_CONFIG_DIR}"
      run_config_path=$(generate_run_config "${run_uuid}" "${proj_rank}" "${rotate_flag}" "${switch_scale_val}" "${new_v_val}" "${qhm_outside_val}" "${lr_value}")
    else
      run_config_path="${CONFIG_FILE}"
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[DRY-RUN][LOCAL] ${run_uuid} (rank=${proj_rank}, lr=${lr_value}, rotate=${rotate_flag}) [${run_progress}]" >&2
      continue
    fi

    launch_index=$((RUN_COUNTER + 1))
    rdzv_port=$((RDZV_BASE_PORT + launch_index * PORT_STRIDE))
    lighthouse_port=$((LIGHTHOUSE_BASE_PORT + launch_index * PORT_STRIDE))
    rdzv_endpoint="${RDZV_HOST}:${rdzv_port}"
    lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"
    RUN_COUNTER=${launch_index}

    echo "[LOCAL] Launching ${run_uuid} (${run_progress}) with rank=${proj_rank}, lr=${lr_value}, rotate=${rotate_flag}" >&2
    run_local_job "${run_uuid}" "${run_progress}" "${rdzv_endpoint}" "${lighthouse_url}" "${combination_index_zero}" "${proj_rank}" "${lr_value}" "${rotate_flag}" "${run_config_path}"
    ((++dispatched_runs))
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run complete: ${SELECTED_RUNS} run(s) matched the filters (no jobs launched)." >&2
else
  echo "Completed ${dispatched_runs} job(s) locally (hash ${SWEEP_HASH})." >&2
fi

echo "Sweep complete." >&2
