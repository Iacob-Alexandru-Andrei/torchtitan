#!/usr/bin/env bash
# Sweep GaLoreGlobal ranks and learning rates via sbatch using warmed checkpoints.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
RUN_PREFIX=${RUN_PREFIX:-"icml2026-global-qhm-tuning"}
LOG_RANK=${LOG_RANK:-0}
# TorchFT/Des-LOC runtime knobs
NGPU=${NGPU:-${SBATCH_GPUS_PER_TASK:-4}}
MIN_REPLICAS=${MIN_REPLICAS:-${NGPU}}
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}
PORT_OFFSET=${PORT_OFFSET:-0}

PROJECTION_RANKS=${PROJECTION_RANKS:-"8 16 32 64 128"}
LR_VALUES=${LR_VALUES:-"0.016 0.008 0.008 0.008 0.016"}
WARMED_CHECKPOINTS=${WARMED_CHECKPOINTS:-"\
  icml2026-check-test-proj-savec-d99a5257-r8-lr0p016-rottrue-20251221-174207-idx0 \
  icml2026-global-checkpoint-4dd9e45e-r16-lr0p008-rottrue-20260104-215851-idx0 \
  icml2026-global-checkpoint-54f19506-r32-lr0p008-rottrue-20260104-222010-idx0 \
  icml2026-global-checkpoint-ebf60169-r64-lr0p008-rottrue-20260104-223953-idx0 \
  icml2026-global-checkpoint-a35e91e2-r128-lr0p016-rottrue-20260105-075751-idx0 \
"}
RESUME_STEP=${RESUME_STEP:-2048}
ROTATE_MOMENTS_OPTIONS=${ROTATE_MOMENTS_OPTIONS:-"true"}
SWITCH_SCALES=${SWITCH_SCALES:-"0.5"}

VS_VALUES=${VS_VALUES:-"0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99"}
QHM_OUTSIDE_OPTIONS=${QHM_OUTSIDE_OPTIONS:-"true false"}
ADAM_SENTINEL_RANK=${ADAM_SENTINEL_RANK:-256}
RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}
RUN_INDEX_RANGE=${RUN_INDEX_RANGE:-}
DRY_RUN=${DRY_RUN:-false}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-8}
SBATCH_GPUS_PER_TASK=${SBATCH_GPUS_PER_TASK:-4}
SBATCH_MAX_CHAINS=${SBATCH_MAX_CHAINS:-1}
SBATCH_MEM=${SBATCH_MEM:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_QOS=${SBATCH_QOS:-}
SBATCH_CONSTRAINT=${SBATCH_CONSTRAINT:-}
SBATCH_COMMENT=${SBATCH_COMMENT:-}
SBATCH_LOG_DIR=${SBATCH_LOG_DIR:-"${SCRIPT_DIR}/logs"}
RUN_LOG_DIR=${RUN_LOG_DIR:-"${SCRIPT_DIR}/run_logs"}
SBATCH_ADDITIONAL_ARGS=${SBATCH_ADDITIONAL_ARGS:-}
SBATCH_NODE=${SBATCH_NODE:-"mauao"}

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
  SBATCH_CPUS_PER_TASK, SBATCH_GPUS_PER_TASK, SBATCH_MAX_CHAINS, SBATCH_MEM, SBATCH_TIME
  SBATCH_PARTITION, SBATCH_ACCOUNT, SBATCH_QOS, SBATCH_CONSTRAINT
  SBATCH_COMMENT, SBATCH_LOG_DIR, SBATCH_ADDITIONAL_ARGS, SBATCH_NODE
  PROJECTION_RANKS, LR_VALUES, ROTATE_MOMENTS_OPTIONS
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

if ! [[ "${SBATCH_MAX_CHAINS}" =~ ^[0-9]+$ ]] || (( SBATCH_MAX_CHAINS < 1 )); then
  echo "SBATCH_MAX_CHAINS must be a positive integer (got ${SBATCH_MAX_CHAINS})." >&2
  exit 1
fi

USER_TRAINING_ARGS=("${TRAINING_ARGS[@]}")
serialize_args_array() {
  local formatted=""
  for arg in "$@"; do
    formatted+=" $(printf '%q' "$arg")"
  done
  echo "${formatted}"
}
TRAINING_ARGS_ESCAPED=$(serialize_args_array "${USER_TRAINING_ARGS[@]}")
read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_ADDITIONAL_ARGS}"
if [[ -z "${SBATCH_ADDITIONAL_ARGS}" ]]; then
  SBATCH_EXTRA_ARRAY=()
fi

read -r -a PROJECTION_RANK_ARRAY <<< "${PROJECTION_RANKS}"
read -r -a LR_ARRAY <<< "${LR_VALUES}"
read -r -a WARMED_CKPT_ARRAY <<< "${WARMED_CHECKPOINTS}"

# Use a fixed LR per rank: lengths must match one-to-one
if (( ${#LR_ARRAY[@]} != ${#PROJECTION_RANK_ARRAY[@]} )); then
  echo "When using fixed LR per rank, PROJECTION_RANKS and LR_VALUES must have the same number of entries." >&2
  exit 1
fi
if (( ${#WARMED_CKPT_ARRAY[@]} != 0 && ${#WARMED_CKPT_ARRAY[@]} != ${#PROJECTION_RANK_ARRAY[@]} )); then
  echo "WARMED_CHECKPOINTS must either be empty or match PROJECTION_RANKS/LR_VALUES length." >&2
  exit 1
fi
if (( ${#WARMED_CKPT_ARRAY[@]} == 0 )); then
  echo "WARMED_CHECKPOINTS is required: provide one warmed checkpoint per rank/lr pair." >&2
  exit 1
fi

if ! [[ "${RESUME_STEP}" =~ ^[0-9]+$ ]]; then
  echo "RESUME_STEP must be a positive integer (got ${RESUME_STEP})." >&2
  exit 1
fi

# Force rotate to true for all runs (ignore ROTATE_MOMENTS_OPTIONS input)
ROTATE_MOMENTS_OPTIONS="true"
ROTATE_MOMENTS_ARRAY=("true")

read -r -a SWITCH_SCALE_ARRAY <<< "${SWITCH_SCALES}"
read -r -a VS_ARRAY <<< "${VS_VALUES}"
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

# TOTAL runs is product of ranks x switch_scales x vs x qhm options x rotate options
TOTAL_RUNS=$(( ${#PROJECTION_RANK_ARRAY[@]} * ${#SWITCH_SCALE_ARRAY[@]} * ${#VS_ARRAY[@]} * ${#QHM_OUTSIDE_ARRAY[@]} * ${#ROTATE_MOMENTS_ARRAY[@]} ))

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
    for switch_scale in "${SWITCH_SCALE_ARRAY[@]}"; do
      for new_v in "${VS_ARRAY[@]}"; do
        for qhm in "${QHM_OUTSIDE_ARRAY[@]}"; do
          for rotate_flag in "${ROTATE_MOMENTS_ARRAY[@]}"; do
            ((++idx))
            if should_run_combination "${idx}"; then
              ((++selected))
            fi
          done
        done
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
declare -a RUN_PLAN_RANKS=()
declare -a RUN_PLAN_LRS=()
declare -a RUN_PLAN_ROTATES=()
declare -a RUN_PLAN_SWITCH_SCALES=()
declare -a RUN_PLAN_NEW_VS=()
declare -a RUN_PLAN_QHM_OUTSIDE=()
declare -a RUN_PLAN_RESUME=()

combination_index=0
# Build runs by index: LR maps 1:1 to PROJECTION_RANKS, and we sweep switch_scale, vs, qhm_outside
for i in "${!PROJECTION_RANK_ARRAY[@]}"; do
  rank=${PROJECTION_RANK_ARRAY[i]}
  lr=${LR_ARRAY[i]}
  resume_run=${WARMED_CKPT_ARRAY[i]}
  for switch_scale in "${SWITCH_SCALE_ARRAY[@]}"; do
    for new_v in "${VS_ARRAY[@]}"; do
      for qhm in "${QHM_OUTSIDE_ARRAY[@]}"; do
        for rotate_flag in "${ROTATE_MOMENTS_ARRAY[@]}"; do
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
          RUN_PLAN_RESUME+=("${resume_run}")
        done
      done
    done
  done
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
RDZV_BASE_PORT=${RDZV_BASE_PORT:-45000}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-46200}
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
  local resume_run=${8:-}
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
  RESUME_RUN="${resume_run}" \
  RESUME_STEP="${RESUME_STEP}" \
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
adam_rank_env = os.environ.get("ADAM_SENTINEL_RANK")
adam_rank = int(adam_rank_env) if adam_rank_env not in {None, ""} else None
# disable_projection = adam_rank is not None and rank == adam_rank

true_values = {"true", "1", "yes", "on"}
false_values = {"false", "0", "no", "off", ""}
if rotate_flag in true_values:
  rotate_moments = True
elif rotate_flag in false_values:
  rotate_moments = False
else:
  raise ValueError(f"Unsupported boolean for rotate_moments_on_refresh: {rotate_flag}")

data = tomllib.loads(base.read_text(encoding="utf-8"))

# Enable fault tolerance for TorchFT/DES-LOC
fault_tolerance = data.setdefault("fault_tolerance", {})
fault_tolerance["enable"] = True

optimizer = data.setdefault("optimizer", {})
# enforce GaLoreGlobal + DesLoc defaults for the tuning sweep
optimizer["name"] = "GaLoreGlobal"
optimizer["galore_rotate_moments_on_refresh"] = rotate_moments
optimizer["galore_use_error_feedback"] = True
optimizer["galore_update_proj_gap"] = 32
desloc_cfg = optimizer.get("desloc") if isinstance(optimizer.get("desloc"), dict) else {}
desloc_cfg["enabled"] = True
desloc_cfg["param_sync_every"] = 32
desloc_cfg["optimizer_sync_every"] = [32, 32, 32]
desloc_cfg["low_rank_server_update"] = True
desloc_cfg["low_rank_projector_error_feedback"] = True
desloc_cfg["low_rank_projector_source"] = "pseudo_grad"
optimizer["desloc"] = desloc_cfg
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

# if disable_projection:
#   normalized = [entry for entry in normalized if entry.get("param_str_match") != pattern]
# else:
updated = False
for entry in normalized:
  if entry.get("param_str_match") == pattern:
    entry["rank"] = rank
    updated = True
    break

  if not updated:
    normalized.append({"param_str_match": pattern, "rank": rank})

optimizer["galore_param_regexes"] = normalized
# Enable local error feedback for GaLore optimizer in generated configs
optimizer["galore_use_error_feedback"] = True

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
if qhm_bool is not None:
  optimizer["galore_qhm_outside_projection"] = qhm_bool
# If a per-run LR value was provided, write it into the optimizer section
lr_env = os.environ.get("LR_VALUE", "")
if lr_env is not None and lr_env.strip() != "":
  try:
    optimizer["lr"] = float(lr_env)
  except Exception:
    optimizer["lr"] = lr_env

# Resume from warmed checkpoint and enforce training steps
resume_run = os.environ.get("RESUME_RUN", "").strip()
resume_step = os.environ.get("RESUME_STEP", "2048").strip()
if resume_run:
  s3_cfg = data.setdefault("s3_checkpoint", {})
  s3_cfg["resume_from_run_step"] = f"{resume_run}/step-{resume_step}"
  data.setdefault("training", {})["steps"] = int(data.get("training", {}).get("steps", 6144))
else:
  raise ValueError("RESUME_RUN must be provided for warmed checkpoint tuning")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(tomli_w.dumps(data), encoding="utf-8")
PY

  echo "${output_path}"
}



if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
  exit 1
fi

declare -a SBATCH_JOB_IDS=()
declare -a SBATCH_CHAIN_LAST_IDS=()
RUN_COUNTER=0

if [[ "${DRY_RUN}" != "true" ]]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found in PATH; this script requires sbatch submission." >&2
    exit 1
  fi
  if [[ -n "${SBATCH_LOG_DIR}" ]]; then
    mkdir -p "${SBATCH_LOG_DIR}"
  fi
  if [[ -n "${RUN_LOG_DIR}" ]]; then
    mkdir -p "${RUN_LOG_DIR}"
  fi
fi

for ((chain_idx = 0; chain_idx < SBATCH_MAX_CHAINS; ++chain_idx)); do
  SBATCH_CHAIN_LAST_IDS[chain_idx]=''
done

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

submit_sbatch_job() {
  local run_uuid=$1
  local run_progress=$2
  local lighthouse_port=$3
  local rdzv_base_port=$4
  local combo_index_label=$5
  local proj_rank=$6
  local lr_value=$7
  local rotate_flag=$8
  local dependency_job=$9
  local chain_index=${10}
  local run_config_path=${11}
  local run_log_dir=${12}

  # Precompute lighthouse URL to avoid set -u issues during heredoc expansion
  local lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"
  local lighthouse_log_file="${run_log_dir}/lighthouse.log"
  # Provide uppercase binding to satisfy set -u during heredoc expansion
  LIGHTHOUSE_LOG_FILE="${lighthouse_log_file}"

  local job_name="${RUN_PREFIX}-${SWEEP_HASH}-idx${combo_index_label}"
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

  # Export all necessary variables for the sbatch job
  local sbatch_export_arg="ALL"
  for kv in \
    "RUN_UUID=${run_uuid}" \
    "WANDB_PROJECT=${WANDB_PROJECT:-galore-tune-lr}" \
    "WANDB_TEAM=${WANDB_TEAM:-camlsys}" \
    "WANDB_RUN_NAME=${run_uuid}" \
    "TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=1" \
    "S3_ENDPOINT_URL=http://taranaki.cl.cam.ac.uk:9000" \
    "LOG_DIR=${run_log_dir}" \
    "NGPU=${NGPU}" \
    "MIN_REPLICAS=${MIN_REPLICAS}" \
    "QUORUM_TICK_MS=${QUORUM_TICK_MS}" \
    "LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST}" \
    "LIGHTHOUSE_PORT=${lighthouse_port}" \
    "RDZV_PORT_BASE=${rdzv_base_port}" \
    "TRAIN_MODULE=${TRAIN_MODULE}" \
    "LIGHTHOUSE_LOG_FILE=${lighthouse_log_file}" \
    "LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-http}" \
    "TORCHFT_LIGHTHOUSE=${lighthouse_url}" \
    "CONFIG_PATH=${run_config_path}" \
    "TRAINING_ARGS_ESCAPED=${TRAINING_ARGS_ESCAPED}"; do
    sbatch_export_arg+="${kv:+,${kv}}"
  done
  sbatch_opts+=("--export=${sbatch_export_arg}")

  local sbatch_output
  sbatch_output=$(sbatch "${sbatch_opts[@]}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "==================================================================="
echo "STARTING SBATCH JOB (ID: $SLURM_JOB_ID)"
echo "Run UUID: ${RUN_UUID}"
echo "Node: $(hostname) at $(date)"
echo "Lighthouse: ${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT} | RDZV base: ${RDZV_PORT_BASE}"
echo "==================================================================="

find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

mkdir -p "${LOG_DIR}"

uv run --no-sync torchft_lighthouse \
  --min_replicas "${MIN_REPLICAS}" \
  --quorum_tick_ms "${QUORUM_TICK_MS}" \
  --bind "${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}" \
  > "${LIGHTHOUSE_LOG_FILE}" 2>&1 &
LIGHTHOUSE_PID=$!
sleep 2
if ! kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
  echo "Lighthouse failed to start; see ${LIGHTHOUSE_LOG_FILE}" >&2
  exit 1
fi

AVAILABLE_GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((i=0; i<NGPU; i++)); do
    AVAILABLE_GPUS+=("${i}")
  done
fi

if (( ${#AVAILABLE_GPUS[@]} < NGPU )); then
  echo "Requested ${NGPU} replicas but only ${#AVAILABLE_GPUS[@]} GPU(s) available." >&2
  exit 1
fi

REPLICA_GPUS=("${AVAILABLE_GPUS[@]:0:NGPU}")
declare -a REPLICA_PIDS=()

for ((replica_id=0; replica_id<NGPU; replica_id++)); do
  gpu_id="${REPLICA_GPUS[$replica_id]}"
  log_file="${LOG_DIR}/replica_${replica_id}.log"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    rdzv_port=$((RDZV_PORT_BASE + replica_id))
    cmd="uv run --no-sync torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=\"localhost:${rdzv_port}\" --role=rank --tee=3 -m \"${TRAIN_MODULE}\" --job.config_file \"${CONFIG_PATH}\" --fault_tolerance.replica_id \"${replica_id}\" --fault_tolerance.group_size \"${NGPU}\" --fault_tolerance.min_replica_size \"${MIN_REPLICAS}\" ${TRAINING_ARGS_ESCAPED}"
    eval "${cmd}"
  ) > "${log_file}" 2>&1 &
  REPLICA_PIDS[$replica_id]=$!
  sleep 1
done

set +e
replica_status=0
for pid in "${REPLICA_PIDS[@]}"; do
  if ! wait "${pid}"; then
    replica_status=$?
  fi
done
set -e

if kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
  kill "${LIGHTHOUSE_PID}"
fi

exit "${replica_status}"
EOF
)

  # Parse the sbatch output to extract job ID
  local job_id
  if [[ "${sbatch_output}" =~ ^[0-9]+$ ]]; then
    job_id="${sbatch_output}"
  else
    echo "Failed to submit sbatch job ${job_name}. Output: ${sbatch_output}" >&2
    return 1
  fi

  # Ensure sbatch command succeeds and sets job_id
  if [[ -z "${job_id:-}" ]]; then
    echo "Failed to retrieve job ID from sbatch command." >&2
    return 1
  fi

  echo "${job_id}"
}

EXECUTION_DESC="sbatch submission"

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
    resume_run=${RUN_PLAN_RESUME[idx]}

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
      run_config_path=$(generate_run_config "${run_uuid}" "${proj_rank}" "${rotate_flag}" "${switch_scale_val}" "${new_v_val}" "${qhm_outside_val}" "${lr_value}" "${resume_run}")
    else
      run_config_path="${CONFIG_FILE}"
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
      echo "[DRY-RUN][SBATCH] ${run_uuid} (rank=${proj_rank}, lr=${lr_value}, rotate=${rotate_flag}) [${run_progress}]" >&2
      continue
    fi

    launch_index=$((RUN_COUNTER + 1))
    rdzv_base_port=$((RDZV_BASE_PORT + PORT_OFFSET + launch_index * PORT_STRIDE))
    lighthouse_port=$((LIGHTHOUSE_BASE_PORT + PORT_OFFSET + launch_index * PORT_STRIDE))
    RUN_COUNTER=${launch_index}

    run_log_dir="${RUN_LOG_DIR}/${run_uuid}"
    mkdir -p "${run_log_dir}"

    chain_index=$((dispatched_runs % SBATCH_MAX_CHAINS))
    dependency_job=${SBATCH_CHAIN_LAST_IDS[chain_index]:-}
    if [[ -n "${dependency_job}" ]]; then
      echo "[SBATCH] Chain ${chain_index}: submitting ${run_uuid} after job ${dependency_job}." >&2
    else
      echo "[SBATCH] Chain ${chain_index}: submitting ${run_uuid} with no dependency." >&2
    fi
    job_id=$(submit_sbatch_job "${run_uuid}" "${run_progress}" "${lighthouse_port}" "${rdzv_base_port}" "${combination_index_zero}" "${proj_rank}" "${lr_value}" "${rotate_flag}" "${dependency_job}" "${chain_index}" "${run_config_path}" "${run_log_dir}")
    if [[ -z "${job_id}" ]]; then
      echo "Aborting sweep after failed sbatch submission." >&2
      exit 1
    fi
    SBATCH_CHAIN_LAST_IDS[chain_index]="${job_id}"
    SBATCH_JOB_IDS+=("${job_id}")
    echo "[SBATCH] Submitted ${run_uuid} (${run_progress}) as job ${job_id} | rank=${proj_rank}, lr=${lr_value}, rotate=${rotate_flag}" >&2
    sleep 0.5

    ((++dispatched_runs))
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run complete: ${SELECTED_RUNS} run(s) matched the filters (no jobs launched)." >&2
else
  echo "Submitted ${dispatched_runs} job(s) via sbatch (hash ${SWEEP_HASH})." >&2
fi

echo "Sweep complete." >&2
