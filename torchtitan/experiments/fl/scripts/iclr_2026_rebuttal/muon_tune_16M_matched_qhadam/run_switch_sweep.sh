#!/usr/bin/env bash
# Sweep AggMoMuon + AggMoAdamW quasi-hyperbolic v values and switch-scale after warmup.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-qhmuon"}
LOG_RANK=${LOG_RANK:-0}

WARMUP_STEPS=${WARMUP_STEPS:-2048}
SWITCH_APPLY_STEP=${SWITCH_APPLY_STEP:-$((WARMUP_STEPS + 1))}
LR_SWITCH_STEP=${LR_SWITCH_STEP:-${SWITCH_APPLY_STEP}}
MUON_LR=${MUON_LR:-0.002}
ADAMW_LR=${ADAMW_LR:-0.001}
BASE_V=${BASE_V:-1.0}
VS_VALUES=${VS_VALUES:-"0.9 0.91 0.93 0.95 0.97 0.98 0.99"}
SWITCH_SCALES=${SWITCH_SCALES:-"1 1.4142135624 2 4"}

RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}
RUN_INDEX_RANGE=${RUN_INDEX_RANGE:-}
USE_SBATCH=${USE_SBATCH:-false}
DRY_RUN=${DRY_RUN:-false}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-8}
SBATCH_GPUS_PER_TASK=${SBATCH_GPUS_PER_TASK:-1}
SBATCH_MAX_CHAINS=${SBATCH_MAX_CHAINS:-2}
SBATCH_MEM=${SBATCH_MEM:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_QOS=${SBATCH_QOS:-}
SBATCH_CONSTRAINT=${SBATCH_CONSTRAINT:-}
SBATCH_COMMENT=${SBATCH_COMMENT:-}
SBATCH_LOG_DIR=${SBATCH_LOG_DIR:-"${SCRIPT_DIR}/logs"}
SBATCH_ADDITIONAL_ARGS=${SBATCH_ADDITIONAL_ARGS:-}
SBATCH_NODE=${SBATCH_NODE:-}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-5}
RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}
PORT_STRIDE=${PORT_STRIDE:-4}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-""}

usage() {
  cat <<'EOF'
Usage: run_switch_sweep.sh [OPTIONS] [-- extra training args]

Options:
  --range START-END        Run only the 0-indexed inclusive range of sweep jobs.
  --run-index INDEX        Run only the 1-indexed job at INDEX.
  --run-index-offset N     Skip the first N jobs before applying other filters.
  --sbatch                 Submit runs via sbatch instead of launching locally.
  --no-sbatch              Force local execution (default).
  --dry-run                Print matching runs without launching anything.
  -h, --help               Show this message.
  --                       Treat the remaining arguments as training args.

Environment knobs:
  VS_VALUES (space-separated v targets applied at switch step)
  SWITCH_SCALES (space-separated lr multipliers at switch step)
  MUON_LR, ADAMW_LR, BASE_V, WARMUP_STEPS, SWITCH_APPLY_STEP, LR_SWITCH_STEP
  RUN_INDEX, RUN_INDEX_OFFSET, RUN_INDEX_RANGE, USE_SBATCH, DRY_RUN
  SBATCH_* variables for Slurm submission, GPU_IDS for local launches
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

if ! [[ "${SBATCH_MAX_CHAINS}" =~ ^[0-9]+$ ]] || (( SBATCH_MAX_CHAINS < 1 )); then
  echo "SBATCH_MAX_CHAINS must be a positive integer (got ${SBATCH_MAX_CHAINS})." >&2
  exit 1
fi

if ! [[ "${RUN_INDEX_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX_OFFSET must be a non-negative integer (got ${RUN_INDEX_OFFSET})." >&2
  exit 1
fi
RUN_INDEX_OFFSET=$((10#${RUN_INDEX_OFFSET}))

read -r -a VS_ARRAY <<< "${VS_VALUES}"
read -r -a SWITCH_SCALE_ARRAY <<< "${SWITCH_SCALES}"

if (( ${#VS_ARRAY[@]} == 0 )); then
  echo "VS_VALUES must contain at least one entry." >&2
  exit 1
fi
if (( ${#SWITCH_SCALE_ARRAY[@]} == 0 )); then
  echo "SWITCH_SCALES must contain at least one entry." >&2
  exit 1
fi

validate_index_range() {
  local range="$1"
  if [[ -z "${range}" ]]; then
    echo "false"
    return
  fi
  if [[ "${range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    local start=$((10#${BASH_REMATCH[1]}))
    local end=$((10#${BASH_REMATCH[2]}))
    if (( end < start )); then
      echo "invalid"
    else
      echo "${start}-${end}"
    fi
  else
    echo "invalid"
  fi
}

RANGE_STATUS=$(validate_index_range "${RUN_INDEX_RANGE}")
if [[ "${RANGE_STATUS}" == "invalid" ]]; then
  echo "RUN_INDEX_RANGE must match START-END (e.g., 0-15)." >&2
  exit 1
fi

if [[ -n "${RUN_INDEX}" ]]; then
  if ! [[ "${RUN_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "RUN_INDEX must be a positive integer." >&2
    exit 1
  fi
  RUN_INDEX=$((10#${RUN_INDEX}))
fi

serialize_training_args() {
  local formatted=""
  for arg in "${TRAINING_ARGS[@]}"; do
    formatted+=" $(printf '%q' "$arg")"
  done
  echo "${formatted}"
}
TRAINING_ARGS_ESCAPED=$(serialize_training_args)
read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_ADDITIONAL_ARGS}"
if [[ -z "${SBATCH_ADDITIONAL_ARGS}" ]]; then
  SBATCH_EXTRA_ARRAY=()
fi

TOTAL_RUNS=$(( ${#SWITCH_SCALE_ARRAY[@]} * ${#VS_ARRAY[@]} ))

RANGE_ENABLED=false
RANGE_START=
RANGE_END=
if [[ "${RANGE_STATUS}" != "false" ]]; then
  RANGE_ENABLED=true
  RANGE_START=$((10#${RANGE_STATUS%-*}))
  RANGE_END=$((10#${RANGE_STATUS#*-}))
fi

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
if [[ -n "${RUN_INDEX}" ]]; then
  if (( RUN_INDEX < 1 || RUN_INDEX > TOTAL_RUNS )); then
    echo "RUN_INDEX must be between 1 and ${TOTAL_RUNS} (got ${RUN_INDEX})." >&2
    exit 1
  fi
fi

should_run_combination() {
  local combo_index_1=$1
  local combo_index_0=$((combo_index_1 - 1))

  if (( combo_index_1 <= RUN_INDEX_OFFSET )); then
    return 1
  fi
  if [[ -n "${RUN_INDEX}" ]]; then
    [[ "${combo_index_1}" -eq "${RUN_INDEX}" ]] && return 0 || return 1
  fi
  if [[ "${RANGE_ENABLED}" == "true" ]]; then
    (( combo_index_0 < RANGE_START )) && return 1
    (( combo_index_0 > RANGE_END )) && return 1
  fi
  return 0
}

declare -a RUN_PLAN_INDICES=()
declare -a RUN_PLAN_SWITCHES=()
declare -a RUN_PLAN_VS=()

combo_idx=0
for switch_scale in "${SWITCH_SCALE_ARRAY[@]}"; do
  for v_value in "${VS_ARRAY[@]}"; do
    ((++combo_idx))
    if should_run_combination "${combo_idx}"; then
      RUN_PLAN_INDICES+=("${combo_idx}")
      RUN_PLAN_SWITCHES+=("${switch_scale}")
      RUN_PLAN_VS+=("${v_value}")
    fi
  done
done

SELECTED_RUNS=${#RUN_PLAN_INDICES[@]}
if (( SELECTED_RUNS == 0 )); then
  echo "No runs match the requested filters; nothing to do." >&2
  exit 0
fi

SWEEP_CONFIG_STRING="vs=${VS_VALUES}|switch_scales=${SWITCH_SCALES}|base=${CONFIG_FILE}"
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

mkdir -p "${SCRIPT_DIR}/tmp_configs"

sanitize_value() {
  local value=$1
  value=${value//./p}
  value=${value//-/m}
  value=${value//+/p}
  echo "${value}"
}

make_temp_config() {
  local new_v=$1
  local switch_scale=$2
  local tmp_file
  tmp_file=$(mktemp "${SCRIPT_DIR}/tmp_configs/muon_switchXXXXXX.toml")
  python3 - <<PY
import tomllib, tomli_w
from pathlib import Path

base = Path("${CONFIG_FILE}")
tmp = Path("${tmp_file}")
new_v = float("${new_v}")
switch_scale = float("${switch_scale}")
muon_nesterov = True

warmup_steps = int("${WARMUP_STEPS}")
switch_step = int("${SWITCH_APPLY_STEP}")
lr_switch_step = int("${LR_SWITCH_STEP}")
muon_lr = float("${MUON_LR}")
adam_lr = float("${ADAMW_LR}")
base_v = float("${BASE_V}")

with base.open("rb") as f:
    data = tomllib.load(f)

opt = data.get("optimizer", {})
opt["name"] = "AggMoAdamW"
opt["builder"] = "mosaic"
opt["vs"] = (base_v,)
opt["muon_nesterov"] = muon_nesterov
opt["lr"] = adam_lr
opt["beta1"] = float(opt.get("beta1", 0.9))
opt["beta2"] = float(opt.get("beta2", 0.999))
opt["weight_decay"] = float(opt.get("weight_decay", 0.1))
opt["decouple"] = bool(opt.get("decouple", False))
opt["implementation"] = opt.get("implementation", "for-loop")
opt["eps"] = float(opt.get("eps", 1e-8))

opt["composite"] = [
    {
        "name": "AggMoMuon",
        "labels": ["decay_lr"],
        "config_overrides": {
            "lr": muon_lr,
            "weight_decay": opt["weight_decay"],
            "eps": float(opt.get("eps", 1e-7)),
            "muon_nesterov": muon_nesterov,
            "beta1": opt["beta1"],
            "adjust_lr_fn": opt.get("adjust_lr_fn", "match_rms_adamw"),
            "vs": (base_v,),
        },
    },
    {
        "name": "AggMoAdamW",
        "default": True,
        "config_overrides": {
            "lr": adam_lr,
            "weight_decay": opt["weight_decay"],
            "betas": (opt["beta1"], opt["beta2"]),
            "decouple": opt["decouple"],
            "vs": (base_v,),
        },
    },
]
data["optimizer"] = opt

lr_cfg = data.get("lr_scheduler", {})
lr_cfg["warmup_steps"] = warmup_steps
lr_cfg["switch_step"] = lr_switch_step
lr_cfg["switch_scale"] = switch_scale
data["lr_scheduler"] = lr_cfg

fl_metrics = data.setdefault("fl_metrics", {})
hyper = fl_metrics.get("hyperparameter_switch", {}) if isinstance(fl_metrics.get("hyperparameter_switch"), dict) else {}
hyper["enabled"] = True
hyper["steps"] = (switch_step,)
hyper["new_vs"] = (new_v,)
hyper["new_nesterov"] = (False,)
hyper["reset_momenta"] = ("exp_avg",)
fl_metrics["hyperparameter_switch"] = hyper
data["fl_metrics"] = fl_metrics

with tmp.open("wb") as f:
    tomli_w.dump(data, f)

print(tmp)
PY
}

GPU_IDS=${GPU_IDS:-""}
GPU_IDS_AUTO_SOURCE=""
GPU_IDS_DEDUPED_COUNT=0
declare -a GPU_ARRAY=()
declare -a GPU_PIDS=()
declare -a GPU_LABELS=()
declare -a ALL_PIDS=()
RUN_COUNTER=0

find_free_port() {
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}

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

configure_gpus() {
  if [[ -z "${GPU_IDS}" || "${GPU_IDS,,}" == "auto" ]]; then
    if ids=$(detect_available_gpu_ids); then
      read -r -a GPU_ARRAY <<< "${ids}"
    else
      echo "Failed to auto-detect GPUs. Set GPU_IDS explicitly." >&2
      exit 1
    fi
  else
    set_gpu_array_from_string "${GPU_IDS}" || exit 1
  fi
  # Initialize tracking arrays with the same length as GPU_ARRAY; command substitution
  # drops empty elements, so populate then blank them out to keep indices aligned.
  GPU_PIDS=("${GPU_ARRAY[@]}")
  GPU_LABELS=("${GPU_ARRAY[@]}")
  for i in "${!GPU_ARRAY[@]}"; do
    GPU_PIDS[i]=""
    GPU_LABELS[i]=""
  done
}

wait_for_gpu() {
  while true; do
    for idx in "${!GPU_ARRAY[@]}"; do
      pid=${GPU_PIDS[idx]}
      if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        GPU_PIDS[idx]=""
        GPU_LABELS[idx]=""
        echo "${idx}"
        return
      fi
    done
    sleep "${POLL_INTERVAL_SEC}"
  done
}

formatted_switch_steps_arg() {
  printf '%s' "${SWITCH_APPLY_STEP}"
}

submit_sbatch_job() {
  local run_uuid=$1
  local run_progress=$2
  local rdzv_endpoint=$3
  local lighthouse_url=$4
  local combo_index_label=$5
  local new_v=$6
  local switch_scale=$7
  local config_path=$8
  local dependency_job=$9
  local chain_index=${10}

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

  local switch_steps_arg
  switch_steps_arg=$(formatted_switch_steps_arg)

  local job_id
  if ! job_id=$(sbatch "${sbatch_opts[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "==================================================================="
echo "STARTING SBATCH JOB: ${job_name} (ID: \$SLURM_JOB_ID)"
echo "Run UUID: ${run_uuid} | Progress: ${run_progress} | v=${new_v} | switch_scale=${switch_scale}"
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
  --job.config_file "${config_path}" \
  ${TRAINING_ARGS_ESCAPED}
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
  configure_gpus
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

printf "\nSelected run configurations (%s total):\n" "${SELECTED_RUNS}" >&2
printf "%-10s %-10s %-10s %-14s\n" "Idx(1-based)" "Idx(0-based)" "v" "switch_scale" >&2
printf "%-10s %-10s %-10s %-14s\n" "----------" "----------" "----------" "--------------" >&2
for idx in "${!RUN_PLAN_INDICES[@]}"; do
  combo_index_1=${RUN_PLAN_INDICES[idx]}
  combo_index_0=$((combo_index_1 - 1))
  printf "%-10s %-10s %-10s %-14s\n" \
    "${combo_index_1}" \
    "${combo_index_0}" \
    "${RUN_PLAN_VS[idx]}" \
    "${RUN_PLAN_SWITCHES[idx]}" >&2
done
echo "" >&2

echo "Starting muon+adam AggMo sweep hash=${SWEEP_HASH} using ${EXECUTION_DESC}: ${SELECTED_RUNS}/${TOTAL_RUNS} run(s) selected (${FILTER_DESC})." >&2

timestamp_global=$(date +"%Y%m%d-%H%M%S")
dispatched_runs=0

declare -a SBATCH_JOB_IDS=()
declare -a SBATCH_CHAIN_LAST_IDS=()

for idx in "${!RUN_PLAN_INDICES[@]}"; do
  combination_index=${RUN_PLAN_INDICES[idx]}
  combination_index_zero=$((combination_index - 1))
  v_value=${RUN_PLAN_VS[idx]}
  switch_scale=${RUN_PLAN_SWITCHES[idx]}
  v_tag=$(sanitize_value "${v_value}")
  scale_tag=$(sanitize_value "${switch_scale}")
  run_uuid="${RUN_PREFIX}-${SWEEP_HASH}-v${v_tag}-sw${scale_tag}-${timestamp_global}-idx${combination_index_zero}"
  run_progress="run ${combination_index}/${TOTAL_RUNS}"

  tmp_config=$(make_temp_config "${v_value}" "${switch_scale}")
  if [[ ! -f "${tmp_config}" ]]; then
    echo "Failed to build temporary config for run ${run_uuid}." >&2
    exit 1
  fi

  if [[ "${DRY_RUN}" == "true" ]]; then
    if [[ "${USE_SBATCH}" == "true" ]]; then
      echo "[DRY-RUN][SBATCH] ${run_uuid} (v=${v_value}, switch_scale=${switch_scale}) [${run_progress}]" >&2
    else
      echo "[DRY-RUN][LOCAL] ${run_uuid} (v=${v_value}, switch_scale=${switch_scale}) [${run_progress}]" >&2
    fi
    continue
  fi

  launch_index=$((RUN_COUNTER + 1))
  RUN_COUNTER=${launch_index}

  if [[ -n "${RDZV_ENDPOINT}" ]]; then
    rdzv_endpoint="${RDZV_ENDPOINT}"
  else
    rdzv_port=$(find_free_port)
    rdzv_endpoint="${RDZV_HOST}:${rdzv_port}"
  fi

  lighthouse_port=$(find_free_port)
  lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"

  if [[ "${USE_SBATCH}" == "true" ]]; then
    chain_index=$((dispatched_runs % SBATCH_MAX_CHAINS))
    dependency_job=${SBATCH_CHAIN_LAST_IDS[chain_index]:-}
    job_id=$(submit_sbatch_job "${run_uuid}" "${run_progress}" "${rdzv_endpoint}" "${lighthouse_url}" "${combination_index_zero}" "${v_value}" "${switch_scale}" "${tmp_config}" "${dependency_job}" "${chain_index}")
    if [[ -z "${job_id}" ]]; then
      echo "Aborting sweep after failed sbatch submission." >&2
      exit 1
    fi
    SBATCH_CHAIN_LAST_IDS[chain_index]="${job_id}"
    SBATCH_JOB_IDS+=("${job_id}")
    echo "[SBATCH] Submitted ${run_uuid} (${run_progress}) as job ${job_id} | v=${v_value}, switch_scale=${switch_scale}" >&2
    sleep 30
  else
    gpu_slot=$(wait_for_gpu)
    gpu_id=${GPU_ARRAY[gpu_slot]}
    echo "[GPU ${gpu_id}] Launching run ${run_uuid} (v=${v_value}, switch_scale=${switch_scale}) [${run_progress}]" >&2
    echo "           torchrun rendezvous: ${rdzv_endpoint}, lighthouse: ${lighthouse_url}" >&2

    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
      export RUN_UUID="${run_uuid}"
      export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_tune_muon"}
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
  --job.config_file "${tmp_config}" \
  "${TRAINING_ARGS[@]}"
    ) &

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
