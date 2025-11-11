#!/usr/bin/env bash
# TorchFT MT-Dao sweep with QH-AdamW (v=0.98) across worker counts/batch sizes.
# Modern features: hashing, run-index filtering, concurrent GPU scheduling, DES-LOC.
set -euo pipefail

export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
MODEL_SIZES=${MODEL_SIZES:-"16M"}
WORKER_COUNTS=${WORKER_COUNTS:-"1 2 4 8"}
GLOBAL_BATCH_SIZES=${GLOBAL_BATCH_SIZES:-"64 128 256 512"}
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-mtdaoWB"}
LOG_RANK=${LOG_RANK:-0}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-16}
BASE_LR=${BASE_LR:-0.04}
VS_VALUE=${VS_VALUE:-0.98}
REFERENCE_GLOBAL_BATCH_SIZE=${REFERENCE_GLOBAL_BATCH_SIZE:-64}
REFERENCE_TRAINING_STEPS=${REFERENCE_TRAINING_STEPS:-20480}
REFERENCE_VALIDATION_FREQ=${REFERENCE_VALIDATION_FREQ:-2048}

WARMUP_RUN_STEP_16M="16M-baseline-20251029-095728/step-2048"

RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}
RUN_INDEX_RANGE=${RUN_INDEX_RANGE:-}
DRY_RUN=${DRY_RUN:-false}

RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
RDZV_BASE_PORT=${RDZV_BASE_PORT:-44000}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-44100}
PORT_STRIDE=${PORT_STRIDE:-8}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-5}
TORCHFT_MIN_REPLICAS=${TORCHFT_MIN_REPLICAS:-}
TORCHFT_QUORUM_TICK_MS=${TORCHFT_QUORUM_TICK_MS:-100}
TORCHFT_LIGHTHOUSE_BOOT_DELAY=${TORCHFT_LIGHTHOUSE_BOOT_DELAY:-2}
TORCHFT_SYNC_STEPS=${TORCHFT_SYNC_STEPS:-32}\

declare -a GPU_ARRAY=()
declare -i GPU_IDS_DEDUPED_COUNT=0
GPU_IDS_AUTO_SOURCE=""
GPU_ALLOCATION_RESULT=""

LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: run_mtdao_sweep.sh [OPTIONS] [-- extra training args]

Options:
  --range START-END        Run only the 0-indexed inclusive range of sweep jobs.
  --run-index INDEX        Run only the 1-indexed job at INDEX.
  --run-index-offset N     Skip the first N jobs before applying other filters.
  --dry-run                Print matching runs without launching anything.
  -h, --help               Show this message.
  --                       Treat the remaining arguments as training args.

Environment customizations:
  CONFIG_FILE, TRAIN_MODULE, RUN_PREFIX, LOG_DIR
  MODEL_SIZES, WORKER_COUNTS, GLOBAL_BATCH_SIZES
  LOCAL_BATCH_SIZE, BASE_LR, VS_VALUE
  REFERENCE_GLOBAL_BATCH_SIZE, REFERENCE_TRAINING_STEPS, REFERENCE_VALIDATION_FREQ
  RUN_INDEX, RUN_INDEX_OFFSET, RUN_INDEX_RANGE, DRY_RUN
  RDZV_HOST, RDZV_BASE_PORT, LIGHTHOUSE_HOST, LIGHTHOUSE_BASE_PORT, PORT_STRIDE
  GPU_IDS (space/comma separated or 'auto'), POLL_INTERVAL_SEC
  TORCHFT_MIN_REPLICAS, TORCHFT_QUORUM_TICK_MS, TORCHFT_LIGHTHOUSE_BOOT_DELAY, TORCHFT_SYNC_STEPS
EOF
}

TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --range)
      [[ $# -ge 2 ]] || { echo "Missing value for --range" >&2; exit 1; }
      RUN_INDEX_RANGE=$2
      shift 2
      ;;
    --run-index)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-index" >&2; exit 1; }
      RUN_INDEX=$2
      shift 2
      ;;
    --run-index-offset)
      [[ $# -ge 2 ]] || { echo "Missing value for --run-index-offset" >&2; exit 1; }
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

DRY_RUN=$(normalize_bool "${DRY_RUN}")

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
  exit 1
fi

if ! [[ "${RUN_INDEX_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "RUN_INDEX_OFFSET must be a non-negative integer (got ${RUN_INDEX_OFFSET})." >&2
  exit 1
fi
RUN_INDEX_OFFSET=$((10#${RUN_INDEX_OFFSET}))

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
    echo "RUN_INDEX_RANGE must match START-END (e.g., 0-35)." >&2
    exit 1
  fi
fi

RUN_INDEX_SET=false
if [[ -n "${RUN_INDEX}" ]]; then
  if ! [[ "${RUN_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "RUN_INDEX must be a positive integer (got ${RUN_INDEX})." >&2
    exit 1
  fi
  RUN_INDEX=$((10#${RUN_INDEX}))
  RUN_INDEX_SET=true
fi

read -r -a MODEL_SIZE_ARRAY <<< "${MODEL_SIZES}"
read -r -a WORKER_COUNT_ARRAY <<< "${WORKER_COUNTS}"
read -r -a GLOBAL_BATCH_ARRAY <<< "${GLOBAL_BATCH_SIZES}"

if (( ${#MODEL_SIZE_ARRAY[@]} == 0 )); then
  echo "MODEL_SIZES must contain at least one entry." >&2
  exit 1
fi
if (( ${#WORKER_COUNT_ARRAY[@]} == 0 )); then
  echo "WORKER_COUNTS must contain at least one entry." >&2
  exit 1
fi
if (( ${#GLOBAL_BATCH_ARRAY[@]} == 0 )); then
  echo "GLOBAL_BATCH_SIZES must contain at least one entry." >&2
  exit 1
fi

if (( REFERENCE_GLOBAL_BATCH_SIZE <= 0 )); then
  echo "REFERENCE_GLOBAL_BATCH_SIZE must be positive (got ${REFERENCE_GLOBAL_BATCH_SIZE})." >&2
  exit 1
fi
if (( REFERENCE_TRAINING_STEPS <= 0 )); then
  echo "REFERENCE_TRAINING_STEPS must be positive (got ${REFERENCE_TRAINING_STEPS})." >&2
  exit 1
fi
if (( REFERENCE_VALIDATION_FREQ <= 0 )); then
  echo "REFERENCE_VALIDATION_FREQ must be positive (got ${REFERENCE_VALIDATION_FREQ})." >&2
  exit 1
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
if (( TOTAL_GPUS == 0 )); then
  echo "GPU_IDS resolved to zero GPUs; provide GPU ids or set GPU_IDS=auto." >&2
  exit 1
fi

if [[ -n "${GPU_IDS_AUTO_SOURCE}" ]]; then
  echo "Auto-detected ${TOTAL_GPUS} GPU id(s) via ${GPU_IDS_AUTO_SOURCE}: ${GPU_ARRAY[*]}." >&2
fi
if (( GPU_IDS_DEDUPED_COUNT > 0 )); then
  echo "Removed ${GPU_IDS_DEDUPED_COUNT} duplicate GPU id(s); using GPU set: ${GPU_ARRAY[*]}." >&2
fi

sanitize_key() {
  echo "$1" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9' '_'
}

# warmup_step_for() {
#   local size=$1
#   local key
#   key=$(sanitize_key "${size}")
#   local var_name="WARMUP_RUN_STEP_${key}"
#   local value=""
#   if [[ -n "${!var_name-}" ]]; then
#     value=${!var_name}
#   elif [[ -n "${WARMUP_RUN_STEP:-}" ]]; then
#     value=${WARMUP_RUN_STEP}
#   fi
#   if [[ -z "${value}" ]]; then
#     echo "Warmup checkpoint not provided. Set WARMUP_RUN_STEP or ${var_name}." >&2
#     exit 1
#   fi
#   printf '%s' "${value}"
# }

ensure_divisible() {
  local numer=$1
  local denom=$2
  if (( numer % denom != 0 )); then
    echo "Global batch size ${numer} must be divisible by worker count ${denom}." >&2
    exit 1
  fi
}

compute_scaled_steps() {
  local global_bs=$1
  if (( global_bs <= 0 )); then
    echo "Global batch size must be positive when computing scaled steps (got ${global_bs})." >&2
    exit 1
  fi
  local numerator=$((REFERENCE_TRAINING_STEPS * REFERENCE_GLOBAL_BATCH_SIZE))
  local steps=$(((numerator + global_bs - 1) / global_bs))
  if (( steps < 1 )); then
    steps=1
  fi
  echo "${steps}"
}

compute_scaled_validation_freq() {
  local global_bs=$1
  if (( global_bs <= 0 )); then
    echo "Global batch size must be positive when computing validation frequency (got ${global_bs})." >&2
    exit 1
  fi
  local numerator=$((REFERENCE_VALIDATION_FREQ * REFERENCE_GLOBAL_BATCH_SIZE))
  local freq=$(((numerator + global_bs - 1) / global_bs))
  if (( freq < 1 )); then
    freq=1
  fi
  echo "${freq}"
}

should_run_combination() {
  local combo_index_1=$1
  local combo_index_0=$((combo_index_1 - 1))

  if (( combo_index_1 <= RUN_INDEX_OFFSET )); then
    return 1
  fi

  if [[ "${RUN_INDEX_SET}" == "true" ]]; then
    if (( combo_index_1 == RUN_INDEX )); then
      return 0
    fi
    return 1
  fi

  if [[ "${RANGE_ENABLED}" == "true" ]]; then
    if (( combo_index_0 < RANGE_START || combo_index_0 > RANGE_END )); then
      return 1
    fi
  fi

  return 0
}

declare -a RUN_PLAN_INDICES=()
declare -a RUN_PLAN_MODELS=()
declare -a RUN_PLAN_WORKERS=()
declare -a RUN_PLAN_GLOBAL_BS=()
declare -a RUN_PLAN_WARMUPS=()
declare -a RUN_PLAN_STEPS=()
declare -a RUN_PLAN_VAL_FREQ=()
declare -i SKIPPED_GPU_RUNS=0

combination_index=0
for model_size in "${MODEL_SIZE_ARRAY[@]}"; do
  warmup_step="16M-baseline-20251029-095728/step-2048"
  for global_bs in "${GLOBAL_BATCH_ARRAY[@]}"; do
    for workers in "${WORKER_COUNT_ARRAY[@]}"; do
      ensure_divisible "${global_bs}" "${workers}"
      ((++combination_index))
      if ! should_run_combination "${combination_index}"; then
        continue
      fi
      if (( workers > TOTAL_GPUS )); then
        ((++SKIPPED_GPU_RUNS))
        echo "[GPU-SKIP] Combination ${combination_index} (model=${model_size}, workers=${workers}) requires ${workers} GPU(s) but only ${TOTAL_GPUS} detected; skipping." >&2
        continue
      fi
      scaled_steps=$(compute_scaled_steps "${global_bs}")
      val_freq=$(compute_scaled_validation_freq "${global_bs}")
      RUN_PLAN_INDICES+=("${combination_index}")
      RUN_PLAN_MODELS+=("${model_size}")
      RUN_PLAN_WORKERS+=("${workers}")
      RUN_PLAN_GLOBAL_BS+=("${global_bs}")
      RUN_PLAN_WARMUPS+=("${warmup_step}")
      RUN_PLAN_STEPS+=("${scaled_steps}")
      RUN_PLAN_VAL_FREQ+=("${val_freq}")
    done
  done
done

TOTAL_RUNS=${combination_index}
if [[ "${RUN_INDEX_SET}" == "true" ]]; then
  if (( RUN_INDEX < 1 || RUN_INDEX > TOTAL_RUNS )); then
    echo "RUN_INDEX must be between 1 and ${TOTAL_RUNS} (got ${RUN_INDEX})." >&2
    exit 1
  fi
fi

SELECTED_RUNS=${#RUN_PLAN_INDICES[@]}
if (( SELECTED_RUNS == 0 )); then
  if (( SKIPPED_GPU_RUNS > 0 )); then
    echo "All selected runs require more GPUs than available (${TOTAL_GPUS}). Provide additional GPU ids or adjust WORKER_COUNTS." >&2
    exit 1
  fi
  echo "No runs match the requested filters; nothing to do." >&2
  exit 0
fi
if (( SKIPPED_GPU_RUNS > 0 )); then
  echo "Skipped ${SKIPPED_GPU_RUNS} configuration(s) that exceeded the ${TOTAL_GPUS} GPU limit." >&2
fi

SWEEP_CONFIG_STRING="models=${MODEL_SIZES}|workers=${WORKER_COUNTS}|gbs=${GLOBAL_BATCH_SIZES}|lbs=${LOCAL_BATCH_SIZE}|lr=${BASE_LR}|vs=${VS_VALUE}|config=${CONFIG_FILE}|refg=${REFERENCE_GLOBAL_BATCH_SIZE}|refs=${REFERENCE_TRAINING_STEPS}|refv=${REFERENCE_VALIDATION_FREQ}"
if command -v sha1sum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | sha1sum | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | shasum -a 1 | awk '{print $1}')
else
  SWEEP_HASH=$(
    SWEEP_CONFIG_STRING="${SWEEP_CONFIG_STRING}" python3 - <<'PY'
import hashlib
import os
cfg = os.environ["SWEEP_CONFIG_STRING"]
print(hashlib.sha1(cfg.encode("utf-8")).hexdigest())
PY
  )
fi
SWEEP_HASH=${SWEEP_HASH:0:8}

declare -A GPU_IN_USE=()
for gpu in "${GPU_ARRAY[@]}"; do
  GPU_IN_USE["$gpu"]=0
done

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_GPU_LISTS=()
declare -a ACTIVE_LABELS=()

cleanup() {
  for pid in "${ACTIVE_PIDS[@]-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    wait "${ACTIVE_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

sanitize_value() {
  local value=$1
  value=${value//./p}
  value=${value//-/m}
  value=${value//+/p}
  echo "${value}"
}

mark_gpus_in_use() {
  local gpu_list=("$@")
  for gpu in "${gpu_list[@]}"; do
    GPU_IN_USE["$gpu"]=1
  done
}

release_gpus() {
  local gpu_list=("$@")
  for gpu in "${gpu_list[@]}"; do
    GPU_IN_USE["$gpu"]=0
  done
}

reap_finished_runs() {
  for i in "${!ACTIVE_PIDS[@]}"; do
    pid=${ACTIVE_PIDS[i]}
    if [[ -z "${pid}" ]]; then
      continue
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi
    wait "${pid}" 2>/dev/null || true
    gpu_list_str=${ACTIVE_GPU_LISTS[i]}
    if [[ -n "${gpu_list_str}" ]]; then
      IFS=',' read -r -a gpu_list <<< "${gpu_list_str}"
      release_gpus "${gpu_list[@]}"
    fi
    ACTIVE_PIDS[i]=''
    ACTIVE_GPU_LISTS[i]=''
    ACTIVE_LABELS[i]=''
  done
}

try_allocate_gpus() {
  local required=$1
  local require_all=$2
  local free=()
  GPU_ALLOCATION_RESULT=""
  for gpu in "${GPU_ARRAY[@]}"; do
    if [[ "${GPU_IN_USE[$gpu]}" -eq 0 ]]; then
      free+=("$gpu")
    fi
  done
  local free_count=${#free[@]}
  if [[ "${require_all}" == "true" ]]; then
    if (( free_count == TOTAL_GPUS )); then
      mark_gpus_in_use "${free[@]}"
      GPU_ALLOCATION_RESULT="$(IFS=','; echo "${free[*]}")"
      return 0
    fi
    return 1
  fi
  if (( free_count >= required )); then
    local selected=("${free[@]:0:${required}}")
    mark_gpus_in_use "${selected[@]}"
    GPU_ALLOCATION_RESULT="$(IFS=','; echo "${selected[*]}")"
    return 0
  fi
  return 1
}

wait_for_gpus() {
  local required=$1
  local require_all=$2
  while true; do
    reap_finished_runs
    if try_allocate_gpus "${required}" "${require_all}"; then
      return 0
    fi
    sleep "${POLL_INTERVAL_SEC}"
  done
}

start_run() {
  local run_uuid=$1
  local model_size=$2
  local workers=$3
  local global_bs=$4
  local warmup_step=$5
  local training_steps=$6
  local validation_freq=$7
  local gpu_csv=$8
  local rdzv_base_port=$9
  local lighthouse_port=${10}
  local combo_label=${11}
  local run_dir="${LOG_DIR}/${run_uuid}"
  mkdir -p "${run_dir}"

  (
    set -euo pipefail
    IFS=',' read -r -a run_gpu_list <<< "${gpu_csv}"
    if (( ${#run_gpu_list[@]} != workers )); then
      echo "[TorchFT] ${run_uuid} expected ${workers} GPUs but received ${#run_gpu_list[@]} (${gpu_csv})." >&2
      exit 1
    fi

    local min_replicas="${TORCHFT_MIN_REPLICAS:-}"
    if [[ -z "${min_replicas}" ]]; then
      min_replicas=${workers}
    fi

    local lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"
    local lighthouse_log="${run_dir}/lighthouse.log"
    local lighthouse_pid=""
    declare -a replica_pids=()

    cleanup_run() {
      for pid in "${replica_pids[@]-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
          kill "${pid}" 2>/dev/null || true
        fi
      done
      if [[ -n "${lighthouse_pid:-}" ]] && kill -0 "${lighthouse_pid}" 2>/dev/null; then
        kill "${lighthouse_pid}" 2>/dev/null || true
      fi
      if [[ -n "${replica_pids[*]-}" ]]; then
        wait "${replica_pids[@]}" 2>/dev/null || true
      fi
      if [[ -n "${lighthouse_pid:-}" ]]; then
        wait "${lighthouse_pid}" 2>/dev/null || true
      fi
    }
    trap cleanup_run EXIT

    uv run --no-sync torchft_lighthouse \
      --min_replicas "${min_replicas}" \
      --quorum_tick_ms "${TORCHFT_QUORUM_TICK_MS}" \
      --bind "${LIGHTHOUSE_HOST}:${lighthouse_port}" \
      > "${lighthouse_log}" 2>&1 &
    lighthouse_pid=$!
    sleep "${TORCHFT_LIGHTHOUSE_BOOT_DELAY}"
    if ! kill -0 "${lighthouse_pid}" 2>/dev/null; then
      echo "[TorchFT] Failed to start lighthouse for ${run_uuid}; see ${lighthouse_log}" >&2
      exit 1
    fi

    export RUN_UUID="${run_uuid}"
    export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan-workers-bs"}
    export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
    export WANDB_RUN_NAME="${RUN_UUID}"
    export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
    export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
    export RESUME_FROM_RUN_STEP="${warmup_step}"

    for ((replica_id = 0; replica_id < workers; ++replica_id)); do
      gpu_id=${run_gpu_list[replica_id]}
      replica_log="${run_dir}/replica_${replica_id}.log"
      replica_rdzv_port=$((rdzv_base_port + replica_id))

      (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        uv run --no-sync torchrun \
          --nproc_per_node=1 \
          --rdzv_backend=c10d \
          --rdzv_endpoint="${RDZV_HOST}:${replica_rdzv_port}" \
          --rdzv_id "${run_uuid}-replica-${replica_id}" \
          --local-ranks-filter="${LOG_RANK}" \
          --role rank \
          --tee 3 \
          -m "${TRAIN_MODULE}" \
          --job.config_file "${CONFIG_FILE}" \
          --model.flavor "${model_size}" \
          --fl_metrics.hyperparameter_switch.new_vs "${VS_VALUE}" \
          --training.global_batch_size "${global_bs}" \
          --training.local_batch_size "${LOCAL_BATCH_SIZE}" \
          --training.steps "${training_steps}" \
          --parallelism.data_parallel_replicate_degree 1 \
          --lr_scheduler.switch_step 2049 \
          --validation.freq "${validation_freq}" \
          --run_uuid "${RUN_UUID}" \
          "${TRAINING_ARGS[@]}"
      ) > "${replica_log}" 2>&1 &
      replica_pids[replica_id]=$!
      sleep 1
    done

    set +e
    replica_status=0
    for pid in "${replica_pids[@]}"; do
      if [[ -n "${pid}" ]]; then
        wait "${pid}"
        status=$?
        if (( status != 0 )); then
          replica_status=${status}
        fi
      fi
    done
    set -e

    if kill -0 "${lighthouse_pid}" 2>/dev/null; then
      kill "${lighthouse_pid}" 2>/dev/null || true
      wait "${lighthouse_pid}" 2>/dev/null || true
    fi

    if (( replica_status != 0 )); then
      exit "${replica_status}"
    fi
  ) &

  pid=$!
  ACTIVE_PIDS+=("${pid}")
  ACTIVE_GPU_LISTS+=("${gpu_csv}")
  ACTIVE_LABELS+=("${run_uuid}")
  echo "[TorchFT RUN] ${run_uuid} | combo=${combo_label} | model=${model_size} | workers=${workers} | gbs=${global_bs} | steps=${training_steps} | val_freq=${validation_freq} | GPUs=${gpu_csv} | logs=${run_dir}" >&2
}

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

print_run_plan() {
  local total=$1
  echo "" >&2
  echo "Selected MT-Dao configurations (${total} total | hash ${SWEEP_HASH}):" >&2
  printf "%-10s %-10s %-10s %-10s %-12s %-12s %-12s\n" "Idx(1)" "Idx(0)" "Model" "Workers" "GlobalBS" "Steps" "ValFreq" >&2
  printf "%-10s %-10s %-10s %-10s %-12s %-12s %-12s\n" "------" "------" "-----" "-------" "--------" "-----" "-------" >&2
  for i in "${!RUN_PLAN_INDICES[@]}"; do
    combo_index=${RUN_PLAN_INDICES[i]}
    combo_index_zero=$((combo_index - 1))
    printf "%-10s %-10s %-10s %-10s %-12s %-12s %-12s\n" \
      "${combo_index}" \
      "${combo_index_zero}" \
      "${RUN_PLAN_MODELS[i]}" \
      "${RUN_PLAN_WORKERS[i]}" \
      "${RUN_PLAN_GLOBAL_BS[i]}" \
      "${RUN_PLAN_STEPS[i]}" \
      "${RUN_PLAN_VAL_FREQ[i]}" >&2
  done
  echo "" >&2
}

echo "Starting MT-Dao sweep (${SELECTED_RUNS}/${TOTAL_RUNS} selected) with hash ${SWEEP_HASH} using GPU set: ${GPU_ARRAY[*]} (${FILTER_DESC})." >&2
print_run_plan "${SELECTED_RUNS}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY-RUN] No commands launched." >&2
  exit 0
fi

timestamp_global=$(date +"%Y%m%d-%H%M%S")
run_counter=0

for i in "${!RUN_PLAN_INDICES[@]}"; do
  combo_index=${RUN_PLAN_INDICES[i]}
  combo_index_zero=$((combo_index - 1))
  model_size=${RUN_PLAN_MODELS[i]}
  workers=${RUN_PLAN_WORKERS[i]}
  global_bs=${RUN_PLAN_GLOBAL_BS[i]}
  warmup_step=${RUN_PLAN_WARMUPS[i]}
  training_steps=${RUN_PLAN_STEPS[i]}
  validation_freq=${RUN_PLAN_VAL_FREQ[i]}
  model_tag=$(sanitize_value "${model_size}")
  run_uuid="${RUN_PREFIX}-${SWEEP_HASH}-${model_tag}-w${workers}-gb${global_bs}-${timestamp_global}-idx${combo_index_zero}"

  require_all="false"
  if (( workers == TOTAL_GPUS )); then
    require_all="true"
  elif (( workers > TOTAL_GPUS )); then
    echo "Run ${run_uuid} requires ${workers} workers but only ${TOTAL_GPUS} GPU ids provided." >&2
    exit 1
  fi

  wait_for_gpus "${workers}" "${require_all}"
  gpu_csv="${GPU_ALLOCATION_RESULT}"
  ((++run_counter))
  rdzv_base_port=$((RDZV_BASE_PORT + run_counter * PORT_STRIDE))
  lighthouse_port=$((LIGHTHOUSE_BASE_PORT + run_counter * PORT_STRIDE))

  start_run "${run_uuid}" "${model_size}" "${workers}" "${global_bs}" "${warmup_step}" "${training_steps}" "${validation_freq}" "${gpu_csv}" "${rdzv_base_port}" "${lighthouse_port}" "${combo_index}/${TOTAL_RUNS}"
done

# Wait for all remaining runs to complete.
for pid in "${ACTIVE_PIDS[@]}"; do
  if [[ -n "${pid}" ]]; then
    wait "${pid}" 2>/dev/null || true
  fi
done

echo "MT-Dao sweep complete (hash ${SWEEP_HASH}). Logs: ${LOG_DIR}" >&2
