#!/usr/bin/env bash
# TorchFT variant of the v-lr sweep. Launches 4 replicas (one per GPU) per run.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-vlr-torchft"}
LOG_RANK=${LOG_RANK:-0}
BASE_LR=${BASE_LR:-0.01}
WORKER_COUNT=${WORKER_COUNT:-4}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((WORKER_COUNT * LOCAL_BATCH_SIZE))}
TRAINING_STEPS=${TRAINING_STEPS:-12288}
LR_SWITCH_STEP=${LR_SWITCH_STEP:-2049}

RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}
RUN_INDEX_RANGE=${RUN_INDEX_RANGE:-}
DRY_RUN=${DRY_RUN:-false}

VS_VALUES=${VS_VALUES:-"0.98 0.99 0.993 0.996 0.999 0.9993 0.9996 0.9999"}
SWITCH_SCALES=${SWITCH_SCALES:-"4 8 16"}

GPU_IDS=${GPU_IDS:-"0 1 2 3"}
RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
RDZV_BASE_PORT=${RDZV_BASE_PORT:-53000}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-53100}
PORT_STRIDE=${PORT_STRIDE:-32}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}

TORCHFT_MIN_REPLICAS=${TORCHFT_MIN_REPLICAS:-${WORKER_COUNT}}
TORCHFT_QUORUM_TICK_MS=${TORCHFT_QUORUM_TICK_MS:-100}
TORCHFT_LIGHTHOUSE_BOOT_DELAY=${TORCHFT_LIGHTHOUSE_BOOT_DELAY:-2}

SWEEP_LOG_DIR=${SWEEP_LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${SWEEP_LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: run_adamw_sweep.sh [OPTIONS] [-- extra training args]

TorchFT-specific defaults:
  * 4 replicas (WORKER_COUNT) launched locally via torchft_lighthouse.
  * Each replica uses local batch size 16 (LOCAL_BATCH_SIZE).
  * Global batch size defaults to WORKER_COUNT * LOCAL_BATCH_SIZE.

Options:
  --range START-END        Run only the 0-indexed inclusive range of sweep jobs.
  --run-index INDEX        Run only the 1-indexed job at INDEX.
  --run-index-offset N     Skip the first N jobs before applying other filters.
  --dry-run                Print matching runs without launching anything.
  -h, --help               Show this message.
  --                       Treat the remaining arguments as training args.

Environment variables (override defaults):
  RUN_INDEX, RUN_INDEX_OFFSET, RUN_INDEX_RANGE, DRY_RUN
  CONFIG_FILE, TRAIN_MODULE, RUN_PREFIX, BASE_LR
  WORKER_COUNT, LOCAL_BATCH_SIZE, GLOBAL_BATCH_SIZE
  VS_VALUES, SWITCH_SCALES
  GPU_IDS (space or comma separated list of GPU ordinals)
  RDZV_HOST, RDZV_BASE_PORT, LIGHTHOUSE_HOST, LIGHTHOUSE_BASE_PORT, PORT_STRIDE
  TORCHFT_MIN_REPLICAS, TORCHFT_QUORUM_TICK_MS, TORCHFT_LIGHTHOUSE_BOOT_DELAY
  SWEEP_LOG_DIR
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

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
  exit 1
fi

if ! [[ "${WORKER_COUNT}" =~ ^[0-9]+$ ]] || (( WORKER_COUNT < 1 )); then
  echo "WORKER_COUNT must be a positive integer (got ${WORKER_COUNT})." >&2
  exit 1
fi

if ! [[ "${LOCAL_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( LOCAL_BATCH_SIZE < 1 )); then
  echo "LOCAL_BATCH_SIZE must be a positive integer (got ${LOCAL_BATCH_SIZE})." >&2
  exit 1
fi

if ! [[ "${GLOBAL_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( GLOBAL_BATCH_SIZE < 1 )); then
  echo "GLOBAL_BATCH_SIZE must be a positive integer (got ${GLOBAL_BATCH_SIZE})." >&2
  exit 1
fi

if (( GLOBAL_BATCH_SIZE % (WORKER_COUNT * LOCAL_BATCH_SIZE) != 0 )); then
  echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) should be divisible by WORKER_COUNT * LOCAL_BATCH_SIZE ($(WORKER_COUNT * LOCAL_BATCH_SIZE))." >&2
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
    echo "RUN_INDEX_RANGE must match START-END (e.g., 0-63)." >&2
    exit 1
  fi
fi

if [[ -n "${RUN_INDEX}" ]]; then
  if ! [[ "${RUN_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "RUN_INDEX must be a positive integer." >&2
    exit 1
  fi
  RUN_INDEX=$((10#${RUN_INDEX}))
fi

if [[ -n "${RUN_INDEX}" && RUN_INDEX_OFFSET -ne 0 ]]; then
  echo "RUN_INDEX and RUN_INDEX_OFFSET cannot both be set." >&2
  exit 1
fi

if [[ -n "${RUN_INDEX}" && "${RANGE_ENABLED}" == "true" ]]; then
  echo "RUN_INDEX and RUN_INDEX_RANGE cannot both be set." >&2
  exit 1
fi

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

TOTAL_RUNS=$(( ${#VS_ARRAY[@]} * ${#SWITCH_SCALE_ARRAY[@]} ))

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
  for sw_val in "${SWITCH_SCALE_ARRAY[@]}"; do
  for v_val in "${VS_ARRAY[@]}"; do
      ((++idx))
      if should_run_combination "${idx}"; then
        ((++selected))
      fi
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
declare -a RUN_PLAN_VS=()
declare -a RUN_PLAN_SWITCHES=()

combination_index=0
for switch_scale in "${SWITCH_SCALE_ARRAY[@]}"; do
  for v_value in "${VS_ARRAY[@]}"; do
    ((++combination_index))
    if ! should_run_combination "${combination_index}"; then
      continue
    fi
    RUN_PLAN_INDICES+=("${combination_index}")
    RUN_PLAN_VS+=("${v_value}")
    RUN_PLAN_SWITCHES+=("${switch_scale}")
  done
done

SWEEP_CONFIG_STRING="vs=${VS_VALUES}|switch=${SWITCH_SCALES}|base_lr=${BASE_LR}|train_module=${TRAIN_MODULE}|config=${CONFIG_FILE}|workers=${WORKER_COUNT}|lbs=${LOCAL_BATCH_SIZE}"
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

GPU_IDS_CLEAN=${GPU_IDS//,/ }
read -r -a GPU_ARRAY <<< "${GPU_IDS_CLEAN}"
if (( ${#GPU_ARRAY[@]} < WORKER_COUNT )); then
  echo "GPU_IDS must provide at least ${WORKER_COUNT} entries (got ${#GPU_ARRAY[@]})." >&2
  exit 1
fi
REPLICA_GPUS=("${GPU_ARRAY[@]:0:${WORKER_COUNT}}")

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
  echo "Selected TorchFT run configurations (${total} total):" >&2
  printf "%-10s %-10s %-14s %-18s\n" "Idx(1-based)" "Idx(0-based)" "v" "switch_scale" >&2
  printf "%-10s %-10s %-14s %-18s\n" "----------" "----------" "------------" "------------------" >&2
  for idx in "${!RUN_PLAN_INDICES[@]}"; do
    local combo_index_1=${RUN_PLAN_INDICES[idx]}
    local combo_index_0=$((combo_index_1 - 1))
    local v_value=${RUN_PLAN_VS[idx]}
    local switch_scale=${RUN_PLAN_SWITCHES[idx]}
    printf "%-10s %-10s %-14s %-18s\n" "${combo_index_1}" "${combo_index_0}" "${v_value}" "${switch_scale}" >&2
  done
  echo "" >&2
}

declare -a ACTIVE_REPLICA_PIDS=()
ACTIVE_LIGHTHOUSE_PID=""
declare -A REPLICA_PID_TO_ID=()

cleanup_active_processes() {
  if (( ${#ACTIVE_REPLICA_PIDS[@]} > 0 )); then
    for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
  fi
  if [[ -n "${ACTIVE_LIGHTHOUSE_PID:-}" ]] && kill -0 "${ACTIVE_LIGHTHOUSE_PID}" 2>/dev/null; then
    kill "${ACTIVE_LIGHTHOUSE_PID}" 2>/dev/null || true
  fi
}
trap cleanup_active_processes EXIT INT TERM

start_lighthouse() {
  local host=$1
  local port=$2
  local log_file=$3

  uv run --no-sync torchft_lighthouse \
    --min_replicas "${TORCHFT_MIN_REPLICAS}" \
    --quorum_tick_ms "${TORCHFT_QUORUM_TICK_MS}" \
    --bind "${host}:${port}" \
    > "${log_file}" 2>&1 &

  local pid=$!
  sleep "${TORCHFT_LIGHTHOUSE_BOOT_DELAY}"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Failed to start torchft_lighthouse (see ${log_file})." >&2
    return 1
  fi
  echo "${pid}"
}

stop_lighthouse() {
  local pid=$1
  if [[ -z "${pid}" ]]; then
    return
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

launch_replica_process() {
  local replica_id=$1
  local gpu_id=$2
  local log_file=$3
  local rdzv_port=$4
  local rdzv_id=$5
  local lighthouse_url=$6
  local v_value=$7
  local switch_scale=$8

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
    export RUN_UUID="${RUN_UUID}"
    export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_tune_Lr"}
    export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
    export WANDB_RUN_NAME="${RUN_UUID}"
    export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}
    export S3_ENDPOINT_URL='http://taranaki.cl.cam.ac.uk:9000'

    cmd=(uv run --no-sync torchrun
      --nproc_per_node=1
      --rdzv_backend=c10d
      --rdzv_endpoint="${RDZV_HOST}:${rdzv_port}"
      --rdzv_id "${rdzv_id}"
      --local-ranks-filter="${LOG_RANK}"
      --role rank
      --tee 3
      -m "${TRAIN_MODULE}"
      --job.config_file "${CONFIG_FILE}"
      --model.flavor "16M"
      --optimizer.builder mosaic
      --optimizer.name QHAdamW
      --optimizer.lr "${BASE_LR}"
      --optimizer.beta1 0.999
      --optimizer.beta2 0.999
      --optimizer.vs "${v_value}"
      --fl_metrics.hyperparameter_switch.new_vs "${v_value}"
      --training.global_batch_size "${GLOBAL_BATCH_SIZE}"
      --training.local_batch_size "${LOCAL_BATCH_SIZE}"
      --training.steps "${TRAINING_STEPS}"
      --lr_scheduler.switch_step "${LR_SWITCH_STEP}"
      --lr_scheduler.switch_scale "${switch_scale}"
      --parallelism.data_parallel_replicate_degree 1
      --fault_tolerance.replica_id "${replica_id}"
      --fault_tolerance.group_size "${WORKER_COUNT}"
      --fault_tolerance.min_replica_size "${TORCHFT_MIN_REPLICAS}"
    )

    if (( ${#TRAINING_ARGS[@]} > 0 )); then
      cmd+=("${TRAINING_ARGS[@]}")
    fi

    "${cmd[@]}"
  ) > "${log_file}" 2>&1 &

  echo $!
}

wait_for_replicas() {
  local status=0
  set +e
  while true; do
    local -a pending_pids=()
    for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
      if [[ -n "${pid}" ]]; then
        pending_pids+=("${pid}")
      fi
    done

    if (( ${#pending_pids[@]} == 0 )); then
      break
    fi

    local finished_pid=""
    wait -n -p finished_pid "${pending_pids[@]}"
    local exit_code=$?
    if (( exit_code == 127 )); then
      # Shell believes there are no unwaited children. Bail out so cleanup can run.
      status=$(( status == 0 ? 127 : status ))
      break
    fi

    local replica_idx="${REPLICA_PID_TO_ID[${finished_pid}]:-unknown}"
    if (( exit_code == 0 )); then
      echo "Replica ${replica_idx} (pid ${finished_pid}) completed successfully." >&2
    else
      echo "Replica ${replica_idx} (pid ${finished_pid}) exited with status ${exit_code}." >&2
      status=${exit_code}
    fi
    unset "REPLICA_PID_TO_ID[${finished_pid}]"

    for idx in "${!ACTIVE_REPLICA_PIDS[@]}"; do
      if [[ "${ACTIVE_REPLICA_PIDS[idx]}" == "${finished_pid}" ]]; then
        ACTIVE_REPLICA_PIDS[idx]=""
        break
      fi
    done

    if (( exit_code != 0 )); then
      break
    fi
  done
  set -e

  if (( status != 0 )); then
    for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
    for pid in "${ACTIVE_REPLICA_PIDS[@]}"; do
      if [[ -n "${pid}" ]]; then
        wait "${pid}" 2>/dev/null || true
      fi
    done
  fi

  ACTIVE_REPLICA_PIDS=()
  REPLICA_PID_TO_ID=()
  return "${status}"
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

echo "Starting TorchFT v-lr sweep hash=${SWEEP_HASH}: ${SELECTED_RUNS}/${TOTAL_RUNS} run(s) selected (${FILTER_DESC})." >&2
print_run_plan_table "${SELECTED_RUNS}"

timestamp_global=$(date +"%Y%m%d-%H%M%S")
dispatched_runs=0

for idx in "${!RUN_PLAN_INDICES[@]}"; do
  combination_index=${RUN_PLAN_INDICES[idx]}
  combination_index_zero=$((combination_index - 1))
  v_value=${RUN_PLAN_VS[idx]}
  switch_scale=${RUN_PLAN_SWITCHES[idx]}
  v_tag=$(sanitize_value "${v_value}")
  scale_tag=$(sanitize_value "${switch_scale}")
  run_uuid="${RUN_PREFIX}-${SWEEP_HASH}-v${v_tag}-sw${scale_tag}-${timestamp_global}-idx${combination_index_zero}"
  run_progress="run ${combination_index}/${TOTAL_RUNS}"

  lighthouse_port=$((LIGHTHOUSE_BASE_PORT + dispatched_runs * PORT_STRIDE))
  rdzv_base_port=$((RDZV_BASE_PORT + dispatched_runs * PORT_STRIDE))
  lighthouse_url="${LIGHTHOUSE_PROTOCOL}://${LIGHTHOUSE_HOST}:${lighthouse_port}"
  run_log_dir="${SWEEP_LOG_DIR}/${run_uuid}"

  echo "[TorchFT] ${run_uuid} | v=${v_value}, switch_scale=${switch_scale} | ${run_progress}" >&2
  echo "          Lighthouse: ${lighthouse_url} | Rendezvous base: ${RDZV_HOST}:${rdzv_base_port}" >&2

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[DRY-RUN] Would launch run %s on GPUs: %s\n' "${run_uuid}" "${REPLICA_GPUS[*]}" >&2
    ((++dispatched_runs))
    continue
  fi

  mkdir -p "${run_log_dir}"
  lighthouse_log="${run_log_dir}/lighthouse.log"

  ACTIVE_LIGHTHOUSE_PID=$(start_lighthouse "${LIGHTHOUSE_HOST}" "${lighthouse_port}" "${lighthouse_log}") || {
    rm -rf "${run_log_dir}"
    echo "Aborting sweep: failed to start lighthouse for ${run_uuid}." >&2
    exit 1
  }

  export RUN_UUID="${run_uuid}"
  export TORCHFT_LIGHTHOUSE="${lighthouse_url}"

  ACTIVE_REPLICA_PIDS=()
  REPLICA_PID_TO_ID=()
  for replica_id in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu_id=${REPLICA_GPUS[$replica_id]}
    replica_log="${run_log_dir}/replica_${replica_id}.log"
    rdzv_port=$((rdzv_base_port + replica_id))
    rdzv_id="${run_uuid}-replica-${replica_id}"
    replica_pid=$(launch_replica_process "${replica_id}" "${gpu_id}" "${replica_log}" "${rdzv_port}" "${rdzv_id}" "${lighthouse_url}" "${v_value}" "${switch_scale}")
    ACTIVE_REPLICA_PIDS+=("${replica_pid}")
    REPLICA_PID_TO_ID["${replica_pid}"]="${replica_id}"
    echo "  Replica ${replica_id} -> GPU ${gpu_id} (log: ${replica_log})" >&2
    sleep 1
  done

  if ! wait_for_replicas; then
    stop_lighthouse "${ACTIVE_LIGHTHOUSE_PID}"
    ACTIVE_LIGHTHOUSE_PID=""
    echo "TorchFT run ${run_uuid} failed; see ${run_log_dir} for details." >&2
    exit 1
  fi

  stop_lighthouse "${ACTIVE_LIGHTHOUSE_PID}"
  ACTIVE_LIGHTHOUSE_PID=""

  echo "Completed TorchFT run ${run_uuid}." >&2
  ((++dispatched_runs))
done

echo "TorchFT sweep complete (${dispatched_runs} launch(es))." >&2
