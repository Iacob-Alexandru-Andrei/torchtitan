#!/usr/bin/env bash
# Sweep QHAdamW hyperparameters (single new_v + switch-scale) for the 16M flavor.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-qhadam"}
LOG_RANK=${LOG_RANK:-0}
BASE_LR=${BASE_LR:-0.01}
WARMUP_SWITCH_STEP=${WARMUP_SWITCH_STEP:-2048}
SWITCH_APPLY_STEP=${SWITCH_APPLY_STEP:-$((WARMUP_SWITCH_STEP + 1))}

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

usage() {
  cat <<'EOF'
Usage: run_aggmo_switch_sweep.sh [OPTIONS] [-- extra training args]

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
  RUN_INDEX, RUN_INDEX_OFFSET, RUN_INDEX_RANGE, USE_SBATCH, DRY_RUN
  SBATCH_* variables for Slurm submission
  GPU_IDS (space separated, used for local launches)
  BASE_LR, WARMUP_SWITCH_STEP, SWITCH_APPLY_STEP, CONFIG_FILE
  QHADAM_NEW_V_VALUES (space-separated floats between 0 and 1)
  SWITCH_SCALE_VALUES (space-separated numeric values)
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

QHADAM_NEW_V_VALUES=${QHADAM_NEW_V_VALUES:-"0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99"}
SWITCH_SCALE_VALUES=${SWITCH_SCALE_VALUES:-"1 2 4 8 16"}
BASE_VS=${BASE_VS:-"1.0"}
BASE_BETAS=${BASE_BETAS:-"0.9 0.999"}

read -r -a NEW_V_ARRAY <<< "${QHADAM_NEW_V_VALUES}"
read -r -a SWITCH_SCALE_ARRAY <<< "${SWITCH_SCALE_VALUES}"
read -r -a BASE_VS_ARRAY <<< "${BASE_VS}"
read -r -a BASE_BETAS_ARRAY <<< "${BASE_BETAS}"

if (( ${#NEW_V_ARRAY[@]} == 0 )); then
  echo "QHADAM_NEW_V_VALUES must contain at least one entry." >&2
  exit 1
fi
if (( ${#SWITCH_SCALE_ARRAY[@]} == 0 )); then
  echo "SWITCH_SCALE_VALUES must contain at least one entry." >&2
  exit 1
fi
if (( ${#BASE_VS_ARRAY[@]} == 0 )); then
  echo "BASE_VS must contain at least one entry." >&2
  exit 1
fi
if (( ${#BASE_BETAS_ARRAY[@]} == 0 )); then
  echo "BASE_BETAS must contain at least one entry." >&2
  exit 1
fi

if (( ${#BASE_BETAS_ARRAY[@]} < 2 )); then
  echo "BASE_BETAS must contain at least two entries (beta1 beta2)." >&2
  exit 1
fi
QH_BETA1=${BASE_BETAS_ARRAY[0]}
QH_BETA2=${BASE_BETAS_ARRAY[1]}

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

validate_new_v() {
  local raw_value=$1
  if [[ -z "${raw_value}" ]]; then
    echo "Invalid new_v entry '${raw_value}' (empty value)." >&2
    exit 1
  fi
  python3 - <<PY
value = float("${raw_value}")
if not (0.0 <= value <= 1.0):
    raise SystemExit("new_v entries must be within [0,1].")
PY
}

for new_v in "${NEW_V_ARRAY[@]}"; do
  validate_new_v "${new_v}"
done

TOTAL_RUNS=$(( ${#NEW_V_ARRAY[@]} * ${#SWITCH_SCALE_ARRAY[@]} ))

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
else
  RANGE_ENABLED=false
  RANGE_START=
  RANGE_END=
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
    [[ ${combo_index_1} -eq ${RUN_INDEX} ]] && return 0 || return 1
  fi
  if [[ "${RANGE_ENABLED}" == "true" ]]; then
    if (( combo_index_0 < RANGE_START || combo_index_0 > RANGE_END )); then
      return 1
    fi
  fi
  return 0
}

count_selected_runs() {
  local idx=0
  local selected=0
  for new_v in "${NEW_V_ARRAY[@]}"; do
    for scale in "${SWITCH_SCALE_ARRAY[@]}"; do
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
declare -a RUN_PLAN_NEW_V=()
declare -a RUN_PLAN_SWITCHES=()

combination_index=0
for new_v in "${NEW_V_ARRAY[@]}"; do
  for switch_scale in "${SWITCH_SCALE_ARRAY[@]}"; do
    ((++combination_index))
    if ! should_run_combination "${combination_index}"; then
      continue
    fi
    RUN_PLAN_INDICES+=("${combination_index}")
    RUN_PLAN_NEW_V+=("${new_v}")
    RUN_PLAN_SWITCHES+=("${switch_scale}")
  done
done

SWEEP_CONFIG_STRING="new_v=${QHADAM_NEW_V_VALUES}|switch=${SWITCH_SCALE_VALUES}|lr=${BASE_LR}|switch_step=${WARMUP_SWITCH_STEP}|config=${CONFIG_FILE}"
if command -v sha1sum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | sha1sum | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  SWEEP_HASH=$(printf "%s" "${SWEEP_CONFIG_STRING}" | shasum -a 1 | awk '{print $1}')
else
  SWEEP_HASH=$(
    SWEEP_CONFIG_STRING="${SWEEP_CONFIG_STRING}" python3 - <<'PY'
import hashlib, os
config = os.environ["SWEEP_CONFIG_STRING"]
print(hashlib.sha1(config.encode("utf-8")).hexdigest())
PY
  )
fi
SWEEP_HASH=${SWEEP_HASH:0:8}

GPU_IDS=${GPU_IDS:-""}
RDZV_HOST=${RDZV_HOST:-"127.0.0.1"}
RDZV_BASE_PORT=${RDZV_BASE_PORT:-46200}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"127.0.0.1"}
LIGHTHOUSE_BASE_PORT=${LIGHTHOUSE_BASE_PORT:-46300}
PORT_STRIDE=${PORT_STRIDE:-4}
LIGHTHOUSE_PROTOCOL=${LIGHTHOUSE_PROTOCOL:-"http"}
POLL_INTERVAL_SEC=${POLL_INTERVAL_SEC:-5}

declare -a GPU_ARRAY=()
declare -i GPU_IDS_DEDUPED_COUNT=0
GPU_IDS_AUTO_SOURCE=""

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

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Base config not found at ${CONFIG_FILE}" >&2
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

declare -a GPU_PIDS=()
declare -a GPU_LABELS=()
declare -a ALL_PIDS=()
declare -a SBATCH_JOB_IDS=()
declare -a SBATCH_CHAIN_LAST_IDS=()
RUN_COUNTER=0

if [[ "${USE_SBATCH}" == "true" && "${DRY_RUN}" != "true" ]]; then
  command -v sbatch >/dev/null 2>&1 || { echo "USE_SBATCH=true but sbatch not found." >&2; exit 1; }
  [[ -z "${SBATCH_LOG_DIR}" ]] || mkdir -p "${SBATCH_LOG_DIR}"
  for ((chain_idx = 0; chain_idx < SBATCH_MAX_CHAINS; ++chain_idx)); do
    SBATCH_CHAIN_LAST_IDS[chain_idx]=''
  done
fi

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

print_run_plan() {
  echo "" >&2
  echo "Selected run configurations (${#RUN_PLAN_INDICES[@]} total):" >&2
  printf "%-10s %-10s %-18s %-18s\n" "Idx(1-based)" "Idx(0-based)" "new_v" "switch_scale" >&2
  printf "%-10s %-10s %-18s %-18s\n" "----------" "----------" "-----------------" "------------------" >&2
  for idx in "${!RUN_PLAN_INDICES[@]}"; do
    combo_index_1=${RUN_PLAN_INDICES[idx]}
    combo_index_0=$((combo_index_1 - 1))
    printf "%-10s %-10s %-18s %-18s\n" \
      "${combo_index_1}" \
      "${combo_index_0}" \
      "${RUN_PLAN_NEW_V[idx]}" \
      "${RUN_PLAN_SWITCHES[idx]}" >&2
  done
  echo "" >&2
}

formatted_new_v_arg() {
  local value=$1
  printf '%s' "${value}"
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
  local dependency_job=$8
  local chain_index=${9}

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

  local new_v_arg
  new_v_arg=$(formatted_new_v_arg "${new_v}")
  local switch_steps_arg
  switch_steps_arg=$(formatted_switch_steps_arg)

  local job_id
  if ! job_id=$(sbatch "${sbatch_opts[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "==================================================================="
echo "STARTING SBATCH JOB: ${job_name} (ID: \$SLURM_JOB_ID)"
echo "Run UUID: ${run_uuid} | Progress: ${run_progress} | new_v=${new_v_arg} | switch_scale=${switch_scale}"
echo "Chain Index: ${chain_index} | Dependency: ${dependency_job:-none}"
echo "Node: \$(hostname) at \$(date)"
echo "==================================================================="
      export TORCHFT_LIGHTHOUSE="${lighthouse_url}"
      export RUN_UUID="${run_uuid}"
export WANDB_PROJECT=\${WANDB_PROJECT:-"torchtitan_tune_Lr"}
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
  --job.config_file "${CONFIG_FILE}" \
  --model.flavor "16M" \
  --optimizer.builder mosaic \
  --optimizer.name QHAdamW \
  --optimizer.lr "${BASE_LR}" \
  --optimizer.vs "${BASE_VS_ARRAY[@]}" \
  --optimizer.beta1 "${QH_BETA1}" \
  --optimizer.beta2 "${QH_BETA2}" \
  --training.global_batch_size 64 \
  --training.local_batch_size 32 \
  --training.steps 12288 \
  --lr_scheduler.switch_step "${WARMUP_SWITCH_STEP}" \
  --lr_scheduler.switch_scale "${switch_scale}" \
  --fl_metrics.hyperparameter_switch.steps "${switch_steps_arg}" \
  --fl_metrics.hyperparameter_switch.new_vs "${new_v}" \
  --parallelism.data_parallel_replicate_degree 1 \
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

echo "Starting QHAdamW sweep hash=${SWEEP_HASH} using ${EXECUTION_DESC}: ${#RUN_PLAN_INDICES[@]}/${TOTAL_RUNS} run(s) selected (${FILTER_DESC})." >&2
print_run_plan

timestamp_global=$(date +"%Y%m%d-%H%M%S")
dispatched_runs=0

for idx in "${!RUN_PLAN_INDICES[@]}"; do
  combination_index=${RUN_PLAN_INDICES[idx]}
  combination_index_zero=$((combination_index - 1))
  new_v=${RUN_PLAN_NEW_V[idx]}
  switch_scale=${RUN_PLAN_SWITCHES[idx]}
  v_tag=$(sanitize_value "${new_v}")
  scale_tag=$(sanitize_value "${switch_scale}")
  run_uuid="${RUN_PREFIX}-${SWEEP_HASH}-v${v_tag}-sw${scale_tag}-${timestamp_global}-idx${combination_index_zero}"
  run_progress="run ${combination_index}/${TOTAL_RUNS}"
  new_v_arg=$(formatted_new_v_arg "${new_v}")
  switch_steps_arg=$(formatted_switch_steps_arg)

  if [[ "${DRY_RUN}" == "true" ]]; then
    if [[ "${USE_SBATCH}" == "true" ]]; then
      echo "[DRY-RUN][SBATCH] ${run_uuid} (new_v=${new_v_arg}, switch_scale=${switch_scale}) [${run_progress}]" >&2
    else
      echo "[DRY-RUN][LOCAL] ${run_uuid} (new_v=${new_v_arg}, switch_scale=${switch_scale}) [${run_progress}]" >&2
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
    job_id=$(submit_sbatch_job "${run_uuid}" "${run_progress}" "${rdzv_endpoint}" "${lighthouse_url}" "${combination_index_zero}" "${new_v}" "${switch_scale}" "${dependency_job}" "${chain_index}")
    if [[ -z "${job_id}" ]]; then
      echo "Aborting sweep after failed sbatch submission." >&2
      exit 1
    fi
    SBATCH_CHAIN_LAST_IDS[chain_index]="${job_id}"
    SBATCH_JOB_IDS+=("${job_id}")
    echo "[SBATCH] Submitted ${run_uuid} (${run_progress}) as job ${job_id} | new_v=${new_v_arg}, switch_scale=${switch_scale}" >&2
    sleep 30
  else
    gpu_slot=$(wait_for_gpu)
    gpu_id=${GPU_ARRAY[gpu_slot]}
    echo "[GPU ${gpu_id}] Launching run ${run_uuid} (new_v=${new_v_arg}, switch_scale=${switch_scale}) [${run_progress}]" >&2
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
  --job.config_file "${CONFIG_FILE}" \
  --model.flavor "16M" \
  --optimizer.builder mosaic \
  --optimizer.name QHAdamW \
  --optimizer.lr "${BASE_LR}" \
  --optimizer.vs "${BASE_VS_ARRAY[@]}" \
  --optimizer.beta1 "${QH_BETA1}" \
  --optimizer.beta2 "${QH_BETA2}" \
  --training.global_batch_size 64 \
  --training.local_batch_size 32 \
  --training.steps 12288 \
  --lr_scheduler.switch_step "${WARMUP_SWITCH_STEP}" \
  --lr_scheduler.switch_scale "${switch_scale}" \
  --fl_metrics.hyperparameter_switch.steps "${switch_steps_arg}" \
  --fl_metrics.hyperparameter_switch.new_vs "${new_v}" \
  --parallelism.data_parallel_replicate_degree 1 \
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
  echo "Dry run complete: ${#RUN_PLAN_INDICES[@]} run(s) matched the filters (no jobs launched)." >&2
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
