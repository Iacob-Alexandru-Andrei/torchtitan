#!/usr/bin/env bash
# Launch the AdeMaMix TorchFT (\"mt_dao\") experiment with TorchFT replicas.
set -euo pipefail

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
USE_SBATCH=${USE_SBATCH:-false}
DRY_RUN=${DRY_RUN:-false}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-32}
SBATCH_GPUS_PER_TASK=${SBATCH_GPUS_PER_TASK:-4}
SBATCH_MEM=${SBATCH_MEM:-}
SBATCH_TIME=${SBATCH_TIME:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_QOS=${SBATCH_QOS:-}
SBATCH_CONSTRAINT=${SBATCH_CONSTRAINT:-}
SBATCH_COMMENT=${SBATCH_COMMENT:-}
SBATCH_NODE=${SBATCH_NODE:-}
SBATCH_LOG_DIR=${SBATCH_LOG_DIR:-"${SCRIPT_DIR}/logs"}
SBATCH_ADDITIONAL_ARGS=${SBATCH_ADDITIONAL_ARGS:-}

usage() {
  cat <<'EOF'
Usage: mt_dao_torchft.sh [--sbatch|--no-sbatch] [--dry-run] [-- ...training args...]
EOF
}

normalize_bool() {
  local value="${1:-}"
  case "${value,,}" in
    true|1|yes|on) echo "true" ;;
    false|0|no|off|"") echo "false" ;;
    *) echo "${value,,}" ;;
  esac
}

serialize_command() {
  if (( $# == 0 )); then
    echo ""
    return
  fi
  local args=("$@")
  local serialized
  serialized=$(printf '%q' "${args[0]}")
  for ((i=1; i<${#args[@]}; i++)); do
    serialized+=" $(printf '%q' "${args[i]}")"
  done
  printf '%s' "${serialized}"
}

POSITIONAL_ARGS=()
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
    --)
      shift
      POSITIONAL_ARGS+=("$@")
      break
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

USE_SBATCH=$(normalize_bool "${USE_SBATCH}")
DRY_RUN=$(normalize_bool "${DRY_RUN}")
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd -P)
else
  REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../.." && pwd -P)
fi
cd "${REPO_ROOT}"
export REPO_ROOT

# Remove stale shared-memory artifacts owned by the current user.
find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi

CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/diloco_torchft.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
MIN_REPLICAS=${MIN_REPLICAS:-${NGPU}}
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}

LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"localhost"}
LIGHTHOUSE_PORT=${LIGHTHOUSE_PORT:-29510}
LIGHTHOUSE_URL="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"

LR_SWITCH_STEP=${LR_SWITCH_STEP:-2049}
ADEMAMIX_SWITCH_SCALE=${ADEMAMIX_SWITCH_SCALE:-2.0}
ADEMAMIX_NEW_VS=${ADEMAMIX_NEW_VS:-"0.95"}
ADEMAMIX_NEW_BETAS=${ADEMAMIX_NEW_BETAS:-"0.9 0.999"}
ADEMAMIX_RESET_MOMENTA=${ADEMAMIX_RESET_MOMENTA:-"exp_avg exp_avg_2"}

read -r -a ADEMAMIX_NEW_VS_ARRAY <<< "${ADEMAMIX_NEW_VS}"
read -r -a ADEMAMIX_NEW_BETAS_ARRAY <<< "${ADEMAMIX_NEW_BETAS}"
read -r -a ADEMAMIX_RESET_MOMENTA_ARRAY <<< "${ADEMAMIX_RESET_MOMENTA}"

TRAINING_ARGS=("${POSITIONAL_ARGS[@]}")

if [[ "${USE_SBATCH}" == "true" && -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${SBATCH_LOG_DIR}"
  declare -a sbatch_opts
  job_name="mt-dao"
  sbatch_opts+=(--parsable "-c" "${SBATCH_CPUS_PER_TASK}" "--gres=gpu:${SBATCH_GPUS_PER_TASK}" "--job-name=${job_name}")
  [[ -n "${SBATCH_MEM}" ]] && sbatch_opts+=("--mem=${SBATCH_MEM}")
  [[ -n "${SBATCH_TIME}" ]] && sbatch_opts+=("--time=${SBATCH_TIME}")
  [[ -n "${SBATCH_PARTITION}" ]] && sbatch_opts+=("--partition=${SBATCH_PARTITION}")
  [[ -n "${SBATCH_ACCOUNT}" ]] && sbatch_opts+=("--account=${SBATCH_ACCOUNT}")
  [[ -n "${SBATCH_QOS}" ]] && sbatch_opts+=("--qos=${SBATCH_QOS}")
  [[ -n "${SBATCH_CONSTRAINT}" ]] && sbatch_opts+=("--constraint=${SBATCH_CONSTRAINT}")
  [[ -n "${SBATCH_COMMENT}" ]] && sbatch_opts+=("--comment=${SBATCH_COMMENT}")
  [[ -n "${SBATCH_NODE}" ]] && sbatch_opts+=("--nodelist=${SBATCH_NODE}")
  sbatch_opts+=("--output=${SBATCH_LOG_DIR}/%j-mtdao.out" "--error=${SBATCH_LOG_DIR}/%j-mtdao.err")
  read -r -a sbatch_extra_array <<< "${SBATCH_ADDITIONAL_ARGS}"
  sbatch_opts+=("${sbatch_extra_array[@]:-}")

  child_cmd=("${SCRIPT_PATH}" "--no-sbatch")
  [[ "${DRY_RUN}" == "true" ]] && child_cmd+=("--dry-run")
  if (( ${#TRAINING_ARGS[@]} )); then
    child_cmd+=("--")
    child_cmd+=("${TRAINING_ARGS[@]}")
  fi
  CHILD_CMD_ESCAPED=$(serialize_command "${child_cmd[@]}")

  SBATCH_JOB_BODY=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "==================================================================="
echo "STARTING SBATCH JOB: ${job_name} (ID: \$SLURM_JOB_ID)"
echo "Node: \$(hostname) at \$(date)"
echo "Command: ${CHILD_CMD_ESCAPED}"
echo "==================================================================="
export USE_SBATCH=false
${CHILD_CMD_ESCAPED}
echo "JOB FINISHED: \$(date)"
EOF
)

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[MT-DAO] sbatch %s <<EOF\n%s\nEOF\n' "${sbatch_opts[*]}" "${SBATCH_JOB_BODY}"
    exit 0
  fi

  job_id=$(sbatch "${sbatch_opts[@]}" <<EOF
${SBATCH_JOB_BODY}
EOF
)
  status=$?
  if (( status != 0 )); then
    echo "Failed to submit sbatch job (exit ${status})." >&2
    exit "${status}"
  fi
  echo "[MT-DAO] Submitted job ${job_id}"
  exit 0
fi

if [[ "${USE_SBATCH}" == "true" && -n "${SLURM_JOB_ID:-}" ]]; then
  USE_SBATCH=false
fi

LOG_DIR="${REPO_ROOT}/outputs/dill_logs"
mkdir -p "${LOG_DIR}"
LIGHTHOUSE_LOG_FILE="${LOG_DIR}/lighthouse.log"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-strdill125M"}
export RUN_UUID=${RUN_UUID:-"${RUN_PREFIX}-${TIMESTAMP}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_streaming"}
export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
export WANDB_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}

echo "=========================================="
echo "TorchFT AdeMaMix Launch"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Config   : ${CONFIG_FILE}"
echo "Replicas : ${NGPU}"
echo "Lighthouse: ${LIGHTHOUSE_URL}"
echo "Log dir  : ${LOG_DIR}"
echo "=========================================="

declare -a REPLICA_PIDS=()
LIGHTHOUSE_PID=""

cleanup() {
  set +e
  for pid in "${REPLICA_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${LIGHTHOUSE_PID}" ]] && kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
    kill "${LIGHTHOUSE_PID}" 2>/dev/null || true
    wait "${LIGHTHOUSE_PID}" 2>/dev/null || true
  fi
  set -e
}
trap cleanup EXIT INT TERM

echo "[Lighthouse] starting at ${LIGHTHOUSE_URL}"
uv run --no-sync torchft_lighthouse \
  --min_replicas "${MIN_REPLICAS}" \
  --quorum_tick_ms "${QUORUM_TICK_MS}" \
  --bind "${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}" \
  > "${LIGHTHOUSE_LOG_FILE}" 2>&1 &
LIGHTHOUSE_PID=$!
sleep 2
if ! kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
  echo "ERROR: torchft_lighthouse failed to start. Check ${LIGHTHOUSE_LOG_FILE}" >&2
  exit 1
fi
echo "Lighthouse PID: ${LIGHTHOUSE_PID}"

export TORCHFT_LIGHTHOUSE="${LIGHTHOUSE_URL}"

AVAILABLE_GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((i=0; i<NGPU; i++)); do
    AVAILABLE_GPUS+=("${i}")
  done
fi

if (( ${#AVAILABLE_GPUS[@]} < NGPU )); then
  echo "ERROR: Requested ${NGPU} replicas but only ${#AVAILABLE_GPUS[@]} GPU(s) available." >&2
  exit 1
fi

REPLICA_GPUS=("${AVAILABLE_GPUS[@]:0:NGPU}")
for ((i=0; i<NGPU; i++)); do
  echo "Replica ${i} -> GPU ${REPLICA_GPUS[$i]}"
done

for ((replica_id=0; replica_id<NGPU; replica_id++)); do
  gpu_id="${REPLICA_GPUS[$replica_id]}"
  log_file="${LOG_DIR}/replica_${replica_id}.log"
  echo "[Replica ${replica_id}] logging to ${log_file}"

  (
    set -euo pipefail
    cd "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    rdzv_port=$((29600 + replica_id))
    uv run --no-sync torchrun \
      --nproc_per_node=1 \
      --rdzv_backend=c10d \
      --rdzv_endpoint="localhost:${rdzv_port}" \
      --role rank \
      --tee 3 \
      -m "${TRAIN_MODULE}" \
      --job.config_file "${CONFIG_FILE}" \
      --run_uuid "${RUN_UUID}" \
      --fault_tolerance.replica_id "${replica_id}" \
      --fault_tolerance.group_size "${NGPU}" \
      --fault_tolerance.min_replica_size "${MIN_REPLICAS}" \
      "${TRAINING_ARGS[@]}"
  ) > "${log_file}" 2>&1 &
  REPLICA_PIDS[$replica_id]=$!
  sleep 1
done

echo ""
echo "Lighthouse log: tail -f ${LIGHTHOUSE_LOG_FILE}"
for ((i=0; i<NGPU; i++)); do
  echo "Replica ${i} log: tail -f ${LOG_DIR}/replica_${i}.log"
done
echo ""

set +e
REPLICA_EXIT=0
for ((replica_id=0; replica_id<NGPU; replica_id++)); do
  pid=${REPLICA_PIDS[$replica_id]}
  if wait "${pid}"; then
    echo "Replica ${replica_id} completed successfully."
  else
    status=$?
    echo "Replica ${replica_id} exited with status ${status}."
    REPLICA_EXIT=${status}
  fi
done
set -e

if [[ -n "${LIGHTHOUSE_PID}" ]] && kill -0 "${LIGHTHOUSE_PID}" 2>/dev/null; then
  echo "Stopping lighthouse..."
  kill "${LIGHTHOUSE_PID}" 2>/dev/null || true
  wait "${LIGHTHOUSE_PID}" 2>/dev/null || true
  LIGHTHOUSE_PID=""
fi

if (( REPLICA_EXIT == 0 )); then
  echo "All replicas completed successfully!"
else
  echo "Some replicas failed. Check logs in ${LOG_DIR}/"
fi

exit "${REPLICA_EXIT}"
