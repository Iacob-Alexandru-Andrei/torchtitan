#!/usr/bin/env bash
# Launch a GaLoreGlobal TorchFT sweep where each run is a chained Slurm job.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../" && pwd -P)
echo ${REPO_ROOT}
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base_torchft_global.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
MIN_REPLICAS=${MIN_REPLICAS:-${NGPU}}
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"localhost"}
LIGHTHOUSE_PORT_BASE=${LIGHTHOUSE_PORT_BASE:-29510}
RDZV_PORT_BASE=${RDZV_PORT_BASE:-30000}
# PORT_OFFSET allows running multiple experiments simultaneously without port conflicts.
# Set PORT_OFFSET=100 for a second experiment, PORT_OFFSET=200 for a third, etc.
PORT_OFFSET=${PORT_OFFSET:-100}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-"http://taranaki.cl.cam.ac.uk:9000"}
if [[ -z "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="${REPO_ROOT}"
else
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi
RUN_PREFIX=${RUN_PREFIX:-"icml2026-exp1-global-sim-scores"}
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Chain definition (rank, lr, resume path per run).
# declare -a CHAIN_RANKS=(8 16 32 64 128 256)
# declare -a CHAIN_LRS=(0.016 0.008 0.008 0.008 0.016 0.008)
# declare -a CHAIN_RESUME_RUNS=(
# 	"icml2026-warmed-up-ddp-d99a5257-r8-lr0p016-rottrue-20251128-163213-idx0"
# 	"icml2026-warmed-up-ddp-4dd9e45e-r16-lr0p008-rottrue-20251128-163510-idx0"
# 	"icml2026-warmed-up-ddp-54f19506-r32-lr0p008-rottrue-20251128-163601-idx0"
# 	"icml2026-warmed-up-ddp-ebf60169-r64-lr0p008-rottrue-20251201-155531-idx0"
# 	"icml2026-warmed-up-ddp-a35e91e2-r128-lr0p016-rottrue-20251128-163708-idx0"
# 	"icml2026-warmed-up-ddp-75a6984d-r256-lr0p008-rottrue-20251128-171759-idx0"
# )
declare -a CHAIN_RANKS=(8)
declare -a CHAIN_LRS=(0.016)
declare -a CHAIN_RESUME_RUNS=(
	"icml2026-warmed-up-ddp-d99a5257-r8-lr0p016-rottrue-20251128-163213-idx0"
)
CHAIN_LENGTH=${#CHAIN_RANKS[@]}

if (( CHAIN_LENGTH == 0 )); then
	echo "No GaLoreGlobal runs specified." >&2
	exit 1
fi
if (( ${#CHAIN_LRS[@]} != CHAIN_LENGTH || ${#CHAIN_RESUME_RUNS[@]} != CHAIN_LENGTH )); then
	echo "Chain arrays are mismatched in length." >&2
	exit 1
fi

CHAIN_COUNT=${CHAIN_COUNT:-1}
if (( CHAIN_COUNT < 1 )); then
	echo "CHAIN_COUNT must be >= 1 (got ${CHAIN_COUNT})." >&2
	exit 1
fi
if (( CHAIN_COUNT > CHAIN_LENGTH )); then
	echo "CHAIN_COUNT (${CHAIN_COUNT}) exceeds number of runs (${CHAIN_LENGTH}); capping to ${CHAIN_LENGTH}." >&2
	CHAIN_COUNT=${CHAIN_LENGTH}
fi

declare -a CHAIN_SIZES=()
base_chain_size=$((CHAIN_LENGTH / CHAIN_COUNT))
chain_remainder=$((CHAIN_LENGTH % CHAIN_COUNT))
for ((chain_idx=0; chain_idx<CHAIN_COUNT; chain_idx++)); do
	size=${base_chain_size}
	if (( chain_idx < chain_remainder )); then
		((size++))
	fi
	CHAIN_SIZES[$chain_idx]=${size}
done

declare -a CHAIN_ASSIGNMENTS=()
offset=0
for ((chain_idx=0; chain_idx<CHAIN_COUNT; chain_idx++)); do
	size=${CHAIN_SIZES[$chain_idx]}
	for ((k=0; k<size; k++)); do
		CHAIN_ASSIGNMENTS[$((offset + k))]=${chain_idx}
	done
	((offset+=size))
done

RESUME_STEP=${RESUME_STEP:-2048}
TRAIN_STEPS=${TRAIN_STEPS:-6144}
ROTATE_MOMENTS=${ROTATE_MOMENTS:-true}
LOW_RANK_SERVER_UPDATE=${LOW_RANK_SERVER_UPDATE:-true}
LOW_RANK_PROJECTOR_ERROR_FEEDBACK=${LOW_RANK_PROJECTOR_ERROR_FEEDBACK:-false}
LOW_RANK_PROJECTOR_MOMENTUM_WEIGHT=${LOW_RANK_PROJECTOR_MOMENTUM_WEIGHT:-""}
GALORE_USE_ERROR_FEEDBACK=${GALORE_USE_ERROR_FEEDBACK:-false}
GALORE_REGEX_PATTERN=${GALORE_REGEX_PATTERN:-"attention\\.w[qkv]|attention\\.wo|feed_forward\\.w[12]"}
FULL_RANK_THRESHOLD=${FULL_RANK_THRESHOLD:-256}
GENERATED_CONFIG_DIR=${GENERATED_CONFIG_DIR:-"${SCRIPT_DIR}/generated_configs"}
ARGS_DIR=${ARGS_DIR:-"${GENERATED_CONFIG_DIR}/args"}
LOG_ROOT=${LOG_ROOT:-"${SCRIPT_DIR}/logs"}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"${SCRIPT_DIR}/slurm_logs"}
mkdir -p "${GENERATED_CONFIG_DIR}" "${ARGS_DIR}" "${LOG_ROOT}" "${SLURM_LOG_DIR}"

TRAINING_ARGS=("$@")

normalize_bool() {
	local value="${1:-}"
	case "${value,,}" in
		true|1|yes|on) echo "true" ;;
		false|0|no|off|"") echo "false" ;;
		*) echo "${value,,}" ;;
	esac
}

run_python_config() {
	local base_cfg=$1
	local output_cfg=$2
	local rank=$3
	local lr=$4
	local resume_run=$5
	local rotate_flag=$(normalize_bool "${ROTATE_MOMENTS}")
	local low_rank_flag=$(normalize_bool "${LOW_RANK_SERVER_UPDATE}")
	local error_feedback_flag=$(normalize_bool "${LOW_RANK_PROJECTOR_ERROR_FEEDBACK}")
	local galore_error_feedback_flag=$(normalize_bool "${GALORE_USE_ERROR_FEEDBACK}")
	local momentum_weight="${LOW_RANK_PROJECTOR_MOMENTUM_WEIGHT}"
	BASE_CONFIG_PATH="${base_cfg}" \
	OUTPUT_CONFIG_PATH="${output_cfg}" \
	TARGET_RANK="${rank}" \
	TARGET_LR="${lr}" \
	TRAIN_STEPS="${TRAIN_STEPS}" \
	RESUME_RUN="${resume_run}" \
	RESUME_STEP="${RESUME_STEP}" \
	SWEEP_REGEX_PATTERN="${GALORE_REGEX_PATTERN}" \
	ROTATE_MOMENTS="${rotate_flag}" \
	DESLOC_LOW_RANK_UPDATE="${low_rank_flag}" \
	DESLOC_PROJECTOR_ERROR_FEEDBACK="${error_feedback_flag}" \
	DESLOC_PROJECTOR_MOMENTUM_WEIGHT="${momentum_weight}" \
	GALORE_USE_ERROR_FEEDBACK="${galore_error_feedback_flag}" \
	FULL_RANK_THRESHOLD="${FULL_RANK_THRESHOLD}" \
	uv run --no-sync python3 <<'PY'
import os
from pathlib import Path

try:
		import tomllib
except ModuleNotFoundError:  # pragma: no cover
		import tomli as tomllib  # type: ignore
import tomli_w

base = Path(os.environ["BASE_CONFIG_PATH"])
output = Path(os.environ["OUTPUT_CONFIG_PATH"])
pattern = os.environ["SWEEP_REGEX_PATTERN"]
rank = int(os.environ["TARGET_RANK"])
lr = float(os.environ["TARGET_LR"])

train_steps = int(os.environ["TRAIN_STEPS"])
resume_run = os.environ["RESUME_RUN"]
resume_step = os.environ["RESUME_STEP"]
rotate_flag = os.environ["ROTATE_MOMENTS"].strip().lower() in {"true", "1", "yes", "on"}
low_rank_update = os.environ["DESLOC_LOW_RANK_UPDATE"].strip().lower() in {"true", "1", "yes", "on"}
error_feedback = os.environ["DESLOC_PROJECTOR_ERROR_FEEDBACK"].strip().lower() in {"true", "1", "yes", "on"}
galore_error_feedback = os.environ["GALORE_USE_ERROR_FEEDBACK"].strip().lower() in {"true", "1", "yes", "on"}
full_rank_threshold = int(os.environ["FULL_RANK_THRESHOLD"])
momentum_weight_raw = os.environ.get("DESLOC_PROJECTOR_MOMENTUM_WEIGHT", "").strip()

data = tomllib.loads(base.read_text(encoding="utf-8"))
optimizer = data.setdefault("optimizer", {})
optimizer["lr"] = lr
desloc_cfg = optimizer.setdefault("desloc", {})
if not isinstance(desloc_cfg, dict):  # pragma: no cover - defensive guard
	desloc_cfg = {}
	optimizer["desloc"] = desloc_cfg
desloc_cfg.setdefault("enabled", True)
is_full_rank = rank >= full_rank_threshold
desloc_cfg["low_rank_server_update"] = low_rank_update and not is_full_rank
desloc_cfg["low_rank_projector_error_feedback"] = error_feedback and not is_full_rank
if not is_full_rank:
	if momentum_weight_raw:
		weight_val = float(momentum_weight_raw)
		if weight_val < 0.0 or weight_val > 1.0:
			raise ValueError("DESLOC_PROJECTOR_MOMENTUM_WEIGHT must be within [0,1]")
		desloc_cfg["low_rank_projector_momentum_weight"] = weight_val
	else:
		desloc_cfg.pop("low_rank_projector_momentum_weight", None)
else:
	desloc_cfg.pop("low_rank_projector_momentum_weight", None)
param_sync = desloc_cfg.get("param_sync_every")
if param_sync is None:
	param_sync = optimizer.get("galore_update_proj_gap", 32)
desloc_cfg["param_sync_every"] = param_sync
if is_full_rank:
	desloc_cfg["optimizer_sync_every"] = [param_sync, param_sync]
	optimizer["name"] = "AdamW"
	optimizer.pop("galore_param_regexes", None)
	for key in list(optimizer.keys()):
		if key.startswith("galore_"):
			optimizer.pop(key, None)
else:
	desloc_cfg["optimizer_sync_every"] = [param_sync, param_sync, param_sync]
	optimizer["name"] = "GaLoreGlobal"
	optimizer["galore_rotate_moments_on_refresh"] = rotate_flag
	optimizer["galore_use_error_feedback"] = galore_error_feedback
	optimizer["galore_update_proj_gap"] = param_sync
	regex_entries = optimizer.get("galore_param_regexes")
	normalized = []
	if isinstance(regex_entries, list):
		for entry in regex_entries:
			if isinstance(entry, dict):
				normalized.append(dict(entry))
	elif isinstance(regex_entries, dict):
		normalized = [dict(regex_entries)]
	if not any(entry.get("param_str_match") == pattern for entry in normalized):
		normalized.append({"param_str_match": pattern, "rank": rank})
	else:
		for entry in normalized:
			if entry.get("param_str_match") == pattern:
				entry["rank"] = rank
				break
	optimizer["galore_param_regexes"] = normalized

data.setdefault("training", {})["steps"] = train_steps
s3_cfg = data.setdefault("s3_checkpoint", {})
if resume_run:
		s3_cfg["resume_from_run_step"] = f"{resume_run}/step-{resume_step}"
else:
		s3_cfg["resume_from_run_step"] = ""

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(tomli_w.dumps(data), encoding="utf-8")
PY
}

write_args_file() {
	local dest=$1
	shift || true
	local -a payload=("${@}")
	if ((${#payload[@]})); then
		{
			for arg in "${payload[@]}"; do
				printf '%s\0' "${arg}"
			done
		} > "${dest}"
	else
		: > "${dest}"
	fi
}

# Slurm submission defaults.
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-8}
SBATCH_GPUS_PER_TASK=${SBATCH_GPUS_PER_TASK:-4}
SBATCH_TIME=${SBATCH_TIME:-11:59:00}
SBATCH_NODE=${SBATCH_NODE:-"mauao"}
SBATCH_PARTITION=${SBATCH_PARTITION:-}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-}
SBATCH_QOS=${SBATCH_QOS:-}
SBATCH_ADDITIONAL_ARGS=${SBATCH_ADDITIONAL_ARGS:-}

read -r -a SBATCH_EXTRA_ARRAY <<< "${SBATCH_ADDITIONAL_ARGS}"
if [[ -z "${SBATCH_ADDITIONAL_ARGS}" ]]; then
	SBATCH_EXTRA_ARRAY=()
fi

# Ensure SLURM_SUBMIT_DIR is always defined to placate set -u when this script
# is run outside of a Slurm allocation.
SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-}"

COMMON_SBATCH_ARGS=(
	-c "${SBATCH_CPUS_PER_TASK}"
	--gres="gpu:${SBATCH_GPUS_PER_TASK}"
	--tasks-per-node=1
	--time="${SBATCH_TIME}"
)
[[ -n "${SBATCH_NODE}" ]] && COMMON_SBATCH_ARGS+=(-w "${SBATCH_NODE}")
[[ -n "${SBATCH_PARTITION}" ]] && COMMON_SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
[[ -n "${SBATCH_ACCOUNT}" ]] && COMMON_SBATCH_ARGS+=(--account="${SBATCH_ACCOUNT}")
[[ -n "${SBATCH_QOS}" ]] && COMMON_SBATCH_ARGS+=(--qos="${SBATCH_QOS}")
COMMON_SBATCH_ARGS+=("${SBATCH_EXTRA_ARRAY[@]}")

# Show port configuration for debugging concurrent runs
effective_lighthouse_port=$((LIGHTHOUSE_PORT_BASE + PORT_OFFSET))
effective_rdzv_base=$((RDZV_PORT_BASE + PORT_OFFSET))
echo "Port configuration (PORT_OFFSET=${PORT_OFFSET}):"
echo "  Lighthouse port: ${effective_lighthouse_port}"
echo "  Rendezvous ports: ${effective_rdzv_base} - $((effective_rdzv_base + NGPU - 1))"
echo ""

echo "Planned GaLoreGlobal chains (${CHAIN_LENGTH} runs split across ${CHAIN_COUNT} chain(s)):"
offset=0
for ((chain_idx=0; chain_idx<CHAIN_COUNT; chain_idx++)); do
	size=${CHAIN_SIZES[chain_idx]}
	chain_label=$((chain_idx + 1))
	if (( size == 0 )); then
		echo "  Chain ${chain_label}: (no runs)"
		continue
	fi
	echo "  Chain ${chain_label} (${size} runs):"
	for ((k=0; k<size; k++)); do
		run_idx=$((offset + k))
		printf '    [%d] rank=%-4d lr=%-6s resume=%s\n' "${run_idx}" "${CHAIN_RANKS[$run_idx]}" "${CHAIN_LRS[$run_idx]}" "${CHAIN_RESUME_RUNS[$run_idx]}"
	done
	((offset+=size))
done
echo ""

declare -a PREVIOUS_JOB_IDS=()
declare -a SUBMITTED_JOB_IDS=()
declare -a SUBMITTED_JOB_IDS_PER_CHAIN=()

for ((idx=0; idx<CHAIN_LENGTH; idx++)); do
	chain_idx=${CHAIN_ASSIGNMENTS[$idx]}
	chain_label=$((chain_idx + 1))
	rank=${CHAIN_RANKS[$idx]}
	lr=${CHAIN_LRS[$idx]}
	resume_run=${CHAIN_RESUME_RUNS[$idx]}
	lr_tag=${lr/./p}
	run_suffix=$(printf "chain%02d-idx%02d-r%d-lr%s" "${chain_label}" $((idx + 1)) "${rank}" "${lr_tag}")
	run_uuid="${RUN_PREFIX}-${TIMESTAMP}-${run_suffix}"
	config_path="${GENERATED_CONFIG_DIR}/${run_uuid}.toml"
	args_file="${ARGS_DIR}/${run_uuid}.args"
	log_dir="${LOG_ROOT}/${run_uuid}"
	mkdir -p "${log_dir}"

	run_python_config "${CONFIG_FILE}" "${config_path}" "${rank}" "${lr}" "${resume_run}"
	write_args_file "${args_file}" "${TRAINING_ARGS[@]}"

	job_name="galore_global_chain${chain_label}_r${rank}_lr${lr_tag}"
	sbatch_args=("${COMMON_SBATCH_ARGS[@]}" --job-name="${job_name}" --output="${SLURM_LOG_DIR}/slurm-${run_uuid}-%j.out")
	previous_job_id="${PREVIOUS_JOB_IDS[$chain_idx]:-}"
	if [[ -n "${previous_job_id}" ]]; then
		sbatch_args+=(--dependency="afterok:${previous_job_id}")
	fi

	# Export run-specific values directly to the sbatch job so we can use a
	# single-quoted heredoc without triggering premature shell expansion.
	# Compute actual ports using base + offset to avoid conflicts with other experiments.
	lighthouse_port=$((LIGHTHOUSE_PORT_BASE + PORT_OFFSET))
	rdzv_port_base=$((RDZV_PORT_BASE + PORT_OFFSET))
	sbatch_export_arg="ALL"
	for kv in \
		"RUN_UUID=${run_uuid}" \
		"CONFIG_PATH=${config_path}" \
		"LOG_DIR=${log_dir}" \
		"ARGS_FILE=${args_file}" \
		"REPO_ROOT=${REPO_ROOT}" \
		"TRAIN_MODULE=${TRAIN_MODULE}" \
		"NGPU=${NGPU}" \
		"MIN_REPLICAS=${MIN_REPLICAS}" \
		"QUORUM_TICK_MS=${QUORUM_TICK_MS}" \
		"LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST}" \
		"LIGHTHOUSE_PORT=${lighthouse_port}" \
		"RDZV_PORT_BASE=${rdzv_port_base}" \
		"S3_ENDPOINT_URL=${S3_ENDPOINT_URL}" \
		"PYTHONPATH=${PYTHONPATH}"; do
		sbatch_export_arg+="${kv:+,${kv}}"
	done
	sbatch_args+=("--export=${sbatch_export_arg}")

	sbatch_output=$(sbatch "${sbatch_args[@]}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

	LOG_DIR="${LOG_DIR:-}"
	if [[ -z "${LOG_DIR}" ]]; then
		echo "LOG_DIR environment variable is not set" >&2
		exit 1
	fi

export REPO_ROOT
export S3_ENDPOINT_URL
export PYTHONPATH="${PYTHONPATH}"

cleanup() {
	pkill -P $$ || true
	pkill -f "torchft_lighthouse" || true
}
trap cleanup EXIT INT TERM

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	cd "${SLURM_SUBMIT_DIR}"
else
	cd "${REPO_ROOT}"
fi

find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

mkdir -p "${LOG_DIR}"
LIGHTHOUSE_LOG_FILE="${LOG_DIR}/lighthouse.log"

export RUN_UUID="${RUN_UUID}"
export WANDB_RUN_NAME="${RUN_UUID}"
export WANDB_PROJECT=${WANDB_PROJECT:-"galore-tune-lr"}
export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=1

TORCHFT_LIGHTHOUSE_URL="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE_URL}"

echo "[TorchFT Chain] RUN_UUID=${RUN_UUID}"
echo "Config: ${CONFIG_PATH}"
echo "Logs: ${LOG_DIR}"

mapfile -d '' TRAIN_ARGS < "${ARGS_FILE}" || true

uv run --no-sync torchft_lighthouse \
	--min_replicas ${MIN_REPLICAS} \
	--quorum_tick_ms ${QUORUM_TICK_MS} \
	--bind ${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT} \
	> "${LIGHTHOUSE_LOG_FILE}" 2>&1 &
LIGHTHOUSE_PID=$!
sleep 2
if ! kill -0 ${LIGHTHOUSE_PID} 2>/dev/null; then
	echo "Lighthouse failed to start, see ${LIGHTHOUSE_LOG_FILE}" >&2
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
		cd "${REPO_ROOT}"
		export CUDA_VISIBLE_DEVICES="${gpu_id}"
		export PYTORCH_ALLOC_CONF="expandable_segments:True"
		rdzv_port=$((RDZV_PORT_BASE + replica_id))
		uv run --no-sync torchrun \
			--nproc_per_node=1 \
			--rdzv_backend=c10d \
			--rdzv_endpoint="localhost:${rdzv_port}" \
			--role=rank \
			--tee=3 \
			-m "${TRAIN_MODULE}" \
			--job.config_file "${CONFIG_PATH}" \
			--fault_tolerance.replica_id "${replica_id}" \
			--fault_tolerance.group_size "${NGPU}" \
			--fault_tolerance.min_replica_size "${MIN_REPLICAS}" \
			"${TRAIN_ARGS[@]}"
	) > "${log_file}" 2>&1 &
	REPLICA_PIDS[$replica_id]=$!
	sleep 1
done

set +e
replica_status=0
for ((replica_id=0; replica_id<NGPU; replica_id++)); do
	pid=${REPLICA_PIDS[$replica_id]}
	if ! wait "${pid}"; then
		replica_status=$?
		echo "Replica ${replica_id} exited with status ${replica_status}."
	else
		echo "Replica ${replica_id} completed successfully."
	fi
done
set -e

if kill -0 ${LIGHTHOUSE_PID} 2>/dev/null; then
	kill ${LIGHTHOUSE_PID}
fi

exit ${replica_status}
EOF
)

	if [[ "${sbatch_output}" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
		job_id=${BASH_REMATCH[1]}
		PREVIOUS_JOB_IDS[$chain_idx]=${job_id}
		SUBMITTED_JOB_IDS+=(${job_id})
		if [[ -n "${SUBMITTED_JOB_IDS_PER_CHAIN[$chain_idx]:-}" ]]; then
			SUBMITTED_JOB_IDS_PER_CHAIN[$chain_idx]+=" ${job_id}"
		else
			SUBMITTED_JOB_IDS_PER_CHAIN[$chain_idx]="${job_id}"
		fi
		echo "Submitted job ${job_id} (${job_name})"
	else
		echo "Failed to submit job for run ${run_uuid}" >&2
		exit 1
	fi
done

echo ""
echo "Submitted jobs by chain:"
for ((chain_idx=0; chain_idx<CHAIN_COUNT; chain_idx++)); do
	chain_label=$((chain_idx + 1))
	jobs="${SUBMITTED_JOB_IDS_PER_CHAIN[$chain_idx]:-}"
	if [[ -n "${jobs}" ]]; then
		echo "  Chain ${chain_label}: ${jobs}"
	else
		echo "  Chain ${chain_label}: (no jobs submitted)"
	fi
done
echo "Submission order across all chains: ${SUBMITTED_JOB_IDS[*]}"
echo "Use 'squeue -u ${USER}' to monitor progress."
