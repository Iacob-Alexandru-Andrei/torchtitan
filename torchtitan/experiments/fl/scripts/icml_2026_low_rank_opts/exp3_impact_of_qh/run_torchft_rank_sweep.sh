#!/usr/bin/env bash
# Launch a GaLore TorchFT sweep where each run is a chained Slurm job.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../" && pwd -P)
echo ${REPO_ROOT}
CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base_torchft.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
MIN_REPLICAS=${MIN_REPLICAS:-${NGPU}}
QUORUM_TICK_MS=${QUORUM_TICK_MS:-100}
LIGHTHOUSE_HOST=${LIGHTHOUSE_HOST:-"localhost"}
LIGHTHOUSE_PORT=${LIGHTHOUSE_PORT:-29510}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-"http://taranaki.cl.cam.ac.uk:9000"}
if [[ -z "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="${REPO_ROOT}"
else
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi
RUN_PREFIX=${RUN_PREFIX:-"icml2026-exp3-pg-test"}
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Chain definition (rank, lr, resume path per run).
# declare -a CHAIN_RANKS=(8 16 32 64 128 256)
# declare -a CHAIN_LRS=(0.016 0.008 0.008 0.008 0.016 0.008)
# declare -a CHAIN_RESUME_RUNS=( # THESE ARE THE DDP WARMED UP RUNS WITHOUT EF!
# ``	"icml2026-warmed-up-ddp-d99a5257-r8-lr0p016-rottrue-20251128-163213-idx0"
# 	"icml2026-warmed-up-ddp-4dd9e45e-r16-lr0p008-rottrue-20251128-163510-idx0"
# 	"icml2026-warmed-up-ddp-54f19506-r32-lr0p008-rottrue-20251128-163601-idx0"
# 	"icml2026-warmed-up-ddp-ebf60169-r64-lr0p008-rottrue-20251201-155531-idx0"
# 	"icml2026-warmed-up-ddp-a35e91e2-r128-lr0p016-rottrue-20251128-163708-idx0"
# 	"icml2026-warmed-up-ddp-75a6984d-r256-lr0p008-rottrue-20251128-171759-idx0"
# )
# declare -a CHAIN_RANKS=(8 16 32 64 128 256)
# declare -a CHAIN_LRS=(0.016 0.008 0.008 0.008 0.016 0.008)
# declare -a CHAIN_RESUME_RUNS=( # THESE ARE THE EF WARMED UP RUNS!
# 	"icml2026-galore-ef-d99a5257-r8-lr0p016-rottrue-20251217-102843-idx0"
# 	"icml2026-warmup-ef-4dd9e45e-r16-lr0p008-rottrue-20251217-144111-idx0"
# 	"icml2026-warmup-ef-54f19506-r32-lr0p008-rottrue-20251217-150027-idx0"
# 	"icml2026-warmup-ef-ebf60169-r64-lr0p008-rottrue-20251217-151755-idx0"
# 	"icml2026-warmup-ef-a35e91e2-r128-lr0p016-rottrue-20251217-151833-idx0"
# 	"icml2026-warmed-up-ddp-75a6984d-r256-lr0p008-rottrue-20251128-171759-idx0"
# )
declare -a CHAIN_RANKS=(8)
declare -a CHAIN_LRS=(0.016)
declare -a CHAIN_RESUME_RUNS=( # THESE ARE THE EF WARMED UP RUNS!
	"icml2026-galore-ef-d99a5257-r8-lr0p016-rottrue-20251217-102843-idx0"
)

CHAIN_LENGTH=${#CHAIN_RANKS[@]}

if (( CHAIN_LENGTH == 0 )); then
	echo "No GaLore runs specified." >&2
	exit 1
fi
if (( ${#CHAIN_LRS[@]} != CHAIN_LENGTH || ${#CHAIN_RESUME_RUNS[@]} != CHAIN_LENGTH )); then
	echo "Chain arrays are mismatched in length." >&2
	exit 1
fi

RESUME_STEP=${RESUME_STEP:-2048}
TRAIN_STEPS=${TRAIN_STEPS:-6144}
ROTATE_MOMENTS=${ROTATE_MOMENTS:-true}
USE_ERROR_FEEDBACK=${USE_ERROR_FEEDBACK:-true}
GALORE_REGEX_PATTERN=${GALORE_REGEX_PATTERN:-"attention\\.w[qkv]|attention\\.wo|feed_forward\\.w[12]"}
FULL_RANK_THRESHOLD=${FULL_RANK_THRESHOLD:-256}
GALORE_VS=${GALORE_VS:-"0.5"}
GALORE_QHM_OUTSIDE=${GALORE_QHM_OUTSIDE:-true}
# Hyperparameter switch overrides (comma-separated lists for arrays)
HP_SWITCH_ENABLED=${HP_SWITCH_ENABLED:-true}
HP_SWITCH_STEPS=${HP_SWITCH_STEPS:-2048}
HP_SWITCH_NEW_VS=${HP_SWITCH_NEW_VS:-"0.95,"}
HP_SWITCH_NEW_BETAS=${HP_SWITCH_NEW_BETAS:-"0.999,0.999"}
HP_SWITCH_RESET_MOMENTA=${HP_SWITCH_RESET_MOMENTA:-"exp_avg,exp_avg_sq"}

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
	local error_feedback_flag=$(normalize_bool "${USE_ERROR_FEEDBACK}")
	BASE_CONFIG_PATH="${base_cfg}" \
	OUTPUT_CONFIG_PATH="${output_cfg}" \
	TARGET_RANK="${rank}" \
	TARGET_LR="${lr}" \
	TRAIN_STEPS="${TRAIN_STEPS}" \
	RESUME_RUN="${resume_run}" \
	RESUME_STEP="${RESUME_STEP}" \
	SWEEP_REGEX_PATTERN="${GALORE_REGEX_PATTERN}" \
	ROTATE_MOMENTS="${rotate_flag}" \
	USE_ERROR_FEEDBACK="${error_feedback_flag}" \
	FULL_RANK_THRESHOLD="${FULL_RANK_THRESHOLD}" \
	GALORE_VS="${GALORE_VS}" \
	GALORE_QHM_OUTSIDE="${GALORE_QHM_OUTSIDE}" \
	HP_SWITCH_ENABLED="${HP_SWITCH_ENABLED}" \
	HP_SWITCH_STEPS="${HP_SWITCH_STEPS}" \
	HP_SWITCH_NEW_VS="${HP_SWITCH_NEW_VS}" \
	HP_SWITCH_NEW_BETAS="${HP_SWITCH_NEW_BETAS}" \
	HP_SWITCH_RESET_MOMENTA="${HP_SWITCH_RESET_MOMENTA}" \
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
use_error_feedback = os.environ["USE_ERROR_FEEDBACK"].strip().lower() in {"true", "1", "yes", "on"}
full_rank_threshold = int(os.environ["FULL_RANK_THRESHOLD"])
# Respect the base config unless the env var is explicitly provided.
galore_qhm_env = os.environ.get("GALORE_QHM_OUTSIDE")
if galore_qhm_env is None or galore_qhm_env.strip() == "":
	# keep existing optimizer value if present; otherwise default to False
	# (don't unconditionally override config with a false default)
	# we'll set this into the optimizer below only if missing
	galore_qhm_outside = None
else:
	galore_qhm_outside = galore_qhm_env.strip().lower() in {
		"true",
		"1",
		"yes",
		"on",
	}

# Hyperparameter switch env overrides (empty = leave base file value)
hp_enabled_env = os.environ.get("HP_SWITCH_ENABLED", "").strip()
hp_steps_env = os.environ.get("HP_SWITCH_STEPS", "").strip()
hp_new_vs_env = os.environ.get("HP_SWITCH_NEW_VS", "").strip()
hp_new_betas_env = os.environ.get("HP_SWITCH_NEW_BETAS", "").strip()
hp_reset_momenta_env = os.environ.get("HP_SWITCH_RESET_MOMENTA", "").strip()

data = tomllib.loads(base.read_text(encoding="utf-8"))
optimizer = data.setdefault("optimizer", {})
optimizer["lr"] = lr
is_full_rank = rank >= full_rank_threshold
if is_full_rank:
		# Use AdamW for full-rank mode (no low-rank projections)
		optimizer["name"] = "AdamW"
		optimizer.pop("galore_param_regexes", None)
		for key in list(optimizer.keys()):
				if key.startswith("galore_"):
						optimizer.pop(key, None)
else:
		# Use GaLore for low-rank mode
		optimizer["name"] = "GaLore"
		optimizer["galore_rotate_moments_on_refresh"] = rotate_flag
		optimizer["galore_use_error_feedback"] = use_error_feedback
		param_sync = optimizer.get("desloc", {}).get("param_sync_every")
		if param_sync is None:
				param_sync = optimizer.get("galore_update_proj_gap", 32)
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

# determine galore vs list from env (GALORE_VS) or fallback to base config/defaults
galore_vs_env = os.environ.get("GALORE_VS", "").strip()
if galore_vs_env:
	galore_vs = [float(x) for x in galore_vs_env.split(",") if x.strip()]
else:
	galore_vs = optimizer.get("galore_vs", [0.0])

# sanitize and assign into optimizer config (primary key expected: galore_vs)
optimizer["galore_vs"] = galore_vs
# Only set/override galore_qhm_outside_projection when the env var was
# explicitly provided; otherwise preserve any value present in the base
# config (or default to False if not present).
if galore_qhm_outside is None:
	optimizer.setdefault("galore_qhm_outside_projection", False)
else:
	optimizer["galore_qhm_outside_projection"] = bool(galore_qhm_outside)

# Apply hyperparameter-switch overrides into the config when provided via env
fl_metrics = data.setdefault("fl_metrics", {})
hp = fl_metrics.setdefault("hyperparameter_switch", {})
if hp_enabled_env:
	hp["enabled"] = hp_enabled_env.lower() in {"true", "1", "yes", "on"}
if hp_steps_env:
	hp["steps"] = [int(x) for x in hp_steps_env.split(",") if x.strip()]
if hp_new_vs_env:
	hp["new_vs"] = [float(x) for x in hp_new_vs_env.split(",") if x.strip()]
if hp_new_betas_env:
	hp["new_betas"] = [float(x) for x in hp_new_betas_env.split(",") if x.strip()]
if hp_reset_momenta_env:
	hp["reset_momenta"] = [s.strip() for s in hp_reset_momenta_env.split(",") if s.strip()]

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

echo "Planned GaLore chain (${CHAIN_LENGTH} runs):"
for ((idx=0; idx<CHAIN_LENGTH; idx++)); do
	printf '  [%d] rank=%-4d lr=%-6s resume=%s\n' "${idx}" "${CHAIN_RANKS[$idx]}" "${CHAIN_LRS[$idx]}" "${CHAIN_RESUME_RUNS[$idx]}"
done
echo ""

previous_job_id=""
declare -a SUBMITTED_JOB_IDS=()

for ((idx=0; idx<CHAIN_LENGTH; idx++)); do
	rank=${CHAIN_RANKS[$idx]}
	lr=${CHAIN_LRS[$idx]}
	resume_run=${CHAIN_RESUME_RUNS[$idx]}
	lr_tag=${lr/./p}
	run_suffix=$(printf "idx%02d-r%d-lr%s" $((idx + 1)) "${rank}" "${lr_tag}")
	run_uuid="${RUN_PREFIX}-${TIMESTAMP}-${run_suffix}"
	config_path="${GENERATED_CONFIG_DIR}/${run_uuid}.toml"
	args_file="${ARGS_DIR}/${run_uuid}.args"
	log_dir="${LOG_ROOT}/${run_uuid}"
	mkdir -p "${log_dir}"

	run_python_config "${CONFIG_FILE}" "${config_path}" "${rank}" "${lr}" "${resume_run}"
	write_args_file "${args_file}" "${TRAINING_ARGS[@]}"

	job_name="galore_chain_r${rank}_lr${lr_tag}"
	sbatch_args=("${COMMON_SBATCH_ARGS[@]}" --job-name="${job_name}" --output="${SLURM_LOG_DIR}/slurm-${run_uuid}-%j.out")
	if [[ -n "${previous_job_id}" ]]; then
		sbatch_args+=(--dependency="afterok:${previous_job_id}")
	fi

	# Export run-specific values directly to the sbatch job so we can use a
	# single-quoted heredoc without triggering premature shell expansion.
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
		"LIGHTHOUSE_PORT=${LIGHTHOUSE_PORT}" \
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

# find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

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
		rdzv_port=$((30000 + replica_id))
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
		previous_job_id=${job_id}
		SUBMITTED_JOB_IDS+=(${job_id})
		echo "Submitted job ${job_id} (${job_name})"
	else
		echo "Failed to submit job for run ${run_uuid}" >&2
		exit 1
	fi
done

echo ""
echo "Submitted jobs in chain order: ${SUBMITTED_JOB_IDS[*]}"
echo "Use 'squeue -u ${USER}' to monitor progress."
