#!/usr/bin/env bash
# Launch a GaLoreGlobal TorchFT sweep locally (no Slurm required).

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
LIGHTHOUSE_PORT_BASE=${LIGHTHOUSE_PORT_BASE:-33710}
RDZV_PORT_BASE=${RDZV_PORT_BASE:-31000}
# PORT_OFFSET allows running multiple experiments simultaneously without port conflicts.
# Set PORT_OFFSET=100 for a second experiment, PORT_OFFSET=200 for a third, etc.
PORT_OFFSET=${PORT_OFFSET:-100}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-"http://taranaki.cl.cam.ac.uk:9000"}
if [[ -z "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="${REPO_ROOT}"
else
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi
RUN_PREFIX=${RUN_PREFIX:-"icml2026-720M-global-qhm"}
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# Chain definition (rank, lr, resume path per run).
declare -a CHAIN_RANKS=(256)
declare -a CHAIN_LRS=(0.008)
declare -a CHAIN_RESUME_RUNS=( # THESE ARE THE EF WARMED UP RUNS!
	"TODO"
)
# declare -a CHAIN_OMEGAS=("0.99," "0.97," "0.97," "0.97," "0.97,")
declare -a CHAIN_OMEGAS=("0.97,")
# Optional per-run lr-scheduler switch scale values (will be written to
declare -a CHAIN_SWITCH_SCALES=("0.5")

CHAIN_LENGTH=${#CHAIN_RANKS[@]}

if (( CHAIN_LENGTH == 0 )); then
	echo "No GaLoreGlobal runs specified." >&2
	exit 1
fi
if (( ${#CHAIN_LRS[@]} != CHAIN_LENGTH || ${#CHAIN_RESUME_RUNS[@]} != CHAIN_LENGTH || ${#CHAIN_OMEGAS[@]} != CHAIN_LENGTH || ${#CHAIN_SWITCH_SCALES[@]} != CHAIN_LENGTH )); then
	echo "Chain arrays are mismatched in length." >&2
	echo "Ensure CHAIN_RANKS, CHAIN_LRS, CHAIN_RESUME_RUNS, CHAIN_OMEGAS, and CHAIN_SWITCH_SCALES all have the same length." >&2
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
TRAIN_STEPS=${TRAIN_STEPS:-12288}
ROTATE_MOMENTS=${ROTATE_MOMENTS:-true}
LOW_RANK_SERVER_UPDATE=${LOW_RANK_SERVER_UPDATE:-true}
LOW_RANK_PROJECTOR_ERROR_FEEDBACK=${LOW_RANK_PROJECTOR_ERROR_FEEDBACK:-false}
GALORE_USE_ERROR_FEEDBACK=${GALORE_USE_ERROR_FEEDBACK:-true}
LOW_RANK_PROJECTOR_SOURCE=${LOW_RANK_PROJECTOR_SOURCE:-"pseudo_grad"}
GALORE_REGEX_PATTERN=${GALORE_REGEX_PATTERN:-"attention\\.w[qkv]|attention\\.wo|feed_forward\\.w[12]"}
FULL_RANK_THRESHOLD=${FULL_RANK_THRESHOLD:-2048}
GALORE_QHM_OUTSIDE=${GALORE_QHM_OUTSIDE:-true}
DRY_RUN=${DRY_RUN:-false}
# Hyperparameter switch overrides (comma-separated lists for arrays)
# HP_SWITCH_ENABLED may be: "true", "false", or "both".
# If set to "both", the script will submit the base run (HP switch off)
# and then submit the HP-switch-expanded grid (HP switch on) for each index.
HP_SWITCH_ENABLED=${HP_SWITCH_ENABLED:-true}
HP_SWITCH_STEPS=${HP_SWITCH_STEPS:-2048}
HP_SWITCH_NEW_VS=${HP_SWITCH_NEW_VS:-"0.98,"}
HP_SWITCH_NEW_BETAS=${HP_SWITCH_NEW_BETAS:-"0.999,0.999"}
HP_SWITCH_RESET_MOMENTA=${HP_SWITCH_RESET_MOMENTA:-"exp_avg,exp_avg_sq"}
GENERATED_CONFIG_DIR=${GENERATED_CONFIG_DIR:-"${SCRIPT_DIR}/generated_configs"}
LOG_ROOT=${LOG_ROOT:-"${SCRIPT_DIR}/logs"}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"${SCRIPT_DIR}/slurm_logs"}
mkdir -p "${GENERATED_CONFIG_DIR}" "${LOG_ROOT}" "${SLURM_LOG_DIR}"

normalize_bool() {
	local value="${1:-}"
	case "${value,,}" in
		true|1|yes|on) echo "true" ;;
		false|0|no|off|"") echo "false" ;;
		*) echo "${value,,}" ;;
	esac
}

TRAINING_ARGS=("$@")

# Normalized hp-switch enabled flag; runs are only submitted when enabled
HP_SWITCH_ENABLED_FLAG=$(normalize_bool "${HP_SWITCH_ENABLED}")
DRY_RUN_FLAG=$(normalize_bool "${DRY_RUN}")

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
	local galore_qhm_outside_flag=$(normalize_bool "${GALORE_QHM_OUTSIDE}")
	# Prefer an explicit per-invocation override `DESLOC_PROJECTOR_SOURCE` if set;
	# otherwise fall back to the global `LOW_RANK_PROJECTOR_SOURCE` default.
	local low_rank_projector_source_val="${DESLOC_PROJECTOR_SOURCE:-${LOW_RANK_PROJECTOR_SOURCE}}"
	local hp_switch_enabled_flag=$(normalize_bool "${HP_SWITCH_ENABLED}")
	local hp_switch_steps_val="${HP_SWITCH_STEPS}"
	local hp_switch_new_vs_val="${HP_SWITCH_NEW_VS}"
	local hp_switch_new_betas_val="${HP_SWITCH_NEW_BETAS}"
	local hp_switch_reset_momenta_val="${HP_SWITCH_RESET_MOMENTA}"
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
	GALORE_USE_ERROR_FEEDBACK="${galore_error_feedback_flag}" \
	GALORE_QHM_OUTSIDE="${galore_qhm_outside_flag}" \
	DESLOC_PROJECTOR_SOURCE="${low_rank_projector_source_val}" \
	HP_SWITCH_ENABLED="${hp_switch_enabled_flag}" \
	HP_SWITCH_STEPS="${hp_switch_steps_val}" \
	HP_SWITCH_NEW_VS="${hp_switch_new_vs_val}" \
	HP_SWITCH_NEW_BETAS="${hp_switch_new_betas_val}" \
	HP_SWITCH_RESET_MOMENTA="${hp_switch_reset_momenta_val}" \
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
# Read optional DES-LOC projector source override. Accepts e.g. "pseudo_grad" or "full_rank_grad".
proj_source_env = os.environ.get("DESLOC_PROJECTOR_SOURCE", "").strip()
if proj_source_env == "":
	low_rank_projector_source = None
else:
	v = proj_source_env.lower()
	if v in {"full", "full_rank", "full_rank_grad"}:
		low_rank_projector_source = "full_rank_grad"
	elif v in {"pseudo", "pseudo_grad"}:
		low_rank_projector_source = "pseudo_grad"
	else:
		# accept literal passthrough for any other value
		low_rank_projector_source = v
# Respect the base config unless the env var is explicitly provided.
galore_qhm_env = os.environ.get("GALORE_QHM_OUTSIDE")
if galore_qhm_env is None or galore_qhm_env.strip() == "":
	galore_qhm_outside = None
else:
	galore_qhm_outside = galore_qhm_env.strip().lower() in {"true", "1", "yes", "on"}

# Hyperparameter switch env overrides (empty = leave base file value)
hp_enabled_env = os.environ.get("HP_SWITCH_ENABLED", "").strip()
hp_steps_env = os.environ.get("HP_SWITCH_STEPS", "").strip()
hp_new_vs_env = os.environ.get("HP_SWITCH_NEW_VS", "").strip()
hp_new_betas_env = os.environ.get("HP_SWITCH_NEW_BETAS", "").strip()
hp_reset_momenta_env = os.environ.get("HP_SWITCH_RESET_MOMENTA", "").strip()
lr_sched_switch_scale_env = os.environ.get("LR_SCHED_SWITCH_SCALE", "").strip()

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
param_sync = desloc_cfg.get("param_sync_every")
if param_sync is None:
	param_sync = optimizer.get("galore_update_proj_gap", 32)
desloc_cfg["param_sync_every"] = param_sync
if is_full_rank:
	desloc_cfg["optimizer_sync_every"] = [param_sync, param_sync]
	# Keep GaLoreGlobal even for full-rank runs but disable projections.
	optimizer["name"] = "GaLoreGlobal"
	optimizer["galore_rotate_moments_on_refresh"] = rotate_flag
	optimizer["galore_use_error_feedback"] = galore_error_feedback
	optimizer["galore_update_proj_gap"] = param_sync
	# An empty regex list prevents low-rank projector application at full rank.
	optimizer["galore_param_regexes"] = []
else:
	desloc_cfg["optimizer_sync_every"] = [param_sync, param_sync, param_sync]
	optimizer["name"] = "GaLoreGlobal"
	optimizer["galore_rotate_moments_on_refresh"] = rotate_flag
	optimizer["galore_use_error_feedback"] = galore_error_feedback
	optimizer["galore_update_proj_gap"] = param_sync

# Apply explicit projector source when requested (and only for low-rank mode).
if not is_full_rank and low_rank_projector_source is not None:
    desloc_cfg["low_rank_projector_source"] = low_rank_projector_source
	# Only set/override galore_qhm_outside_projection when the env var was
	# explicitly provided; otherwise preserve any value present in the base
	# config (or default to False if not present).
if galore_qhm_outside is None:
	optimizer.setdefault("galore_qhm_outside_projection", False)
else:
	optimizer["galore_qhm_outside_projection"] = galore_qhm_outside

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

# Apply optional lr-scheduler switch scale override when provided.
if lr_sched_switch_scale_env:
	try:
		switch_scale_val = float(lr_sched_switch_scale_env)
	except Exception:
		switch_scale_val = None
	if switch_scale_val is not None:
		data.setdefault("lr_scheduler", {})["switch_scale"] = switch_scale_val

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

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 )); then
	echo "NPROC_PER_NODE must be a positive integer (got ${NPROC_PER_NODE})." >&2
	exit 1
fi

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
		printf '    [%d] rank=%-4d lr=%-6s resume=%s omega=%s switch_scale=%s\n' "${run_idx}" "${CHAIN_RANKS[$run_idx]}" "${CHAIN_LRS[$run_idx]}" "${CHAIN_RESUME_RUNS[$run_idx]}" "${CHAIN_OMEGAS[$run_idx]}" "${CHAIN_SWITCH_SCALES[$run_idx]}"
	done
	((offset+=size))
done
echo ""

run_local_job() {
	local run_uuid=$1
	local config_path=$2
	local log_dir=$3
	local chain_label=$4
	local idx_label=$5
	local rank=$6
	local lr=$7
	local resume_run=$8

	cleanup() {
		pkill -P $$ || true
		pkill -f "torchft_lighthouse" || true
	}
	trap cleanup EXIT INT TERM

	cd "${REPO_ROOT}"

	find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true
	mkdir -p "${log_dir}"
	local lighthouse_port=$((LIGHTHOUSE_PORT_BASE + PORT_OFFSET))
	local rdzv_port_base=$((RDZV_PORT_BASE + PORT_OFFSET))
	local lighthouse_log_file="${log_dir}/lighthouse.log"

	export RUN_UUID="${run_uuid}"
	export WANDB_RUN_NAME="${RUN_UUID}"
	export WANDB_PROJECT=${WANDB_PROJECT:-"galore-tune-lr"}
	export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
	export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
	export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=1
	export REPO_ROOT
	export S3_ENDPOINT_URL
	export PYTHONPATH="${PYTHONPATH}"

	local lighthouse_url="http://${LIGHTHOUSE_HOST}:${lighthouse_port}"
	export TORCHFT_LIGHTHOUSE="${lighthouse_url}"

	echo "[LOCAL TorchFT Chain] RUN_UUID=${run_uuid} chain=${chain_label} idx=${idx_label} rank=${rank} lr=${lr} resume=${resume_run}"
	echo "Config: ${config_path}"
	echo "Logs: ${log_dir}"

	uv run --no-sync torchft_lighthouse \
		--min_replicas ${MIN_REPLICAS} \
		--quorum_tick_ms ${QUORUM_TICK_MS} \
		--bind ${LIGHTHOUSE_HOST}:${lighthouse_port} \
		> "${lighthouse_log_file}" 2>&1 &
	local lighthouse_pid=$!
	sleep 2
	if ! kill -0 ${lighthouse_pid} 2>/dev/null; then
		echo "Lighthouse failed to start, see ${lighthouse_log_file}" >&2
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
		log_file="${log_dir}/replica_${replica_id}.log"
		(
			set -euo pipefail
			export CUDA_VISIBLE_DEVICES="${gpu_id}"
			export PYTORCH_ALLOC_CONF="expandable_segments:True"
			rdzv_port=$((rdzv_port_base + replica_id))
			uv run --no-sync torchrun \
				--nproc_per_node=${NPROC_PER_NODE} \
				--rdzv_backend=c10d \
				--rdzv_endpoint="localhost:${rdzv_port}" \
				--role=rank \
				--tee=3 \
				-m "${TRAIN_MODULE}" \
				--job.config_file "${config_path}" \
				--fault_tolerance.replica_id "${replica_id}" \
				--fault_tolerance.group_size "${NGPU}" \
				--fault_tolerance.min_replica_size "${MIN_REPLICAS}" \
				"${TRAINING_ARGS[@]}"
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

	if kill -0 ${lighthouse_pid} 2>/dev/null; then
		kill ${lighthouse_pid}
	fi

	return ${replica_status}
}

total_runs=0
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
	log_dir="${LOG_ROOT}/${run_uuid}"
	mkdir -p "${log_dir}"

	if [[ "${HP_SWITCH_ENABLED_FLAG,,}" != "true" ]]; then
		echo "Skipping run ${run_uuid}: hyperparameter switch is disabled (set HP_SWITCH_ENABLED=true)." >&2
		continue
	fi

	if [[ "${DRY_RUN_FLAG,,}" == "true" ]]; then
		echo "[DRY RUN][LOCAL] ${run_uuid} | chain=${chain_label} idx=$((idx + 1)) rank=${rank} lr=${lr} resume=${resume_run} omega=${CHAIN_OMEGAS[$idx]:-} switch_scale=${CHAIN_SWITCH_SCALES[$idx]:-}" >&2
		((total_runs++))
		continue
	fi

	HP_SWITCH_ENABLED="true" \
	HP_SWITCH_NEW_VS="${CHAIN_OMEGAS[$idx]:-}" \
	LR_SCHED_SWITCH_SCALE="${CHAIN_SWITCH_SCALES[$idx]:-}" \
	run_python_config "${CONFIG_FILE}" "${config_path}" "${rank}" "${lr}" "${resume_run}"

	echo "[LOCAL] Launching ${run_uuid} (chain=${chain_label}, idx=$((idx + 1)), rank=${rank}, lr=${lr})"
	run_local_job "${run_uuid}" "${config_path}" "${log_dir}" "${chain_label}" "$((idx + 1))" "${rank}" "${lr}" "${resume_run}"
	((total_runs++))
done

if [[ "${DRY_RUN_FLAG,,}" == "true" ]]; then
	echo "Dry run complete: ${total_runs} run(s) planned (no execution)."
else
	echo "Completed ${total_runs} run(s) locally."
fi
