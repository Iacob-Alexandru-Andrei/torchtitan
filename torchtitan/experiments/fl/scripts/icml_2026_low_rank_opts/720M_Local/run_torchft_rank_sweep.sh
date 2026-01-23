#!/usr/bin/env bash
# Launch a GaLore TorchFT sweep locally (no Slurm required).

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
LIGHTHOUSE_PORT=${LIGHTHOUSE_PORT:-49610}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-"http://taranaki.cl.cam.ac.uk:9000"}
if [[ -z "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="${REPO_ROOT}"
else
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
fi
RUN_PREFIX=${RUN_PREFIX:-"icml2026-720M-local-qhm"}
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

declare -a CHAIN_RANKS=(256 2048)
declare -a CHAIN_LRS=(0.008 0.008)
declare -a CHAIN_RESUME_RUNS=( # THESE ARE THE EF WARMED UP RUNS!
	"TODO"
)

# Optional per-run hyperparameter-switch omega values (will be written to
# HP_SWITCH_NEW_VS for the per-run config). Provide one entry per run.
declare -a CHAIN_OMEGAS=("0.94," "0.94,")
# Optional per-run lr-scheduler switch scale values (will be written to
# lr_scheduler.switch_scale in the generated config). Provide one entry per run.
declare -a CHAIN_SWITCH_SCALES=("2.0" "1.0")

CHAIN_LENGTH=${#CHAIN_RANKS[@]}

if (( CHAIN_LENGTH == 0 )); then
	echo "No GaLore runs specified." >&2
	exit 1
fi
if (( ${#CHAIN_LRS[@]} != CHAIN_LENGTH || ${#CHAIN_RESUME_RUNS[@]} != CHAIN_LENGTH || ${#CHAIN_OMEGAS[@]} != CHAIN_LENGTH || ${#CHAIN_SWITCH_SCALES[@]} != CHAIN_LENGTH )); then
	echo "Chain arrays are mismatched in length." >&2
	echo "Ensure CHAIN_RANKS, CHAIN_LRS, CHAIN_RESUME_RUNS, CHAIN_OMEGAS, and CHAIN_SWITCH_SCALES all have the same length." >&2
	exit 1
fi

RESUME_STEP=${RESUME_STEP:-2048}
TRAIN_STEPS=${TRAIN_STEPS:-12288}
ROTATE_MOMENTS=${ROTATE_MOMENTS:-true}
USE_ERROR_FEEDBACK=${USE_ERROR_FEEDBACK:-true}
GALORE_REGEX_PATTERN=${GALORE_REGEX_PATTERN:-"attention\\.w[qkv]|attention\\.wo|feed_forward\\.w[12]"}
FULL_RANK_THRESHOLD=${FULL_RANK_THRESHOLD:-2048}
GALORE_VS=${GALORE_VS:-"0.0"}
GALORE_QHM_OUTSIDE=${GALORE_QHM_OUTSIDE:-false}
# Hyperparameter switch overrides (comma-separated lists for arrays)
HP_SWITCH_ENABLED=${HP_SWITCH_ENABLED:-true}
DRY_RUN=${DRY_RUN:-false}
HP_SWITCH_STEPS=${HP_SWITCH_STEPS:-2048}
HP_SWITCH_NEW_VS=${HP_SWITCH_NEW_VS:-"0.95,"}
HP_SWITCH_NEW_BETAS=${HP_SWITCH_NEW_BETAS:-"0.999,0.999"}
HP_SWITCH_RESET_MOMENTA=${HP_SWITCH_RESET_MOMENTA:-"exp_avg,exp_avg_sq"}
LOCAL_DESLOC_PROJECTOR_SOURCE=${LOCAL_DESLOC_PROJECTOR_SOURCE:-"pseudo_grad"}

GENERATED_CONFIG_DIR=${GENERATED_CONFIG_DIR:-"${SCRIPT_DIR}/generated_configs"}
LOG_ROOT=${LOG_ROOT:-"${SCRIPT_DIR}/logs"}
RDZV_PORT_BASE=${RDZV_PORT_BASE:-40000}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
mkdir -p "${GENERATED_CONFIG_DIR}" "${LOG_ROOT}"

if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( NPROC_PER_NODE < 1 )); then
	echo "NPROC_PER_NODE must be a positive integer (got ${NPROC_PER_NODE})." >&2
	exit 1
fi

TRAINING_ARGS=("$@")


normalize_bool() {
	local value="${1:-}"
	case "${value,,}" in
		true|1|yes|on) echo "true" ;;
		false|0|no|off|"") echo "false" ;;
		*) echo "${value,,}" ;;
	esac
}

# Normalized flags used later for gating runs
HP_SWITCH_ENABLED_FLAG=$(normalize_bool "${HP_SWITCH_ENABLED}")
DRY_RUN_FLAG=$(normalize_bool "${DRY_RUN}")



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
proj_source_env = os.environ.get("DESLOC_PROJECTOR_SOURCE", "").strip()
if proj_source_env == "":
	desloc_projector_source = None
else:
	desloc_projector_source = proj_source_env
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

lr_sched_switch_scale_env = os.environ.get("LR_SCHED_SWITCH_SCALE", "").strip()

data = tomllib.loads(base.read_text(encoding="utf-8"))
optimizer = data.setdefault("optimizer", {})
optimizer["lr"] = lr
is_full_rank = rank >= full_rank_threshold
if is_full_rank:
		# Keep GaLore but disable projections for full-rank mode.
		desloc_cfg = optimizer.setdefault("desloc", {})
		if not isinstance(desloc_cfg, dict):  # pragma: no cover - defensive guard
			desloc_cfg = {}
			optimizer["desloc"] = desloc_cfg
		param_sync = desloc_cfg.get("param_sync_every")
		if param_sync is None:
			param_sync = optimizer.get("galore_update_proj_gap", 32)
		desloc_cfg["param_sync_every"] = param_sync
		optimizer["name"] = "GaLore"
		optimizer["galore_rotate_moments_on_refresh"] = rotate_flag
		optimizer["galore_use_error_feedback"] = use_error_feedback
		optimizer["galore_update_proj_gap"] = param_sync
		# Empty regex list prevents low-rank projections at full rank.
		optimizer["galore_param_regexes"] = []
else:
		# Use GaLore for low-rank mode
		optimizer["name"] = "GaLore"
		optimizer["galore_rotate_moments_on_refresh"] = rotate_flag
		optimizer["galore_use_error_feedback"] = use_error_feedback
		desloc_cfg = optimizer.setdefault("desloc", {})
		if not isinstance(desloc_cfg, dict):  # pragma: no cover - defensive guard
			desloc_cfg = {}
			optimizer["desloc"] = desloc_cfg
		param_sync = desloc_cfg.get("param_sync_every")
		if param_sync is None:
			param_sync = optimizer.get("galore_update_proj_gap", 32)
		desloc_cfg["param_sync_every"] = param_sync
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
		if desloc_projector_source is not None:
			desloc_cfg["low_rank_projector_source"] = desloc_projector_source

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

# Apply optional lr-scheduler switch scale override when provided.
if lr_sched_switch_scale_env:
	try:
		switch_scale_val = float(lr_sched_switch_scale_env)
	except Exception:
		switch_scale_val = None
	if switch_scale_val is not None:
		data.setdefault("lr_scheduler", {})["switch_scale"] = switch_scale_val

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(tomli_w.dumps(data), encoding="utf-8")
PY
}

total_planned=${CHAIN_LENGTH}
echo "Planned GaLore chain (${total_planned} runs, HP switch enabled only):"
for ((idx=0; idx<CHAIN_LENGTH; idx++)); do
	printf '  [%d] rank=%-4d lr=%-6s resume=%s omega=%s switch_scale=%s\n' "${idx}" "${CHAIN_RANKS[$idx]}" "${CHAIN_LRS[$idx]}" "${CHAIN_RESUME_RUNS[$idx]}" "${CHAIN_OMEGAS[$idx]}" "${CHAIN_SWITCH_SCALES[$idx]}"
done
echo ""

run_local_job() {
	local run_uuid=$1
	local config_path=$2
	local log_dir=$3
	local idx_label=$4
	local rank=$5
	local lr=$6
	local resume_run=$7

	cleanup() {
		pkill -P $$ || true
		pkill -f "torchft_lighthouse" || true
	}
	trap cleanup EXIT INT TERM

	cd "${REPO_ROOT}"

	find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true
	mkdir -p "${log_dir}"
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

	export TORCHFT_LIGHTHOUSE="http://${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT}"

	echo "[LOCAL TorchFT] RUN_UUID=${run_uuid} idx=${idx_label} rank=${rank} lr=${lr} resume=${resume_run}"
	echo "Config: ${config_path}"
	echo "Logs: ${log_dir}"

	uv run --no-sync torchft_lighthouse \
		--min_replicas ${MIN_REPLICAS} \
		--quorum_tick_ms ${QUORUM_TICK_MS} \
		--bind ${LIGHTHOUSE_HOST}:${LIGHTHOUSE_PORT} \
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
			rdzv_port=$((RDZV_PORT_BASE + replica_id))
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
	rank=${CHAIN_RANKS[$idx]}
	lr=${CHAIN_LRS[$idx]}
	resume_run=${CHAIN_RESUME_RUNS[$idx]}
	lr_tag=${lr/./p}

	if [[ "${HP_SWITCH_ENABLED_FLAG,,}" != "true" ]]; then
		echo "Skipping idx $((idx + 1)) (rank=${rank} lr=${lr}): hyperparameter switch is disabled (set HP_SWITCH_ENABLED=true)." >&2
		continue
	fi

	if [[ "${DRY_RUN_FLAG,,}" == "true" ]]; then
		echo "[DRY RUN][LOCAL] idx=$((idx + 1)) rank=${rank} lr=${lr} resume=${resume_run} omega=${CHAIN_OMEGAS[$idx]:-} switch_scale=${CHAIN_SWITCH_SCALES[$idx]:-}" >&2
		((total_runs++))
		continue
	fi

	run_suffix=$(printf "idx%02d-r%d-lr%s" $((idx + 1)) "${rank}" "${lr_tag}")
	run_uuid="${RUN_PREFIX}-${TIMESTAMP}-${run_suffix}"
	config_path="${GENERATED_CONFIG_DIR}/${run_uuid}.toml"
	log_dir="${LOG_ROOT}/${run_uuid}"
	mkdir -p "${log_dir}"

	HP_SWITCH_ENABLED="true" HP_SWITCH_NEW_VS="${CHAIN_OMEGAS[$idx]:-}" LR_SCHED_SWITCH_SCALE="${CHAIN_SWITCH_SCALES[$idx]:-}" \
		GALORE_QHM_OUTSIDE="${GALORE_QHM_OUTSIDE}" DESLOC_PROJECTOR_SOURCE="${LOCAL_DESLOC_PROJECTOR_SOURCE}" \
		run_python_config "${CONFIG_FILE}" "${config_path}" "${rank}" "${lr}" "${resume_run}"

	echo "[LOCAL] Launching ${run_uuid} (idx=$((idx + 1)), rank=${rank}, lr=${lr})"
	run_local_job "${run_uuid}" "${config_path}" "${log_dir}" "$((idx + 1))" "${rank}" "${lr}" "${resume_run}"
	((total_runs++))
done

if [[ "${DRY_RUN_FLAG,,}" == "true" ]]; then
	echo "Dry run complete: ${total_runs} run(s) planned (no execution)."
else
	echo "Completed ${total_runs} run(s) locally."
fi
