#!/usr/bin/env bash
# Launch the quasi-hyperbolic Muon 16M run with Mosaic's trainer (DDP).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../.." && pwd -P)
cd "${REPO_ROOT}"

# Remove stale shared-memory artifacts owned by the current user.
find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}

CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/base.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
LOG_RANK=${LOG_RANK:-0}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"localhost:0"}

LR_SWITCH_STEP=${LR_SWITCH_STEP:-2049}
SWITCH_SCALE=${SWITCH_SCALE:-1.0}
QHMUON_V=${QHMUON_V:-0.98}
QHMUON_NEW_BETAS=${QHMUON_NEW_BETAS:-"0.9 0.999"}
QHMUON_RESET_MOMENTA=${QHMUON_RESET_MOMENTA:-"exp_avg"}

read -r -a QHMUON_NEW_BETAS_ARRAY <<< "${QHMUON_NEW_BETAS}"
read -r -a QHMUON_RESET_MOMENTA_ARRAY <<< "${QHMUON_RESET_MOMENTA}"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_PREFIX=${RUN_PREFIX:-"iclr2026-qhmuon16m"}
export RUN_UUID=${RUN_UUID:-"${RUN_PREFIX}-${TIMESTAMP}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_tune_N"}
export WANDB_TEAM=${WANDB_TEAM:-"camlsys"}
export WANDB_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_WANDB_BASE_RUN_NAME="${RUN_UUID}"
export TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX=${TORCHTITAN_FORCE_WANDB_WORKER_SUFFIX:-1}

TRAINING_ARGS=("$@")

PYTORCH_ALLOC_CONF="expandable_segments:True" \
uv run --no-sync torchrun \
  --nproc_per_node="${NGPU}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  --local-ranks-filter "${LOG_RANK}" \
  --role rank \
  --tee 3 \
  -m "${TRAIN_MODULE}" \
  --job.config_file "${CONFIG_FILE}" \
  --run_uuid "${RUN_UUID}" \
  --lr_scheduler.switch_step "${LR_SWITCH_STEP}" \
  --lr_scheduler.switch_scale "${SWITCH_SCALE}" \
  --fl_metrics.hyperparameter_switch.steps "${LR_SWITCH_STEP}" \
  --fl_metrics.hyperparameter_switch.new_vs "${QHMUON_V}" \
  --fl_metrics.hyperparameter_switch.new_betas "${QHMUON_NEW_BETAS_ARRAY[@]}" \
  --fl_metrics.hyperparameter_switch.reset_momenta "${QHMUON_RESET_MOMENTA_ARRAY[@]}" \
  "${TRAINING_ARGS[@]}"
