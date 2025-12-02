#!/usr/bin/env bash
# Launch Flux-schnell with Mosaic trainer (DDP style).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../.." && pwd -P)
cd "${REPO_ROOT}"
echo REPO_ROOT="${REPO_ROOT}"

# Ensure autoencoder checkpoint is available.
AE_PATH=${AUTOENCODER_PATH:-"${REPO_ROOT}/torchtitan/experiments/flux/assets/autoencoder/ae.safetensors"}
if [ ! -f "${AE_PATH}" ]; then
  echo "[flux] Autoencoder missing at ${AE_PATH}; attempting download..."
  AE_DIR=$(dirname "${AE_PATH}")
  mkdir -p "${AE_DIR}"
  HF_TOKEN_VALUE=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}
  if [ -z "${HF_TOKEN_VALUE}" ]; then
    echo "[flux] Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN to download private checkpoint." >&2
    exit 1
  fi
  uv run --no-sync python torchtitan/experiments/flux/scripts/download_autoencoder.py \
    --repo_id "${AUTOENCODER_REPO:-black-forest-labs/FLUX.1-dev}" \
    --ae_path "${AUTOENCODER_FILENAME:-ae.safetensors}" \
    --local_dir "${AE_DIR}" \
    --hf_token "${HF_TOKEN_VALUE}"
fi

# Remove stale shared-memory artifacts owned by the current user.
find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}

CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/flux_baseline_ddp.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
LOG_RANK=${LOG_RANK:-0}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"localhost:0"}

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_PREFIX=${RUN_PREFIX:-"flux-ddp"}
export RUN_UUID=${RUN_UUID:-"${RUN_PREFIX}-${TIMESTAMP}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan_flux"}
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
  "${TRAINING_ARGS[@]}"
