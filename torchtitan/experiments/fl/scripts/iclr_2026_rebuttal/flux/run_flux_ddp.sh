#!/usr/bin/env bash
# Launch Flux-schnell with Mosaic trainer (DDP style).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../.." && pwd -P)
cd "${REPO_ROOT}"
echo REPO_ROOT="${REPO_ROOT}"

CONFIG_FILE=${CONFIG_FILE:-"${SCRIPT_DIR}/flux_baseline_ddp.toml"}
TRAIN_MODULE=${TRAIN_MODULE:-"torchtitan.experiments.fl.train"}
NGPU=${NGPU:-4}
LOG_RANK=${LOG_RANK:-0}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-"localhost:0"}

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

# Persist HuggingFace caches so streaming datasets are reused across runs.
DATA_CACHE_ROOT=${DATA_CACHE_ROOT:-"${REPO_ROOT}/.cache/hf"}
export HF_HOME="${HF_HOME:-${DATA_CACHE_ROOT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATA_CACHE_ROOT}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_CACHE_ROOT}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Prefetch dataset shards to cover the planned training duration.
if [ "${PREFETCH_DATA:-1}" != "0" ]; then
  PREFETCH_STEPS=${PREFETCH_STEPS:-5120}
  PREFETCH_GLOBAL_BATCH=${PREFETCH_GLOBAL_BATCH:-}
  PREFETCH_SAMPLES=${PREFETCH_SAMPLES:-}
  DATASET_NAME=${DATASET_NAME:-}
  DATASET_PATH=${DATASET_PATH:-}
  echo "[flux] Prefetching dataset into ${HF_DATASETS_CACHE} (steps=${PREFETCH_STEPS})..."
  uv run --no-sync python - <<'PY'
import itertools
import os
import tomllib
from datasets import load_dataset, DownloadConfig

cfg_file = os.environ.get("CONFIG_FILE")
dataset_name = os.environ.get("DATASET_NAME")
dataset_path = os.environ.get("DATASET_PATH") or None
prefetch_steps = int(os.environ.get("PREFETCH_STEPS", "0"))
prefetch_samples_env = os.environ.get("PREFETCH_SAMPLES")
prefetch_global_batch_env = os.environ.get("PREFETCH_GLOBAL_BATCH")
ngpu = int(os.environ.get("NGPU", "1"))

if cfg_file and (not dataset_name or not prefetch_global_batch_env):
    with open(cfg_file, "rb") as f:
        cfg = tomllib.load(f)
    training_cfg = cfg.get("training", {})
    if not dataset_name:
        dataset_name = training_cfg.get("dataset", "pixparse/cc12m-wds")
    if not prefetch_global_batch_env:
        local_bs = int(training_cfg.get("local_batch_size", 1))
        prefetch_global_batch_env = str(local_bs * ngpu)

dataset_name = dataset_name or "pixparse/cc12m-wds"
global_batch = int(prefetch_global_batch_env or 256)
target = int(prefetch_samples_env or global_batch * prefetch_steps)

cache_dir = os.environ["HF_DATASETS_CACHE"]
download_cfg = DownloadConfig(cache_dir=cache_dir, max_retries=5)

print(f"[prefetch] dataset={dataset_name} target_samples={target} cache={cache_dir}")
if target <= 0:
    raise SystemExit("[prefetch] Target samples is zero; skipping prefetch.")

ds = load_dataset(
    dataset_name,
    split="train",
    streaming=True,
    download_config=download_cfg,
    data_dir=dataset_path,
)

for i, _ in zip(range(target), ds):
    if (i + 1) % 10000 == 0:
        print(f"[prefetch] streamed {i + 1}/{target}", flush=True)

print(f"[prefetch] streamed {target} samples into cache.")
PY
fi

# Remove stale shared-memory artifacts owned by the current user.
find /dev/shm -maxdepth 1 -user "${USER}" -exec rm -rf {} + 2>/dev/null || true

export S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-'http://taranaki.cl.cam.ac.uk:9000'}

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
