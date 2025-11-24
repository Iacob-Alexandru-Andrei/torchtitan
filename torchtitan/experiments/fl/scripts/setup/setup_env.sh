#!/bin/bash
uv sync
uv run --no-sync pre-commit install
uv pip install llm-foundry==0.22.0
uv pip install --pre --force-reinstall \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --extra-index-url https://pypi.nvidia.com \
  --only-binary=:all: \
  "torch==2.10.0.dev20251110+cu128" \
  "torchvision==0.25.0.dev20251110+cu128" \
  "nvidia-nvshmem-cu12==3.4.5"
uv pip install "torchft-nightly==2025.10.11"
