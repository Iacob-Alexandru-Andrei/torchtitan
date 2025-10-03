# TorchTitan + Mosaic streaming example

This directory contains a minimal integration layer that lets TorchTitan models
consume streaming dataloaders configured with MosaicML's `llm-foundry`
utilities. The example keeps TorchTitan unchanged and demonstrates how to
extend it from the outside by registering custom `TrainSpec`s and by providing
launch scripts and configuration files.

## Prerequisites

* Install TorchTitan in editable mode (or add it to your `PYTHONPATH`).
* Install the Mosaic stack: `llm-foundry`, `composer`, and
  `mosaicml-streaming` (a.k.a. `streaming`).
* Download model tokenizer assets referenced by the config (defaults to the
  debug tokenizer shipped in `tests/assets/tokenizer`).

## Files

* `train.py` – light wrapper around `torchtitan.train.Trainer` that
  registers the `llama3_mosaic` TrainSpec, parses TorchTitan CLI options, attaches the Mosaic
  dataloader config, and starts training.
* `configs/mosaic_job.toml` – example TorchTitan job config that targets
  `llama3_mosaic`.
* `configs/mosaic_dataloader.toml` – example Mosaic streaming configuration
  forwarded verbatim to `llmfoundry.data.dataloader.build_dataloader`.

## Usage

1. Adjust `configs/mosaic_dataloader.toml` to point at your Mosaic streams. The
   sample configuration shows the most common options and includes comments
   describing what to change. Update the job config in `configs/mosaic_job.toml`
   as you would for any TorchTitan run (batch size, steps, optimizer, etc.).

2. Launch training via `torchrun`:

   ```bash
   torchrun --nproc_per_node=1 -m examples.mosaic_streaming.train \
     --config examples/mosaic_streaming/configs/mosaic_job.toml \
     --mosaic-config examples/mosaic_streaming/configs/mosaic_dataloader.toml
   ```

   Any additional TorchTitan CLI flags can be appended after a `--`, e.g.

   ```bash
   torchrun --nproc_per_node=1 -m examples.mosaic_streaming.train \
     --config examples/mosaic_streaming/configs/mosaic_job.toml \
     --mosaic-config examples/mosaic_streaming/configs/mosaic_dataloader.toml -- \
     --training.steps=20 --metrics.enable_tensorboard=true
   ```

The wrapper script mirrors the behavior of the stock `torchtitan/train.py`
entrypoint, including seed checkpoint creation and graceful shutdown of the
process group.