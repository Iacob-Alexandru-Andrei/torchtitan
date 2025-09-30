# TorchTitan + Mosaic streaming example

This directory contains a minimal integration layer that lets TorchTitan models
consume streaming dataloaders configured with MosaicML's `llm-foundry`
utilities. The example keeps TorchTitan unchanged and demonstrates how to
extend it from the outside by registering custom `TrainSpec`s and by providing
launch scripts and configuration files.

## Prerequisites

* Install TorchTitan in editable mode (or add it to your `PYTHONPATH`).
* Install the Mosaic stack that Photon depends on: `llm-foundry`, `composer`, and
  `mosaicml-streaming` (a.k.a. `streaming`).
* Download model tokenizer assets referenced by the config (defaults to the
  debug tokenizer shipped in `tests/assets/tokenizer`).

## Files

* `train_spec.py` – registers two TrainSpecs:
  * `llama3_mosaic` reuses the stock Llama 3 components but swaps in the Mosaic
    streaming dataloader.
  * `mpt_mup_mosaic` exposes a Composer-based MPT implementation that follows
    the muP/CompleteP recipe from Photon. The registration is skipped at runtime
    if `llm-foundry` is not available.
* `train_with_mosaic.py` – light wrapper around `torchtitan.train.Trainer` that
  registers the TrainSpecs, parses TorchTitan CLI options, attaches the Mosaic
  dataloader config, and starts training.
* `models/` – holds the muP-enabled MPT port used by the `mpt_mup_mosaic`
  TrainSpec.
* `callbacks/` – Composer callbacks that mirror Photon behaviour, including the
  optimizer monitor and quasi-hyperbolic parameter logger. They live entirely in
  this example package so TorchTitan itself stays untouched.
* `optimizers/` – the QHADOPT implementation registered with llm-foundry's
  optimizer registry.
* `schedulers/` – warmup/stable/decay schedulers ported from Photon and
  registered with the llm-foundry scheduler registry.
* `configs/mosaic_job.toml` – example TorchTitan job config that targets
  `llama3_mosaic`.
* `configs/mpt_mup_job.toml` – example job config that selects the muP-enabled
  MPT TrainSpec.
* `configs/mosaic_dataloader.toml` – example Mosaic streaming configuration
  forwarded verbatim to `llmfoundry.data.dataloader.build_dataloader`.

## Usage

1. Adjust `configs/mosaic_dataloader.toml` to point at your Mosaic streams. The
   sample configuration shows the most common options and includes comments
   describing what to change. Update the job config corresponding to the model
   you want to train (`mosaic_job.toml` for Llama or `mpt_mup_job.toml` for the
   muP MPT) as you would for any TorchTitan run (batch size, steps, optimizer,
   etc.).
2. Launch training via `torchrun` (this example uses the Llama config):

   ```bash
   torchrun --nproc_per_node=1 -m examples.mosaic_photon.train_with_mosaic \
     --config examples/mosaic_photon/configs/mosaic_job.toml \
     --mosaic-config examples/mosaic_photon/configs/mosaic_dataloader.toml
   ```

   Any additional TorchTitan CLI flags can be appended after a `--`, e.g.

   ```bash
   torchrun --nproc_per_node=1 -m examples.mosaic_photon.train_with_mosaic \
     --config examples/mosaic_photon/configs/mpt_mup_job.toml \
     --mosaic-config examples/mosaic_photon/configs/mosaic_dataloader.toml -- \
     --training.steps=20 --metrics.enable_tensorboard=true
   ```

The wrapper script mirrors the behavior of the stock `torchtitan/train.py`
entrypoint, including seed checkpoint creation and graceful shutdown of the
process group.
