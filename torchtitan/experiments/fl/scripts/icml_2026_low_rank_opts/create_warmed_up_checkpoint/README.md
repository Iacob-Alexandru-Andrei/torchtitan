# create_warmed_up_checkpoint

Utilities for producing a "warmed up" checkpoint by running only the warmup phase of Mosaic Llama training. The launcher script pins the training horizon to the warmup length, fixes the worker count to four GPUs by default, and exposes knobs for model size, batch size, optimizer choice, and optimizer hyperparameters.

## Quick start

```bash
cd torchtitan/experiments/fl/scripts/iclr_2026_rebuttal/create_warmed_up_checkpoint
./run_create_warmed_up_checkpoint.sh
```

The defaults match the request from the rebuttal experiments:

- model size: `125M`
- global batch size: `256` (each of 4 workers trains with a local batch of 64)
- steps (and warmup length): `2048`
- optimizer: `QHAdamW` with `vs = [1.0]`, `betas = [0.9, 0.999]`, `lr = 0.01`

Override any value via flags or environment variables, e.g.

```bash
GLOBAL_BATCH_SIZE=512 TARGET_STEPS=1024 \
  ./run_create_warmed_up_checkpoint.sh --model-size 350M --vs "[1.0, 0.98]"
```

Additional training overrides can be appended after `--`, for example:

```bash
./run_create_warmed_up_checkpoint.sh --steps 1024 -- \
  --training.seq_len 4096 --training.dataset c4
```
