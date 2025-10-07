# Mosaic Streaming Dataloader Training - Run Instructions

## Quick Start

### Run from Repository Root

The `run_train.sh` script is designed to be run from the **repository root directory**:

```bash
cd /nfs-share/aai30/projects/torchtitan
./torchtitan/experiments/mosaic/run_train.sh
```

### Basic Usage Examples

**1. Default run (2 GPUs):**
```bash
./torchtitan/experiments/mosaic/run_train.sh
```

**2. Single GPU:**
```bash
NGPU=1 ./torchtitan/experiments/mosaic/run_train.sh
```

**3. 4 GPUs:**
```bash
NGPU=4 ./torchtitan/experiments/mosaic/run_train.sh
```

**4. Custom log rank:**
```bash
LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/mosaic/run_train.sh
```

**5. Custom config file:**
```bash
CONFIG_FILE=./my_custom_config.toml ./torchtitan/experiments/mosaic/run_train.sh
```

**6. With config overrides:**
```bash
./torchtitan/experiments/mosaic/run_train.sh --training.steps=1000 --model.name=llama3
```

### Environment Variables

The script accepts these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NGPU` | `2` | Number of GPUs to use |
| `LOG_RANK` | `0` | Which rank(s) to log from (e.g., `0,1` for ranks 0 and 1) |
| `S3_ENDPOINT_URL` | `http://taranaki.cl.cam.ac.uk:9000` | S3 endpoint for Mosaic streaming |
| `CONFIG_FILE` | `./torchtitan/experiments/mosaic/configs/mosaic_job.toml` | Path to config file |
| `TORCHFT_LIGHTHOUSE` | `http://localhost:29510` | Lighthouse server for fault tolerance |

### Command-line Overrides

You can override any config parameter using the `--job.config_key=value` syntax:

```bash
./torchtitan/experiments/mosaic/run_train.sh \
    --training.steps=5000 \
    --optimizer.lr=3e-4 \
    --model.name=llama3
```

## File Locations

### Script Location
- **Script**: `/nfs-share/aai30/projects/torchtitan/torchtitan/experiments/mosaic/run_train.sh`
- **Run from**: `/nfs-share/aai30/projects/torchtitan/` (repository root)

### Config Location
- **Default config**: `./torchtitan/experiments/mosaic/configs/mosaic_job.toml`
- Paths in the script are relative to the repository root

### Train Script
- **Python module**: `torchtitan/experiments/mosaic/train.py`
- Called directly as a Python script (not as a module with `-m`)

## Important Notes

1. **Working Directory**: Always run from the repository root (`/nfs-share/aai30/projects/torchtitan/`)
2. **Path Resolution**: All paths in the script are relative to the repository root
3. **S3 Endpoint**: Make sure the S3_ENDPOINT_URL is accessible from your environment
4. **Dependencies**: Ensure `llm-foundry` and `streaming` are installed:
   ```bash
   uv pip install llm-foundry mosaicml-streaming --no-deps
   ```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the repository root:
```bash
cd /nfs-share/aai30/projects/torchtitan
./torchtitan/experiments/mosaic/run_train.sh
```

### S3 Connection Issues
Check that the S3 endpoint is accessible:
```bash
curl http://taranaki.cl.cam.ac.uk:9000
```

### CUDA Errors
Ensure you have the correct number of GPUs available:
```bash
nvidia-smi
```

## Advanced Usage

### Multi-Node Training

For multi-node training, you'll need to set up the rendezvous endpoint:

```bash
# On the master node (e.g., node-0)
MASTER_ADDR=node-0 MASTER_PORT=29500 \
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node-0:29500 \
    torchtitan/experiments/mosaic/train.py \
    --job.config_file ./torchtitan/experiments/mosaic/configs/mosaic_job.toml
```

### With SLURM

If using SLURM, you can create a wrapper script:
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1

cd /nfs-share/aai30/projects/torchtitan
NGPU=8 ./torchtitan/experiments/mosaic/run_train.sh
```

## Example Session

```bash
# Navigate to repository root
cd /nfs-share/aai30/projects/torchtitan

# Run with 2 GPUs (default)
./torchtitan/experiments/mosaic/run_train.sh

# Or with custom settings
NGPU=4 LOG_RANK=0,1 ./torchtitan/experiments/mosaic/run_train.sh \
    --training.steps=10000 \
    --optimizer.lr=1e-4
```
