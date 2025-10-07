# MosaicML Streaming Integration for TorchTitan

This directory contains a refactored and generalized integration for using MosaicML's streaming datasets with TorchTitan. This integration is designed to be model-agnostic, allowing you to use Mosaic streaming with any model supported by TorchTitan through simple configuration changes.

## Overview

The integration consists of three main components:

1.  **Dataloader (`dataloader/`)**: Contains the `MosaicParallelAwareDataloader`, a wrapper around Mosaic's `StreamingTextDataset` that adds support for TorchTitan's distributed training and checkpointing. It also includes a `build_mosaic_dataloader` factory function to construct the dataloader from a configuration.

2.  **Configuration (`configs/`)**: Includes `MosaicJobConfig`, a custom configuration class that inherits from `JobConfig` and adds fields for Mosaic-specific settings. This provides a clean separation of concerns and makes the integration more modular. An example configuration file, `mosaic_job.toml`, demonstrates how to set up a training job.

3.  **Model Utilities (`models/`)**: Provides a `get_mosaic_train_spec` function that takes a base `TrainSpec` and dynamically replaces its dataloader and tokenizer builders with the Mosaic versions. This is the key to the model-agnostic design of the integration.

## How to Run

To run a training job with Mosaic streaming, you can use the provided `train.py` script. This script handles the configuration parsing, `TrainSpec` modification, and trainer launch.

Here's an example command to launch a training job:

```bash
torchrun --nproc_per_node=2 experiments/mosaic/train.py --config-path experiments/mosaic/configs/mosaic_job.toml
```

## Configuration

The training job is configured using the `mosaic_job.toml` file. Here are the key sections to be aware of:

*   **`[model]`**:
    *   `name`: The name of the base model's `TrainSpec` you want to use (e.g., "llama3"). The training script will dynamically apply the Mosaic customizations to this `TrainSpec`.

*   **`[mosaic_dataloader]`**:
    *   This section contains all the configuration for the Mosaic `StreamingTextDataset`. You can specify the dataset name, paths, shuffling options, and other parameters here. Refer to the llm-foundry documentation for a complete list of available options.

*   **`[mosaic_tokenizer]`**:
    *   This section configures the tokenizer to be used with the Mosaic dataloader. You can specify the tokenizer name (e.g., from HuggingFace) and any additional keyword arguments.

## Using with Different Models

To use this integration with a different model, you only need to make two simple changes to the `mosaic_job.toml` file:

1.  **Update `model.name`**: Change the `name` field in the `[model]` section to the `TrainSpec` name of the model you want to train (e.g., "my_cool_model").
2.  **Update `model.flavor`**: Change the `flavor` field to the specific model configuration you want to use.

The `train.py` script will automatically fetch the correct base `TrainSpec`, apply the Mosaic customizations, and launch the training job. No code changes are required.