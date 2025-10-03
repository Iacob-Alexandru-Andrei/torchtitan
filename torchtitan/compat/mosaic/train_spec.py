"""Custom TrainSpecs that swap in Mosaic streaming dataloaders."""

from __future__ import annotations

from dataclasses import replace

from torchtitan.compat.mosaic.dataloader import build_mosaic_dataloader
from torchtitan.models.llama3 import get_train_spec as get_llama3_train_spec
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec


def register_llama3_mosaic() -> TrainSpec:
    """Register a Llama 3 TrainSpec that pulls data from Mosaic streaming."""

    base_spec = get_llama3_train_spec()
    mosaic_spec = replace(
        base_spec,
        name="llama3_mosaic",
        build_dataloader_fn=build_mosaic_dataloader,
    )
    register_train_spec(mosaic_spec)
    return mosaic_spec


__all__ = ["register_llama3_mosaic"]