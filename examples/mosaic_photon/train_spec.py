"""Custom TrainSpecs that swap in Mosaic streaming dataloaders and models."""

from __future__ import annotations

from dataclasses import replace

from .dataloader import build_mosaic_dataloader
from .models import (
    HAS_MPT_MUP_SUPPORT,
    MPT_MUP_CONFIGS,
    MPTMuPModelArgs,
    TitanComposerMPTMuP,
)
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
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


def register_mpt_mup_mosaic() -> TrainSpec:
    """Register a muP-enabled MPT TrainSpec that uses Mosaic streaming."""

    if not HAS_MPT_MUP_SUPPORT:
        raise RuntimeError(
            "llm-foundry must be installed to register the mpt_mup_mosaic TrainSpec."
        )

    mpt_args = {
        name: MPTMuPModelArgs(config_name=name)
        for name in MPT_MUP_CONFIGS
    }

    def _noop_parallelize(model, parallel_dims, job_config):  # pragma: no cover - simple passthrough
        return model

    spec = TrainSpec(
        name="mpt_mup_mosaic",
        model_cls=TitanComposerMPTMuP,
        model_args=mpt_args,
        parallelize_fn=_noop_parallelize,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mosaic_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
    register_train_spec(spec)
    return spec


__all__ = ["register_llama3_mosaic", "register_mpt_mup_mosaic"]
