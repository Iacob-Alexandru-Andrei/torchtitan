# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization helpers for the MuP-enabled MPT model."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import (
    _op_sac_save_list,
    apply_compile as _apply_compile,
    apply_ddp as _apply_ddp,
    apply_fsdp as _apply_fsdp,
)
from torchtitan.tools.logging import logger


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torchtitan.distributed import ParallelDims

    from ..model.mup_model import MPTCompletePBlock, Transformer


def _apply_mpt_tp(
    model: Transformer,
    tp_mesh: DeviceMesh,
    *,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
) -> None:
    transformer = model.transformer

    parallelize_module(
        transformer,
        tp_mesh,
        {
            "wte": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm_f": SequenceParallel(),
        },
    )

    if model.model.lm_head is not None:
        parallelize_module(
            model.model,
            tp_mesh,
            {
                "lm_head": ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1) if loss_parallel else Replicate(),
                    use_local_output=not loss_parallel,
                ),
            },
        )

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (  # noqa: PLC0415
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel = Float8RowwiseParallel
        colwise_parallel = Float8ColwiseParallel
        prepare_module_input = PrepareFloat8ModuleInput
    else:
        rowwise_parallel = RowwiseParallel
        colwise_parallel = ColwiseParallel
        prepare_module_input = PrepareModuleInput

    for block in transformer.blocks:
        layer_plan: dict[str, object] = {}
        block = cast("MPTCompletePBlock", block)

        if block.fuse_norm_attn_norm:
            layer_plan["norm_attn_norm.norm_1"] = SequenceParallel()
            if block.norm_attn_norm.norm_2 is not None:  # type: ignore[attr-defined]
                layer_plan["norm_attn_norm.norm_2"] = SequenceParallel()
            attn_module = block.norm_attn_norm.attn  # type: ignore[attr-defined]
            attn_prefix = "norm_attn_norm.attn"
        else:
            layer_plan["norm_1"] = SequenceParallel()
            if block.norm_2 is not None:
                layer_plan["norm_2"] = SequenceParallel()
            attn_module = block.attn
            attn_prefix = "attn"

        layer_plan[attn_prefix] = prepare_module_input(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )

        if hasattr(attn_module, "Wqkv"):
            layer_plan[f"{attn_prefix}.Wqkv"] = colwise_parallel()
        else:
            if hasattr(attn_module, "Wq"):
                layer_plan[f"{attn_prefix}.Wq"] = colwise_parallel()
            if hasattr(attn_module, "Wk"):
                layer_plan[f"{attn_prefix}.Wk"] = colwise_parallel()
            if hasattr(attn_module, "Wv"):
                layer_plan[f"{attn_prefix}.Wv"] = colwise_parallel()
        if hasattr(attn_module, "out_proj"):
            layer_plan[f"{attn_prefix}.out_proj"] = rowwise_parallel(
                output_layouts=Shard(1)
            )

        layer_plan["ffn"] = prepare_module_input(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        ffn_module = block.ffn
        if hasattr(ffn_module, "up_proj"):
            layer_plan["ffn.up_proj"] = colwise_parallel()
        if hasattr(ffn_module, "gate"):
            layer_plan["ffn.gate"] = colwise_parallel()
        if hasattr(ffn_module, "down_proj"):
            layer_plan["ffn.down_proj"] = rowwise_parallel(output_layouts=Shard(1))

        if getattr(block, "post_attn_norm", None) is not None:
            layer_plan["post_attn_norm"] = SequenceParallel()
        if getattr(block, "post_ffn_norm", None) is not None:
            layer_plan["post_ffn_norm"] = SequenceParallel()

        parallelize_module(block, tp_mesh, layer_plan)

    logger.info(
        "Applied %sTensor Parallelism to MuP MPT model",
        "Float8 tensorwise " if enable_float8_tensorwise_tp else "",
    )


def parallelize_mpt_mup(
    model: Transformer,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> Transformer:
    world_mesh = parallel_dims.world_mesh
    assert job_config.training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {job_config.training.seq_len} must be divisible by the "
        f"product of TP degree ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp})."
    )

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        _apply_mpt_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if job_config.activation_checkpoint.mode != "none":
        ac_target = getattr(model, "transformer", None)
        if ac_target is None:
            ac_target = model
        apply_ac(
            ac_target,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=False,
            op_sac_save_list=_op_sac_save_list,
        )

    if model_compile_enabled:
        _apply_compile(model, job_config.compile)

    if parallel_dims.fsdp_enabled:
        dp_mesh_dim_names = (
            ("dp_replicate", "dp_shard_cp")
            if parallel_dims.dp_replicate_enabled
            else ("dp_shard_cp",)
        )

        _apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the MuP MPT model")
        else:
            logger.info("Applied FSDP to the MuP MPT model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the MuP MPT model")
        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the MuP MPT model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            msg = "DDP has not supported > 1D parallelism"
            raise RuntimeError(msg)
        _apply_ddp(
            model,
            world_mesh,
            enable_compile=model_compile_enabled,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )
        logger.info("Applied DDP to the MuP MPT model")

    if model.model.config.tie_word_embeddings:
        model.model.tie_weights()

    return model


__all__ = ["parallelize_mpt_mup"]
