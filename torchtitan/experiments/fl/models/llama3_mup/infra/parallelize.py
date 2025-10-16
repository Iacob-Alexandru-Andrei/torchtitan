# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization helpers tuned for the MuP-flavoured LLaMA 3 architecture."""

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
    from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import Transformer


def _apply_mup_tp(
    model: Transformer,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
) -> None:
    """Apply tensor parallelism while accounting for MuP-specific norms."""
    root_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        ),
    }
    if getattr(model, "embedding_norm", None) is not None:
        root_plan["embedding_norm"] = SequenceParallel()

    parallelize_module(model, tp_mesh, root_plan)

    if enable_float8_tensorwise_tp:
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }

        if getattr(transformer_block, "post_attn_norm", None) is not None:
            layer_plan["post_attn_norm"] = SequenceParallel()
        if getattr(transformer_block, "post_ffn_norm", None) is not None:
            layer_plan["post_ffn_norm"] = SequenceParallel()

        parallelize_module(transformer_block, tp_mesh, layer_plan)

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the MuP LLaMA model"
    )


def parallelize_llama_mup(
    model: Transformer,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> Transformer:
    """Apply PT-D parallelisms to the MuP LLaMA variant, respecting MuP norms."""
    world_mesh = parallel_dims.world_mesh
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    use_flex_attn = getattr(model.model_args, "use_flex_attn", False)
    if job_config.parallelism.context_parallel_degree > 1 and use_flex_attn:
        msg = "CP support for FlexAttention is still in progress."
        raise NotImplementedError(msg)

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        _apply_mup_tp(
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
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
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
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
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
        logger.info("Applied DDP to the model")

    if (
        model.model_args.tie_word_embeddings
        and model.output is not None
        and model.tok_embeddings is not None
    ):
        model.output.weight = model.tok_embeddings.weight

    return cast("Transformer", model)
