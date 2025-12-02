# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flux validator variant that uses the FL dataloader builder."""

from __future__ import annotations

import os
from typing import Generator

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.fl.dataloader.flux_builder import (
    build_fl_flux_validation_dataloader,
)
from torchtitan.experiments.flux.dataset.tokenizer import build_flux_tokenizer
from torchtitan.experiments.flux.model.autoencoder import AutoEncoder
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.sampling import generate_image, save_image
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
    unpack_latents,
)
from torchtitan.tools.logging import logger


class FLFluxValidator(Validator):
    """Validator that mirrors FluxValidator but uses the FL dataloader builder."""

    def __init__(
        self,
        job_config: JobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
    ):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.all_timesteps = self.job_config.validation.all_timesteps
        self.validation_dataloader = build_fl_flux_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            generate_timestamps=not self.all_timesteps,
            infinite=self.job_config.validation.steps != -1,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor
        self.t5_tokenizer, self.clip_tokenizer = build_flux_tokenizer(self.job_config)

        if self.job_config.validation.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    def flux_init(
        self,
        device: torch.device,
        _dtype: torch.dtype,
        autoencoder: AutoEncoder,
        t5_encoder: FluxEmbedder,
        clip_encoder: FluxEmbedder,
    ):
        self.device = device
        self._dtype = _dtype
        self.autoencoder = autoencoder
        self.t5_encoder = t5_encoder
        self.clip_encoder = clip_encoder

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        model = model_parts[0]
        model.eval()

        training_cfg_prob = self.job_config.training.classifier_free_guidance_prob
        self.job_config.training.classifier_free_guidance_prob = 0.0

        save_img_count = self.job_config.validation.save_img_count
        parallel_dims = self.parallel_dims
        accumulated_losses = []
        device_type = dist_utils.device_type
        num_steps = 0

        for input_dict, labels in self.validation_dataloader:
            if (
                self.job_config.validation.steps != -1
                and num_steps >= self.job_config.validation.steps
            ):
                break

            prompt = input_dict.pop("prompt")
            if not isinstance(prompt, list):
                prompt = [prompt]
            for p in prompt:
                if save_img_count != -1 and save_img_count <= 0:
                    break
                image = generate_image(
                    device=self.device,
                    dtype=self._dtype,
                    job_config=self.job_config,
                    model=model,
                    prompt=p,
                    autoencoder=self.autoencoder,
                    t5_tokenizer=self.t5_tokenizer,
                    clip_tokenizer=self.clip_tokenizer,
                    t5_encoder=self.t5_encoder,
                    clip_encoder=self.clip_encoder,
                )

                save_image(
                    name=f"image_rank{str(torch.distributed.get_rank())}_{step}.png",
                    output_dir=os.path.join(
                        self.job_config.job.dump_folder,
                        self.job_config.validation.save_img_folder,
                    ),
                    x=image,
                    add_sampling_metadata=True,
                    prompt=p,
                )
                save_img_count -= 1

            input_dict["image"] = labels
            input_dict = preprocess_data(
                device=self.device,
                dtype=self._dtype,
                autoencoder=self.autoencoder,
                clip_encoder=self.clip_encoder,
                t5_encoder=self.t5_encoder,
                batch=input_dict,
            )
            labels = input_dict["img_encodings"].to(device_type)
            clip_encodings = input_dict["clip_encodings"]
            t5_encodings = input_dict["t5_encodings"]

            bsz = labels.shape[0]

            if self.all_timesteps:
                stratified_timesteps = torch.tensor(
                    [1 / 8 * (i + 0.5) for i in range(8)],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(bsz)
                clip_encodings = clip_encodings.repeat_interleave(8, dim=0)
                t5_encodings = t5_encodings.repeat_interleave(8, dim=0)
                labels = labels.repeat_interleave(8, dim=0)
            else:
                stratified_timesteps = input_dict.pop("timestep")

            self.metrics_processor.ntokens_since_last_log += labels.numel()

            with torch.no_grad(), torch.device(self.device):
                noise = torch.randn_like(labels)
                timesteps = stratified_timesteps.to(labels)
                sigmas = timesteps.view(-1, 1, 1, 1)
                noised = sigmas * labels + (1 - sigmas) * noise

                packed_noised, pack_shape = pack_latents(noised)
                packed_clip_encodings, _ = pack_latents(clip_encodings)
                packed_t5_encodings, _ = pack_latents(t5_encodings)

                clip_pos_enc = create_position_encoding_for_latents(
                    bsz=packed_clip_encodings.shape[0],
                    latent_height=parallel_dims.img_seq_len,
                    latent_width=parallel_dims.img_seq_len,
                    position_dim=parallel_dims.position_dim,
                ).to(self.device)

                outputs = model(
                    packed_noised,
                    img_ids=clip_pos_enc,
                    txt=packed_t5_encodings,
                    txt_ids=parallel_dims.txt_seq_pos_ids.to(self.device),
                    timesteps=timesteps,
                    y=packed_clip_encodings,
                )

                preds = unpack_latents(outputs, *pack_shape)
                loss = self.loss_fn(preds, noised, sigmas=sigmas)
                accumulated_losses.append(loss.detach())
                num_steps += 1

        losses = torch.stack(accumulated_losses)
        loss = losses.mean().item()
        logger.info("[Flux Validation] Step %s | Mean loss: %s", step, loss)
        self.metrics_processor.log_loss(step, loss)

        self.job_config.training.classifier_free_guidance_prob = training_cfg_prob


def build_fl_flux_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
):
    """Build a Flux validator wired to the FL dataloader."""
    return FLFluxValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )
