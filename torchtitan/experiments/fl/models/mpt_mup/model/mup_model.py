# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MuP-enabled MPT model adapted for TorchTitan experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch

from llmfoundry.layers_registry import norms
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.layers.custom_embedding import SharedEmbedding
from llmfoundry.models.layers.layer_builders import build_norm
from llmfoundry.models.mpt import modeling_mpt as llmfoundry_mpt_modeling
from llmfoundry.models.mpt.modeling_mpt import (
    MPTForCausalLM,
    MPTModel,
    PartialLlamaConfig,
)
from torch import nn


# Allow newer transformers configs to access label-map helpers when constructing RoPE.
_patched_allowed_keys = set(llmfoundry_mpt_modeling._ALLOWED_LLAMA_CONFIG_KEYS)
_patched_allowed_keys.update(
    {
        "_create_id_label_maps",
        "_set_label2id_id2label",
        "id2label",
        "label2id",
        "num_labels",
    }
)
llmfoundry_mpt_modeling._ALLOWED_LLAMA_CONFIG_KEYS = frozenset(_patched_allowed_keys)

_partial_llama_orig_getattribute = PartialLlamaConfig.__getattribute__
_partial_llama_orig_getitem = PartialLlamaConfig.__getitem__


def _partial_llama_getattribute(self: PartialLlamaConfig, key: str):
    if key in {
        "_create_id_label_maps",
        "_set_label2id_id2label",
        "id2label",
        "label2id",
        "num_labels",
    }:
        return super(PartialLlamaConfig, self).__getattribute__(key)
    return _partial_llama_orig_getattribute(self, key)


PartialLlamaConfig.__getattribute__ = _partial_llama_getattribute  # type: ignore[assignment]


def _partial_llama_getitem(self: PartialLlamaConfig, key: str):
    if key in {
        "_create_id_label_maps",
        "_set_label2id_id2label",
        "id2label",
        "label2id",
        "num_labels",
    }:
        return super(PartialLlamaConfig, self).__getitem__(key)
    return _partial_llama_orig_getitem(self, key)


PartialLlamaConfig.__getitem__ = _partial_llama_getitem  # type: ignore[assignment]

from .mup_args import ModelInitConfig, MPTMuPConfig, MuPConfig, TransformerModelArgs

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, MutableMapping

    from torch.nn.parameter import Parameter
    from transformers.modeling_outputs import (
        BaseModelOutputWithPast,
        CausalLMOutputWithPast,
    )


@dataclass
class MuPOptimizerOverride:
    """MuP-specific optimizer adjustments returned by compatible models."""

    param_groups: list[dict[str, Any]] | None
    config_updates: dict[str, Any]


class MuPSharedEmbedding(SharedEmbedding):
    """SharedEmbedding wrapper that applies MuP-specific scaling and optional norm."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        scale: float = 1.0,
        use_embedding_norm: bool = False,
        norm_type: str = "low_precision_layernorm",
        norm_eps: float = 1e-5,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab_size, d_model, device=device, **kwargs)
        self.scale = scale
        self.norm = (
            None
            if not use_embedding_norm
            else build_norm(
                name=norm_type.lower(),
                normalized_shape=d_model,
                eps=norm_eps,
                device=device,
            )
        )

    def forward(self, input: torch.Tensor, unembed: bool = False) -> torch.Tensor:
        out = super().forward(input, unembed)
        if not unembed:
            if self.norm is not None:
                out = self.norm(out)
            out = out * self.scale
        return out


class MPTCompletePBlock(MPTBlock):
    """Extends :class:`MPTBlock` with CompleteP residual scaling and peri norms."""

    def __init__(
        self,
        *,
        depth_multiplier: float = 1.0,
        depth_alpha_enabled: bool = False,
        depth_alpha_exp: float = 1.0,
        use_peri_norm: bool = False,
        norm_type: str = "low_precision_layernorm",
        norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(norm_type=norm_type, norm_eps=norm_eps, **kwargs)
        self.residual_scaling = (
            1.0 / (depth_multiplier**depth_alpha_exp) if depth_alpha_enabled else 1.0
        )
        self.post_attn_norm = (
            build_norm(
                name=norm_type.lower(),
                normalized_shape=kwargs["d_model"],
                eps=norm_eps,
                device=kwargs.get("device"),
            )
            if use_peri_norm
            else None
        )
        self.post_ffn_norm = (
            build_norm(
                name=norm_type.lower(),
                normalized_shape=kwargs["d_model"],
                eps=norm_eps,
                device=kwargs.get("device"),
            )
            if use_peri_norm
            else None
        )

    def forward(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_bias: torch.Tensor | None = None,
        rotary_emb_w_meta_info: dict | None = None,
        attention_mask: torch.ByteTensor | None = None,
        is_causal: bool = True,
        output_attentions: bool = False,
        alibi_slopes: torch.Tensor | None = None,
        flash_attn_padding_info: dict[str, torch.Tensor] | None = None,
        prev_layer_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_value_states: torch.Tensor | None = None,
        x_prev: torch.Tensor | None = None,
        pos_id_within_seq: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None
    ]:
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs["prev_layer_key_value"] = prev_layer_key_value
        if key_value_states is not None:
            extra_kwargs["key_value_states"] = key_value_states
        if pos_id_within_seq is not None:
            extra_kwargs["pos_id_within_seq"] = pos_id_within_seq

        if self.fuse_norm_attn_norm:
            a, attn_weights, past_key_value = self.norm_attn_norm(  # type: ignore[attr-defined]
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=is_causal,
                output_attentions=output_attentions,
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
                x_prev=x_prev,
                **extra_kwargs,
            )
        else:
            a = self.norm_1(x)
            a, attn_weights, past_key_value = self.attn(
                a,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=is_causal,
                needs_weights=output_attentions,
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
                x_prev=x_prev,
                **extra_kwargs,
            )
        if self.post_attn_norm is not None:
            a = self.post_attn_norm(a)

        x = x + self.resid_attn_dropout(a) * self.residual_scaling
        m = (
            self.norm_attn_norm.norm_2(x)  # type: ignore[attr-defined]
            if self.fuse_norm_attn_norm and self.norm_attn_norm.norm_2 is not None  # type: ignore[attr-defined]
            else (self.norm_2(x) if self.norm_2 is not None else x)
        )

        n = self.ffn(m)
        if self.post_ffn_norm is not None:
            n = self.post_ffn_norm(n)
        x = x + self.resid_ffn_dropout(n) * self.residual_scaling
        return x, attn_weights, past_key_value


class MPTMuPModel(MPTModel):
    """Wraps :class:`MPTModel` with MuP-specific components."""

    config: MPTMuPConfig

    def __init__(self, config: MPTMuPConfig) -> None:
        self.mup_config = MuPConfig(**config.mup_config)
        self.init_config = ModelInitConfig(
            init_std=config.init_config.get("init_std"),
            emb_init_std=config.init_config.get("emb_init_std"),
            output_mult=config.init_config.get("output_mult"),
        )
        super().__init__(config)
        self.layers = self.blocks  # align with TorchTitan expectations for AC

        if self.mup_config.mup_enabled:
            self.wte = MuPSharedEmbedding(
                config.vocab_size,
                config.d_model,
                padding_idx=config.pad_token_id,
                device=config.init_device,
                scale=self.mup_config.mup_input_alpha,
                use_embedding_norm=config.use_embedding_norm and config.use_peri_norm,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
            )
            if config.init_device != "meta":
                self.param_init_fn(self.wte)

    @property
    def block_class(self) -> type[MPTBlock]:
        return MPTCompletePBlock

    def extract_block_args(self, block_args: dict[str, Any]) -> dict[str, Any]:
        block_args = super().extract_block_args(block_args)
        block_args["depth_multiplier"] = self.mup_config.completep_depth_multiplier
        block_args[
            "depth_alpha_enabled"
        ] = self.mup_config.completep_depth_alpha_enabled
        block_args["depth_alpha_exp"] = self.mup_config.completep_depth_alpha_exp
        block_args["use_peri_norm"] = self.config.use_peri_norm
        return block_args

    def forward(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast:
        outputs = super().forward(*args, **kwargs)
        if self.mup_config.mup_enabled and outputs.last_hidden_state is not None:
            scaling = self.mup_config.mup_output_alpha / max(
                self.mup_config.mup_width_multiplier, 1e-8
            )
            outputs.last_hidden_state = outputs.last_hidden_state * scaling  # type: ignore[assignment]
            if outputs.hidden_states:
                hidden_states = list(outputs.hidden_states)
                hidden_states[-1] = outputs.last_hidden_state
                outputs.hidden_states = tuple(hidden_states)
        return outputs


class MPTMuPForCausalLM(MPTForCausalLM):
    """Adapter around :class:`MPTForCausalLM` that injects MuP behaviour."""

    config: MPTMuPConfig

    def __init__(self, config: MPTMuPConfig) -> None:
        super().__init__(config)
        self.mup_config = MuPConfig(**self.config.mup_config)
        self.init_config = ModelInitConfig(
            init_std=self.config.init_config.get("init_std"),
            emb_init_std=self.config.init_config.get("emb_init_std"),
            output_mult=self.config.init_config.get("output_mult"),
        )
        self.transformer = cast("MPTMuPModel", self.transformer)

    @property
    def backbone_model_class(self) -> type[MPTModel]:
        return MPTMuPModel

    def reset_parameters(self) -> None:
        for module in self.modules():
            self.param_init_fn(module)

    def param_init_fn(self, module: nn.Module) -> None:
        if not self.mup_config.mup_enabled:
            return super().param_init_fn(module)

        base_std = (
            self.init_config.init_std if self.init_config.init_std is not None else 0.02
        )
        emb_std = (
            self.init_config.emb_init_std
            if self.init_config.emb_init_std is not None
            else base_std
        )
        width_mult = max(self.mup_config.mup_width_multiplier, 1e-8)
        mup_std = base_std / math.sqrt(width_mult)

        with torch.no_grad():
            if isinstance(module, MuPSharedEmbedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=emb_std)
                if module.norm is not None and hasattr(module.norm, "reset_parameters"):
                    module.norm.reset_parameters()
            elif isinstance(module, SharedEmbedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=emb_std)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=mup_std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif type(module) in norms.get_all().values():
                torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            else:
                super().param_init_fn(module)
        return None

    def get_optimizer_param_groups(
        self,
        optimizer_config: Mapping[str, Any],
    ) -> tuple[Iterator[Parameter] | list[dict[str, Any]], dict[str, Any]]:
        if not (
            self.mup_config.mup_enabled
            and not self.mup_config.mup_disable_hidden_lr_scaling
        ):
            return self.parameters(), dict(optimizer_config)

        emb_params: list[Parameter] = []
        hidden_ln_params: list[Parameter] = []
        decay_lr_params: list[Parameter] = []
        hidden_bias_params: list[Parameter] = []
        no_decay_params: list[Parameter] = []

        transformer = self.transformer

        emb_params.append(transformer.wte.weight)
        if hasattr(transformer, "wpe") and transformer.wpe is not None:
            emb_params.append(transformer.wpe.weight)
        if self.lm_head is not None:
            emb_params.append(self.lm_head.weight)
            if self.lm_head.bias is not None:
                hidden_bias_params.append(self.lm_head.bias)

        embedding_norm = getattr(transformer.wte, "norm", None)
        if embedding_norm is not None:
            hidden_ln_params.extend(p for p in embedding_norm.parameters())

        hidden_ln_params.extend(p for p in transformer.norm_f.parameters())

        for block in transformer.blocks:
            block = cast("MPTCompletePBlock", block)
            if block.fuse_norm_attn_norm:
                hidden_ln_params.extend(block.norm_attn_norm.norm_1.parameters())  # type: ignore[attr-defined]
                if block.norm_attn_norm.norm_2 is not None:  # type: ignore[attr-defined]
                    hidden_ln_params.extend(block.norm_attn_norm.norm_2.parameters())  # type: ignore[attr-defined]
                attn_module = block.norm_attn_norm.attn  # type: ignore[attr-defined]
            else:
                hidden_ln_params.extend(block.norm_1.parameters())
                if block.norm_2 is not None:
                    hidden_ln_params.extend(block.norm_2.parameters())
                attn_module = block.attn

            if block.post_attn_norm is not None:
                hidden_ln_params.extend(block.post_attn_norm.parameters())
            if block.post_ffn_norm is not None:
                hidden_ln_params.extend(block.post_ffn_norm.parameters())

            if hasattr(attn_module, "q_ln") and attn_module.q_ln is not None:
                hidden_ln_params.extend(attn_module.q_ln.parameters())
            if hasattr(attn_module, "k_ln") and attn_module.k_ln is not None:
                hidden_ln_params.extend(attn_module.k_ln.parameters())

            linear_modules: list[nn.Module] = []
            if hasattr(attn_module, "Wqkv"):
                linear_modules.append(attn_module.Wqkv)
            else:
                for attr in ("Wq", "Wk", "Wv"):
                    if hasattr(attn_module, attr):
                        linear_modules.append(getattr(attn_module, attr))
            if hasattr(attn_module, "out_proj"):
                linear_modules.append(attn_module.out_proj)

            ffn_module = block.ffn
            for attr in ("up_proj", "down_proj", "gate"):
                if hasattr(ffn_module, attr):
                    linear_modules.append(getattr(ffn_module, attr))

            for module in linear_modules:
                weight = getattr(module, "weight", None)
                if isinstance(weight, nn.Parameter):
                    decay_lr_params.append(weight)
                bias = getattr(module, "bias", None)
                if isinstance(bias, nn.Parameter):
                    hidden_bias_params.append(bias)

        emb_params = list(dict.fromkeys(emb_params))
        hidden_ln_params = list(dict.fromkeys(hidden_ln_params))
        decay_lr_params = list(dict.fromkeys(decay_lr_params))
        hidden_bias_params = list(dict.fromkeys(hidden_bias_params))
        no_decay_params = list(dict.fromkeys(no_decay_params))

        assigned = (
            set(emb_params)
            | set(hidden_ln_params)
            | set(decay_lr_params)
            | set(hidden_bias_params)
            | set(no_decay_params)
        )
        unassigned = [
            p for p in self.parameters() if p.requires_grad and p not in assigned
        ]
        if unassigned:
            names = [n for n, p in self.named_parameters() if p in unassigned]
            msg = f"Unassigned parameters found in MuP grouping: {names}"
            raise ValueError(msg)

        base_lr = optimizer_config["lr"]
        weight_decay = optimizer_config.get("weight_decay", 0.0)
        eps = optimizer_config.get("eps", 1e-8)

        width_lr_scaling = 1.0 / max(self.mup_config.mup_width_multiplier, 1e-8)
        depth_lr_scaling = 1.0
        if self.mup_config.completep_depth_alpha_enabled:
            depth_lr_scaling = self.mup_config.completep_depth_multiplier ** (
                self.mup_config.completep_depth_alpha_exp - 1.0
            )

        if self.mup_config.completep_eps_scaling_enabled:
            depth_eps_scaling = self.mup_config.completep_depth_multiplier ** (
                -1.0 * self.mup_config.completep_depth_alpha_exp
            )
            eps = eps * width_lr_scaling * depth_eps_scaling

        param_groups: list[dict[str, Any]] = [
            {"params": emb_params, "weight_decay": weight_decay, "lr": base_lr},
            {
                "params": hidden_ln_params,
                "weight_decay": 0.0,
                "lr": base_lr * depth_lr_scaling,
            },
            {
                "params": decay_lr_params,
                "weight_decay": weight_decay / width_lr_scaling,
                "lr": base_lr * width_lr_scaling * depth_lr_scaling,
            },
            {
                "params": hidden_bias_params,
                "weight_decay": 0.0,
                "lr": base_lr * depth_lr_scaling,
            },
            {"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr},
        ]
        param_groups = [group for group in param_groups if group["params"]]

        updated_config = dict(optimizer_config)
        updated_config["eps"] = eps

        return param_groups, updated_config

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.ByteTensor | None = None,
        sequence_id: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            labels=labels,
            **kwargs,
        )


class Transformer(nn.Module):
    """TorchTitan wrapper around :class:`MPTMuPForCausalLM`."""

    def __init__(self, model_args: TransformerModelArgs) -> None:
        super().__init__()
        self.model_args = model_args
        config = model_args.to_config()
        self.model = MPTMuPForCausalLM(config)

    @property
    def transformer(self) -> MPTMuPModel:
        return self.model.transformer

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        self.model.reset_parameters()
        self.model.transformer.attn_bias = None
        self.model.transformer._attn_bias_initialized = False
        if buffer_device is not None:
            for name, buffer in self.model.transformer.named_buffers():
                if buffer.device.type == "meta":
                    setattr(
                        self.model.transformer,
                        name,
                        buffer.to(buffer_device),
                    )

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: MutableMapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del input_batch
        outputs = self.model(input_ids=tokens)
        if outputs.logits is None:
            msg = "MPTMuPForCausalLM did not return logits."
            raise RuntimeError(msg)
        return outputs.logits

    def _call_get_param_groups(
        self,
        *,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> tuple[Iterator[Parameter] | list[dict[str, Any]], dict[str, Any]]:
        return self.model.get_optimizer_param_groups(
            {"lr": lr, "eps": eps, "weight_decay": weight_decay}
        )

    def build_mup_optimizer_overrides(
        self,
        *,
        lr: float,
        eps: float,
        weight_decay: float,
    ) -> MuPOptimizerOverride | None:
        param_groups, updated_config = self._call_get_param_groups(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        base_config = {"lr": lr, "eps": eps, "weight_decay": weight_decay}
        config_updates = {
            key: value
            for key, value in updated_config.items()
            if key in base_config and base_config[key] != value
        }

        if (
            isinstance(param_groups, list)
            and param_groups
            and isinstance(param_groups[0], dict)
        ):
            return MuPOptimizerOverride(
                param_groups=param_groups,
                config_updates=config_updates,
            )
        if config_updates:
            return MuPOptimizerOverride(
                param_groups=None, config_updates=config_updates
            )
        return None

    def get_optimizer_param_groups(
        self,
        optimizer_config: Mapping[str, Any],
    ) -> tuple[Iterator[Parameter] | list[dict[str, Any]], dict[str, Any]]:
        return self._call_get_param_groups(
            lr=optimizer_config["lr"],
            eps=optimizer_config.get("eps", 1e-8),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )


__all__ = [
    "MPTCompletePBlock",
    "MPTMuPForCausalLM",
    "MPTMuPModel",
    "MuPOptimizerOverride",
    "MuPSharedEmbedding",
    "Transformer",
]
