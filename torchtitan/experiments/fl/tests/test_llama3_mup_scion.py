# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Scion-specific MuP optimizer behavior."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from torchtitan.experiments.fl.models.llama3_mup.model.mup_args import (
    TransformerModelArgs,
)
from torchtitan.experiments.fl.models.llama3_mup.model.mup_model import (
    Attention,
    FeedForward,
    Transformer,
)
from torchtitan.experiments.fl.models.llama3_mup.train_configs import (
    llama3_mup_configs,
)


def _build_model(use_scion: bool, *, use_disco: bool = False) -> Transformer:
    args = replace(llama3_mup_configs["16M"], use_scion=use_scion, use_disco=use_disco)
    return Transformer(args)


def _build_small_model(use_scion: bool, *, use_disco: bool = False, **overrides: object) -> Transformer:
    base = llama3_mup_configs["16M"]
    small_args = replace(
        base,
        dim=16,
        n_layers=1,
        n_heads=2,
        vocab_size=32,
        multiple_of=16,
        tie_word_embeddings=False,
        use_scion=use_scion,
        use_disco=use_disco,
        **overrides,
    )
    return Transformer(small_args)


def _build_param_groups_from_args(model_args: TransformerModelArgs) -> tuple[Transformer, list[dict]]:
    model = Transformer(model_args)
    overrides = model.build_mup_optimizer_overrides(
        lr=1e-3,
        eps=1e-8,
        weight_decay=0.0,
    )
    assert overrides is not None and overrides.param_groups is not None
    return model, overrides.param_groups


def _build_param_groups(use_scion: bool, *, use_disco: bool = False) -> tuple[Transformer, list[dict]]:
    args = replace(llama3_mup_configs["16M"], use_scion=use_scion, use_disco=use_disco)
    return _build_param_groups_from_args(args)



def _find_group(param_groups: list[dict], parameter) -> dict:
    for group in param_groups:
        for candidate in group["params"]:
            if candidate is parameter:
                return group
    raise AssertionError("Parameter not present in optimizer groups.")


def test_use_disco_sets_sign_norm_for_embeddings() -> None:
    model, param_groups = _build_param_groups(use_scion=True, use_disco=True)
    emb_group = _find_group(param_groups, model.tok_embeddings.weight)
    assert emb_group["norm"] == "embed_linear"
    assert emb_group["norm_kwargs"].get("backend") == "identity"


def test_use_disco_sets_spectral_norm_for_hidden_weights() -> None:
    model, param_groups = _build_param_groups(use_scion=True, use_disco=True)
    wq_weight = model.layers["0"].attention.wq.weight
    hidden_group = _find_group(param_groups, wq_weight)
    assert hidden_group["norm"] == "spectral"
    assert hidden_group["norm_kwargs"].get("backend") == "newtonschulz5"


def test_plain_scion_sets_spectral_norm_for_hidden_weights() -> None:
    model, param_groups = _build_param_groups(use_scion=True, use_disco=False)
    wq_weight = model.layers["0"].attention.wq.weight
    hidden_group = _find_group(param_groups, wq_weight)
    assert hidden_group["norm"] == "spectral"
    assert hidden_group["norm_kwargs"].get("backend") == "newtonschulz5"
    assert hidden_group["norm_kwargs"].get("normalized") is True


def test_plain_scion_sets_sign_norm_for_embeddings() -> None:
    model, param_groups = _build_param_groups(use_scion=True, use_disco=False)
    emb_group = _find_group(param_groups, model.tok_embeddings.weight)
    assert emb_group["norm"] == "sign"
    assert emb_group["norm_kwargs"].get("normalized") is True


def test_scion_assigns_hidden_and_output_scales() -> None:
    model, param_groups = _build_param_groups(use_scion=True, use_disco=False)
    hidden_group = _find_group(param_groups, model.layers["0"].attention.wq.weight)
    emb_group = _find_group(param_groups, model.tok_embeddings.weight)
    assert hidden_group["scale"] == model.model_args.scion_hidden_scale
    assert emb_group["scale"] == model.model_args.scion_output_scale


def test_scion_scale_overrides_respected_via_optimizer_config() -> None:
    model = Transformer(replace(llama3_mup_configs["16M"], use_scion=True, use_disco=False))
    overrides = model.build_mup_optimizer_overrides(
        lr=0.01,
        eps=1e-8,
        weight_decay=0.0,
        scion_hidden_scale=12.5,
        scion_output_scale=4321.0,
        scion_hidden_norm="sign",
        scion_output_norm="spectral",
        scion_output_norm_kwargs={"backend": "identity", "backend_steps": 0},
    )
    assert overrides is not None
    assert overrides.param_groups is not None
    hidden_group = _find_group(overrides.param_groups, model.layers["0"].attention.wq.weight)
    emb_group = _find_group(overrides.param_groups, model.tok_embeddings.weight)
    assert hidden_group["scale"] == pytest.approx(12.5)
    assert emb_group["scale"] == pytest.approx(4321.0)
    assert hidden_group["norm"] == "sign"
    assert hidden_group["norm_kwargs"].get("normalized") is True
    assert emb_group["norm"] == "spectral"
    assert emb_group["norm_kwargs"].get("backend") == "identity"


def test_scion_preserves_width_multiplier_value() -> None:
    base_args = llama3_mup_configs["16M"]
    overridden = replace(
        base_args,
        use_scion=True,
        mup_config=dict(base_args.mup_config, mup_width_multiplier=6.0),
    )
    assert isinstance(overridden, TransformerModelArgs)
    assert overridden.mup_config_obj.mup_width_multiplier == 6.0


def test_scion_lr_scaling_ignores_width_multiplier() -> None:
    base_args = llama3_mup_configs["16M"]
    args = replace(
        base_args,
        use_scion=True,
        mup_config=dict(base_args.mup_config, mup_width_multiplier=6.0),
    )
    model = Transformer(args)
    width_scale, _ = model._compute_lr_scaling()
    assert width_scale == 1.0


def test_layernorm_impl_overrides_flag() -> None:
    base_args = llama3_mup_configs["16M"]
    args = replace(base_args, use_torch_layernorm=True, layernorm_impl="rms")
    assert not args.use_torch_layernorm


def test_qk_layernorm_impl_independent() -> None:
    base_args = llama3_mup_configs["16M"]
    args = replace(
        base_args,
        use_torch_layernorm=False,
        qk_layernorm_impl="torch",
    )
    assert args.use_torch_layernorm is False
    assert args.use_torch_qk_layernorm is True


def test_qk_layernorm_inherits_general_when_unset() -> None:
    base_args = replace(
        llama3_mup_configs["16M"],
        use_torch_layernorm=False,
        use_torch_qk_layernorm=None,
    )
    base_args.__post_init__()
    assert base_args.use_torch_qk_layernorm is False


def test_attention_value_norm_flag_creates_layer() -> None:
    base_args = replace(llama3_mup_configs["16M"], use_attention_value_norm=True)
    attn = Attention(base_args)
    assert not isinstance(attn.v_norm, nn.Identity)


def test_attention_output_norm_flag_creates_layer() -> None:
    base_args = replace(llama3_mup_configs["16M"], use_attention_output_norm=True)
    attn = Attention(base_args)
    assert not isinstance(attn.o_norm, nn.Identity)


def test_mlp_mid_norm_flag_creates_layer() -> None:
    base_args = replace(llama3_mup_configs["16M"], use_mlp_mid_norm=True)
    ffn = FeedForward(base_args)
    assert not isinstance(ffn.mid_norm, nn.Identity)



def _bucketize_and_get(model: Transformer, param_name: str) -> str:
    params = model._iter_trainable_params()
    model._bucketize_parameters(params)
    return model._last_bucket_assignments[param_name]


def test_attention_value_norm_bucketed_with_hidden_ln() -> None:
    args = replace(llama3_mup_configs["16M"], use_attention_value_norm=True)
    model = Transformer(args)
    bucket = _bucketize_and_get(model, "layers.0.attention.v_norm.weight")
    assert bucket == "hidden_ln"


def test_attention_output_norm_bucketed_with_hidden_ln() -> None:
    args = replace(llama3_mup_configs["16M"], use_attention_output_norm=True)
    model = Transformer(args)
    bucket = _bucketize_and_get(model, "layers.0.attention.o_norm.weight")
    assert bucket == "hidden_ln"


def test_mlp_mid_norm_bucketed_with_hidden_ln() -> None:
    args = replace(llama3_mup_configs["16M"], use_mlp_mid_norm=True)
    model = Transformer(args)
    bucket = _bucketize_and_get(model, "layers.0.feed_forward.mid_norm.weight")
    assert bucket == "hidden_ln"


def test_default_disco_init_types() -> None:
    model = Transformer(replace(llama3_mup_configs["16M"], use_scion=True, use_disco=True))
    assert model._hidden_init_type == "disco_normal"
    assert model._embed_init_type == "disco_normal_input"
    assert model._output_init_type == "disco_normal_output"


def test_default_non_scion_init_types() -> None:
    model = Transformer(replace(llama3_mup_configs["16M"], use_scion=False))
    assert model._hidden_init_type == "normal"
    assert model._embed_init_type == "normal"
    assert model._output_init_type == "normal"


def test_standard_scion_without_disco_uses_normal_init_types() -> None:
    model = Transformer(replace(llama3_mup_configs["16M"], use_scion=True, use_disco=False))
    assert model._hidden_init_type == "normal"
    assert model._embed_init_type == "normal"
    assert model._output_init_type == "normal"


def test_unembed_bucket_created_when_weights_untied() -> None:
    args = replace(llama3_mup_configs["16M"], tie_word_embeddings=False)
    model = Transformer(args)
    params = model._iter_trainable_params()
    model._bucketize_parameters(params)
    assert model._last_bucket_assignments["tok_embeddings.weight"] == "emb"
    assert model._last_bucket_assignments["output.weight"] == "unembed"


def test_untied_embeddings_use_distinct_param_groups() -> None:
    args = replace(llama3_mup_configs["16M"], tie_word_embeddings=False)
    model, param_groups = _build_param_groups_from_args(args)
    embed_group = _find_group(param_groups, model.tok_embeddings.weight)
    unembed_group = _find_group(param_groups, model.output.weight)
    assert embed_group is not unembed_group


def test_disco_embedding_init_matches_expected_norm() -> None:
    model = _build_small_model(use_scion=True, use_disco=True)
    emb = model.tok_embeddings.weight
    norms = torch.linalg.vector_norm(emb, dim=1)
    expected = math.sqrt(model.model_args.dim)
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-6)


def test_disco_output_init_matches_expected_norm() -> None:
    model = _build_small_model(use_scion=True, use_disco=True)
    out = model.output.weight
    norms = torch.linalg.vector_norm(out, dim=1)
    expected = 1.0 / math.sqrt(model.model_args.dim)
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-6)


def test_disco_hidden_inits_are_unit_norm() -> None:
    model = _build_small_model(use_scion=True, use_disco=True)
    wq = model.layers["0"].attention.wq.weight
    norms = torch.linalg.vector_norm(wq, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_custom_hidden_init_type_applied() -> None:
    base = llama3_mup_configs["16M"]
    init_cfg = dict(base.init_config, hidden_init="disco_normal_input")
    args = replace(base, use_scion=True, use_disco=True, init_config=init_cfg)
    model = Transformer(args)
    wq = model.layers["0"].attention.wq.weight
    norms = torch.linalg.vector_norm(wq, dim=1)
    expected = math.sqrt(wq.shape[1])
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-6)


def test_scion_skips_mup_input_output_alpha_scaling() -> None:
    base = llama3_mup_configs["16M"]
    mup_override = dict(base.mup_config)
    mup_override.update({"mup_input_alpha": 0.0, "mup_output_alpha": 0.0})
    scion_model = Transformer(
        replace(
            base,
            use_scion=True,
            tie_word_embeddings=False,
            mup_config=dict(mup_override),
        )
    )
    mup_model = Transformer(
        replace(
            base,
            use_scion=False,
            tie_word_embeddings=False,
            mup_config=dict(mup_override),
        )
    )
    tokens = torch.randint(0, base.vocab_size, (1, 4))
    with torch.no_grad():
        scion_logits = scion_model(tokens)
        mup_logits = mup_model(tokens)
    assert torch.linalg.vector_norm(scion_logits) > 0
    assert torch.count_nonzero(mup_logits) == 0


def test_disco_assigns_sqrt_norms_when_untied() -> None:
    args = replace(
        llama3_mup_configs["16M"],
        use_scion=True,
        use_disco=True,
        tie_word_embeddings=False,
    )
    model, param_groups = _build_param_groups_from_args(args)
    emb_group = _find_group(param_groups, model.tok_embeddings.weight)
    unembed_group = _find_group(param_groups, model.output.weight)
    assert emb_group["norm"] == "embed_sqrt"
    assert unembed_group["norm"] == "unembed_sqrt"


def test_trunc_normal_init_respects_cutoff() -> None:
    base = llama3_mup_configs["16M"]
    init_cfg = dict(base.init_config)
    init_cfg.update({"hidden_init": "trunc_normal", "trunc_normal_cutoff": 0.5, "init_std": 0.01})
    args = replace(base, use_scion=False, init_config=init_cfg)
    model = Transformer(args)
    wq = model.layers["0"].attention.wq.weight
    assert torch.max(torch.abs(wq)) <= 0.0050001
