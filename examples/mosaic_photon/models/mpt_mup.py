"""muP-enabled MPT model definitions for the Mosaic Photon example package."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Iterable, MutableMapping, cast

import torch
from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from llmfoundry import ComposerMPTCausalLM, MPTForCausalLM
    from llmfoundry.layers_registry import ffns_with_megablocks
    from llmfoundry.models.layers.blocks import MPTBlock
    from llmfoundry.models.layers.custom_embedding import SharedEmbedding
    from llmfoundry.models.layers.layer_builders import build_norm
    from llmfoundry.models.mpt import MPTConfig, MPTModel
    from llmfoundry.models.utils.mpt_param_count import megablocks_n_active_params
except ImportError:  # pragma: no cover - keep module importable without llmfoundry
    ComposerMPTCausalLM = None  # type: ignore[assignment]
    MPTForCausalLM = None  # type: ignore[assignment]
    ffns_with_megablocks = set()  # type: ignore[assignment]
    MPTBlock = nn.Module  # type: ignore[assignment]
    SharedEmbedding = nn.Module  # type: ignore[assignment]
    MPTConfig = object  # type: ignore[assignment]
    MPTModel = nn.Module  # type: ignore[assignment]

    def build_norm(*args: Any, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        raise RuntimeError(
            "llm-foundry must be installed to build MPT muP models."
        )

    def megablocks_n_active_params(model: nn.Module) -> int:  # type: ignore[override]
        raise RuntimeError(
            "llm-foundry must be installed to compute active parameter counts."
        )

try:  # pragma: no cover - optional dependency
    from llmfoundry.models.ffns.sigma_moe import SigmaMoE
except ImportError:  # pragma: no cover - optional dependency
    class SigmaMoE(nn.Module):  # fallback stub used only for isinstance checks
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

            raise RuntimeError(
                "llm-foundry is required for SigmaMoE support in the muP example."
            )


HAS_MPT_MUP_SUPPORT = ComposerMPTCausalLM is not None and isinstance(MPTConfig, type)


@dataclass
class ModelInitConfig:
    """Initialization configuration used by the muP example."""

    init_std: float | None = None
    emb_init_std: float | None = None


@dataclass
class MuPConfig:
    """Minimal muP configuration used by the example implementation."""

    mup_enabled: bool = True
    mup_disable_attention_scaling: bool = True
    mup_disable_hidden_lr_scaling: bool = False
    mup_width_multiplier: float = 1.0
    mup_input_alpha: float = 1.0
    mup_output_alpha: float = 1.0
    completep_depth_alpha_enabled: bool = False
    completep_depth_multiplier: float = 1.0
    completep_depth_alpha_exp: float = 1.0
    completep_eps_scaling_enabled: bool = True
    mup_scale_expert_sel: bool = False


class _SharedEmbeddingFallback(nn.Module):
    """Simple embedding layer used when llmfoundry is unavailable."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device} if device is not None else {}
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            **factory_kwargs,
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.embedding(input_ids)


_SharedEmbeddingBase = (
    SharedEmbedding
    if isinstance(SharedEmbedding, type) and issubclass(SharedEmbedding, nn.Module)
    else _SharedEmbeddingFallback
)


class MuPSharedEmbedding(_SharedEmbeddingBase):
    """Shared embedding that optionally applies a post-embedding norm."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int | None = None,
        device: str | torch.device | None = None,
        scale: float = 1.0,
        use_embedding_norm: bool = False,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            device=device,
        )
        self.scale = scale
        self.post_norm: nn.Module | None = None
        if use_embedding_norm:
            try:
                self.post_norm = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=embedding_dim,
                    eps=norm_eps,
                    device=device,
                )
            except Exception:  # pragma: no cover - norm builder unavailable
                logger.warning(
                    "Falling back to nn.LayerNorm for embedding norm.",
                    exc_info=True,
                )
                self.post_norm = nn.LayerNorm(embedding_dim, eps=norm_eps, device=device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D401
        outputs = super().forward(input_ids)
        outputs = outputs * self.scale
        if self.post_norm is not None:
            outputs = self.post_norm(outputs)
        return outputs


def dtensor_safe_check_numel(param: nn.Parameter | torch.Tensor) -> int:
    """Return the number of elements in a parameter, ignoring DTensor wrappers."""

    return int(param.numel())


def sigma_moe_n_active_params(model: "ComposerMPTMuPCausalLM") -> int:
    """Calculate the number of active parameters for SigmaMoE layers."""

    n_active_params = 0
    config = cast("MPTMuPConfig", model.config)
    n_active_experts = int(config.ffn_config.get("ff_n_active_experts", 0))
    n_total_experts = int(config.ffn_config.get("ff_n_experts", 1))
    if n_active_experts <= 0:
        raise ValueError("ff_n_active_experts must be positive for SigmaMoE configs")

    for module in model.modules():
        if isinstance(module, SigmaMoE):
            up_proj_n_params = dtensor_safe_check_numel(module.up_proj)
            down_proj_n_params = dtensor_safe_check_numel(module.down_proj)
            n_active_params += int(
                (up_proj_n_params + down_proj_n_params)
                / max(1, n_total_experts)
                * n_active_experts
            )
            if hasattr(module, "expert_sel"):
                expert_sel = getattr(module, "expert_sel")
                if isinstance(expert_sel, torch.Tensor):
                    n_active_params += int(expert_sel.numel())
        else:
            for param in module.parameters(recurse=False):
                n_active_params += int(param.numel())

    return n_active_params


def convert_sigma_moe_config_to_dense(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a SigmaMoE feed-forward config to a dense equivalent."""

    config = copy.deepcopy(config)
    ffn_cfg = dict(config.get("ffn_config", {}))
    ff_n_experts = int(ffn_cfg.get("ff_n_experts", 1))
    ff_expert_size = int(ffn_cfg.get("ff_expert_size", config["ffn_hidden_size"]))
    config["ffn_config"] = {
        "ffn_act_fn": {"name": "silu"},
        "ffn_type": "mptmlp",
        "fc_type": {"name": "torch"},
    }
    hidden = config["ffn_hidden_size"]
    dense_hidden = ff_n_experts * ff_expert_size
    if dense_hidden != hidden:
        logger.warning(
            "Adjusting ffn_hidden_size from %s to %s to match dense equivalent.",
            hidden,
            dense_hidden,
        )
        config["ffn_hidden_size"] = dense_hidden
    return config


if HAS_MPT_MUP_SUPPORT:

    class MPTMuPConfig(MPTConfig):
        """Configuration wrapper that augments :class:`MPTConfig` with muP fields."""

        def __init__(
            self,
            *args: Any,
            n_non_expert_layers: int = 0,
            mup_config: dict[str, Any] | None = None,
            use_peri_norm: bool = True,
            use_embedding_norm: bool = True,
            **kwargs: Any,
        ) -> None:
            self.n_non_expert_layers = int(n_non_expert_layers)
            self.use_peri_norm = bool(use_peri_norm)
            self.use_embedding_norm = bool(use_embedding_norm)
            obj_mup_config = MuPConfig(**(mup_config or {}))
            self.mup_config = asdict(obj_mup_config)
            super().__init__(*args, **kwargs)

            if self.n_non_expert_layers < 0 or self.n_non_expert_layers > self.n_layers:
                raise ValueError(
                    "n_non_expert_layers must be between 0 and n_layers inclusive."
                )

            if not hasattr(self, "init_config") or self.init_config is None:
                self.init_config = asdict(ModelInitConfig())  # type: ignore[assignment]

            if (
                self.mup_config["mup_enabled"]
                and not self.mup_config["mup_disable_attention_scaling"]
            ):
                head_dim = self.d_model // self.n_heads
                attn_cfg = dict(self.attn_config)
                attn_cfg.setdefault("softmax_scale", 1.0 / float(head_dim))
                self.attn_config = attn_cfg

    class MPTCompletePBlock(MPTBlock):
        """Specialised block that supports CompleteP-style residual scaling."""

        def __init__(  # noqa: PLR0913
            self,
            depth_multiplier: float = 1.0,
            depth_alpha_exp: float = 1.0,
            norm_type: str = "layernorm",
            norm_eps: float = 1e-5,
            *,
            use_peri_norm: bool = False,
            depth_alpha_enabled: bool = False,
            device: str | torch.device | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                norm_type=norm_type,
                norm_eps=norm_eps,
                device=device,
                **kwargs,
            )
            self.residual_scaling = (
                1.0 / (depth_multiplier**depth_alpha_exp)
                if depth_alpha_enabled
                else 1.0
            )
            self.post_attn_norm: nn.Module | None = None
            self.post_ffn_norm: nn.Module | None = None
            if use_peri_norm:
                self.post_attn_norm = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=kwargs["d_model"],
                    eps=norm_eps,
                    device=device,
                )
                self.post_ffn_norm = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=kwargs["d_model"],
                    eps=norm_eps,
                    device=device,
                )

        def forward(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            output = super().forward(*args, **kwargs)
            if isinstance(output, tuple) and len(output) >= 1:
                residual = output[0] * self.residual_scaling
                if self.post_attn_norm is not None:
                    residual = self.post_attn_norm(residual)
                if self.post_ffn_norm is not None:
                    residual = self.post_ffn_norm(residual)
                output = (residual,) + output[1:]
            return output

    class MPTMuPModel(MPTModel):
        """MPT model variant that plugs in muP-aware components."""

        config_class = MPTMuPConfig

        def __init__(self, config: MPTMuPConfig) -> None:
            self.mup_config = MuPConfig(**config.mup_config)
            init_cfg = config.init_config or {}
            self.init_config = ModelInitConfig(**init_cfg)
            super().__init__(config)
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
        def block_class(self) -> type["MPTCompletePBlock"]:
            return MPTCompletePBlock

        def construct_blocks(self, config: MPTConfig) -> nn.ModuleList:  # type: ignore[override]
            block_args = self.extract_block_args(config.to_dict())
            if config.block_overrides is not None:
                block_args_list = self._get_override_block_args_list(config, block_args)
            else:
                block_args_list = [copy.deepcopy(block_args) for _ in range(config.n_layers)]

            blocks = []
            for idx, block_cfg in enumerate(block_args_list):
                if idx < self.config.n_non_expert_layers:
                    block_cfg = convert_sigma_moe_config_to_dense(block_cfg)
                blocks.append(
                    self.block_class(
                        device=config.init_device,
                        **block_cfg,
                    )
                )
            return nn.ModuleList(blocks)

        def extract_block_args(self, block_args: dict[str, Any]) -> dict[str, Any]:  # noqa: D401
            block_args = super().extract_block_args(block_args)
            block_args["depth_multiplier"] = self.mup_config.completep_depth_multiplier
            block_args["depth_alpha_enabled"] = (
                self.mup_config.completep_depth_alpha_enabled
            )
            block_args["depth_alpha_exp"] = self.mup_config.completep_depth_alpha_exp
            block_args["use_peri_norm"] = self.config.use_peri_norm
            if block_args.get("ffn_config", {}).get("ffn_type") == "sigma_moe":
                block_args["ffn_config"]["mup_config"] = self.config.mup_config
                block_args["ffn_config"]["d_model"] = self.config.d_model
                init_std = (
                    self.init_config.init_std if self.init_config.init_std is not None else 0.02
                )
                block_args["ffn_config"]["init_std"] = init_std
            return block_args

        def forward(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            outputs = super().forward(*args, **kwargs)
            if self.mup_config.mup_enabled and hasattr(outputs, "last_hidden_state"):
                outputs.last_hidden_state = outputs.last_hidden_state * (
                    self.mup_config.mup_output_alpha / self.mup_config.mup_width_multiplier
                )
            return outputs

    class MPTMuPForCausalLM(MPTForCausalLM):
        """Causal LM head that resets parameters with muP-aware init."""

        config_class = MPTMuPConfig

        def __init__(self, config: MPTMuPConfig) -> None:
            super().__init__(config)
            self.config = cast("MPTMuPConfig", self.config)
            self.mup_config = MuPConfig(**self.config.mup_config)
            self.init_config = ModelInitConfig(**(self.config.init_config or {}))
            self.transformer = cast("MPTMuPModel", self.transformer)
            if config.init_device != "meta":
                self.reset_parameters()

        @property
        def backbone_model_class(self) -> type[MPTModel]:  # type: ignore[override]
            return MPTMuPModel

        def reset_parameters(self) -> None:  # noqa: D401
            for module in self.modules():
                self.param_init_fn(module)

        def param_init_fn(self, module: nn.Module) -> None:  # noqa: D401
            if not self.mup_config.mup_enabled:
                return super().param_init_fn(module)

            emb_std = self.init_config.emb_init_std or self.init_config.init_std or 0.02
            base_std = self.init_config.init_std or 0.02
            width_mult = self.mup_config.mup_width_multiplier
            mup_std = base_std / (width_mult**0.5)

            with torch.no_grad():
                if module is self.transformer.wte or module is getattr(self, "lm_head", None):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=emb_std)
                    if hasattr(module, "bias") and module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, SigmaMoE):
                    module.reset_parameters()
                elif isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=mup_std)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

        def get_optimizer_param_groups(
            self,
            optimizer_config: dict[str, Any],
        ) -> tuple[Iterable[torch.Tensor] | Iterable[dict[str, Any]], dict[str, Any]]:
            if not (
                self.mup_config.mup_enabled
                and not self.mup_config.mup_disable_hidden_lr_scaling
            ):
                return super().get_optimizer_param_groups(optimizer_config)

            emb_params: list[nn.Parameter | torch.Tensor] = []
            hidden_ln_params: list[nn.Parameter] = []
            decay_lr_params: list[nn.Parameter] = []
            hidden_bias_params: list[nn.Parameter] = []
            no_decay_params: list[nn.Parameter] = []

            emb_params.append(self.transformer.wte.weight)
            if hasattr(self.transformer, "wpe") and self.transformer.wpe is not None:
                emb_params.append(self.transformer.wpe.weight)
            if hasattr(self.transformer.wte, "norm") and self.transformer.wte.norm is not None:
                no_decay_params.extend(self.transformer.wte.norm.parameters())
            no_decay_params.extend(self.transformer.norm_f.parameters())

            for block in self.transformer.blocks:
                block = cast("MPTCompletePBlock", block)
                for attr in ("norm_1", "norm_2", "post_attn_norm", "post_ffn_norm"):
                    module = getattr(block, attr, None)
                    if module is not None:
                        hidden_ln_params.extend(module.parameters())

                attn = block.attn
                for linear_name in ("Wqkv", "out_proj"):
                    linear = getattr(attn, linear_name, None)
                    if linear is not None and hasattr(linear, "weight"):
                        decay_lr_params.append(linear.weight)
                        if getattr(linear, "bias", None) is not None:
                            hidden_bias_params.append(linear.bias)  # type: ignore[arg-type]
                for ln_name in ("q_ln", "k_ln"):
                    ln = getattr(attn, ln_name, None)
                    if ln is not None:
                        hidden_ln_params.extend(ln.parameters())

                ffn = block.ffn
                for linear_name in ("up_proj", "down_proj"):
                    linear = getattr(ffn, linear_name, None)
                    if linear is not None and hasattr(linear, "weight"):
                        decay_lr_params.append(linear.weight)
                        if getattr(linear, "bias", None) is not None:
                            hidden_bias_params.append(linear.bias)  # type: ignore[arg-type]
                if isinstance(ffn, SigmaMoE) and hasattr(ffn, "expert_sel"):
                    if ffn.config.mup_config.get("mup_scale_expert_sel", False):
                        decay_lr_params.append(ffn.expert_sel)
                    else:
                        emb_params.append(ffn.expert_sel)

            def _unique(params: list[nn.Parameter | torch.Tensor]) -> list[nn.Parameter | torch.Tensor]:
                seen: dict[int, nn.Parameter | torch.Tensor] = {}
                for param in params:
                    seen[id(param)] = param
                return list(seen.values())

            emb_params = _unique(emb_params)
            hidden_ln_params = _unique(hidden_ln_params)  # type: ignore[assignment]
            decay_lr_params = _unique(decay_lr_params)  # type: ignore[assignment]
            hidden_bias_params = _unique(hidden_bias_params)  # type: ignore[assignment]
            no_decay_params = _unique(no_decay_params)  # type: ignore[assignment]

            assigned = (
                set(id(p) for p in emb_params)
                | set(id(p) for p in hidden_ln_params)
                | set(id(p) for p in decay_lr_params)
                | set(id(p) for p in hidden_bias_params)
                | set(id(p) for p in no_decay_params)
            )
            unassigned = [p for p in self.parameters() if id(p) not in assigned]
            if unassigned:
                names = []
                for name, param in self.named_parameters():
                    if id(param) in {id(x) for x in unassigned}:
                        names.append(name)
                raise ValueError(
                    f"Unassigned parameters found in muP grouping: {names}."
                )

            base_lr = optimizer_config["lr"]
            weight_decay = optimizer_config.get("weight_decay", 0.0)
            width_lr_scaling = 1.0 / self.mup_config.mup_width_multiplier
            depth_lr_scaling = (
                self.mup_config.completep_depth_multiplier
                ** (self.mup_config.completep_depth_alpha_exp - 1.0)
                if self.mup_config.completep_depth_alpha_enabled
                else 1.0
            )

            if self.mup_config.completep_eps_scaling_enabled:
                depth_eps_scaling = (
                    self.mup_config.completep_depth_multiplier
                    ** (-1.0 * self.mup_config.completep_depth_alpha_exp)
                )
                optimizer_config = dict(optimizer_config)
                optimizer_config["eps"] = (
                    optimizer_config.get("eps", 1e-8)
                    * width_lr_scaling
                    * depth_eps_scaling
                )

            param_groups = [
                {
                    "params": emb_params,
                    "weight_decay": weight_decay,
                    "lr": base_lr,
                },
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
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": base_lr,
                },
            ]

            param_groups = [group for group in param_groups if group["params"]]
            return param_groups, optimizer_config

        def forward(self, batch: MutableMapping[str, torch.Tensor]):  # type: ignore[override]
            return self.model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                sequence_id=batch.get("sequence_id"),
                inputs_embeds=batch.get("inputs_embeds"),
                position_ids=batch.get("position_ids"),
            )

    class ComposerMPTMuPCausalLM(ComposerMPTCausalLM):
        """Composer wrapper that exposes the muP MPT backbone."""

        @property
        def model_class(self) -> type[MPTMuPForCausalLM]:  # type: ignore[override]
            return MPTMuPForCausalLM

        @property
        def config_class(self) -> type[MPTMuPConfig]:  # type: ignore[override]
            return MPTMuPConfig

else:

    class MPTMuPConfig:  # type: ignore[redeclaration]
        """Placeholder config when llmfoundry is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "llm-foundry must be installed to use the MPT muP example."
            )

    class MPTMuPModel(nn.Module):  # type: ignore[redeclaration]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "llm-foundry must be installed to use the MPT muP example."
            )

    class MPTMuPForCausalLM(nn.Module):  # type: ignore[redeclaration]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "llm-foundry must be installed to use the MPT muP example."
            )

    class ComposerMPTMuPCausalLM(nn.Module):  # type: ignore[redeclaration]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "llm-foundry must be installed to use the MPT muP example."
            )


@dataclass
class MPTMuPModelArgs(BaseModelArgs):
    """TorchTitan-compatible wrapper around :class:`MPTMuPConfig`."""

    config_name: str = "debug"
    overrides: dict[str, Any] = field(default_factory=dict)

    def update_from_config(self, job_config: JobConfig, **kwargs: Any) -> None:  # noqa: D401
        self.overrides = dict(self.overrides)
        self.overrides["max_seq_len"] = job_config.training.seq_len

    def get_nparams_and_flops(
        self,
        model: nn.Module,
        seq_len: int,
    ) -> tuple[int, float]:
        nparams = sum(p.numel() for p in model.parameters())
        approx_flops = float(nparams) * float(seq_len) * 6.0
        return nparams, approx_flops

    def build_config(self) -> MPTMuPConfig:
        if self.config_name not in MPT_MUP_CONFIGS:
            raise ValueError(
                f"Unknown MPT muP config '{self.config_name}'. Available: {list(MPT_MUP_CONFIGS)}"
            )
        base_cfg = copy.deepcopy(MPT_MUP_CONFIGS[self.config_name])
        base_cfg.update(copy.deepcopy(self.overrides))
        return MPTMuPConfig(**base_cfg)


class TitanComposerMPTMuP(nn.Module, ModelProtocol):
    """Adapter that exposes ``ComposerMPTMuPCausalLM`` through TorchTitan's protocol."""

    def __init__(self, model_args: MPTMuPModelArgs) -> None:
        nn.Module.__init__(self)
        if ComposerMPTCausalLM is None:
            raise RuntimeError(
                "llm-foundry must be installed to instantiate the MPT muP model."
            )
        self.model_args = copy.deepcopy(model_args)
        self.config = self.model_args.build_config()
        self.model = ComposerMPTMuPCausalLM(self.config)

    def forward(self, *args: Any, **kwargs: Any):  # noqa: D401
        return self.model(*args, **kwargs)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:  # noqa: D401
        if buffer_device is not None:
            self.to(buffer_device)
        if hasattr(self.model, "reset_parameters"):
            self.model.reset_parameters()

    def get_optimizer_param_groups(  # pragma: no cover - passthrough helper
        self,
        optimizer_config: dict[str, Any],
    ) -> tuple[Iterable[torch.Tensor] | Iterable[dict[str, Any]], dict[str, Any]]:
        if hasattr(self.model, "get_optimizer_param_groups"):
            return self.model.get_optimizer_param_groups(optimizer_config)
        raise AttributeError("Underlying model does not support param group extraction")

    @cached_property
    def n_active_params(self) -> int:  # pragma: no cover - informational helper
        if hasattr(self.model, "n_active_params"):
            return int(self.model.n_active_params)  # type: ignore[return-value]
        if self.config.ffn_config.get("ffn_type") in ffns_with_megablocks:
            return megablocks_n_active_params(self.model)
        return sum(p.numel() for p in self.model.parameters())


MPT_MUP_CONFIGS: dict[str, dict[str, Any]] = {
    "debug": {
        "d_model": 512,
        "n_layers": 4,
        "n_heads": 8,
        "expansion_ratio": 4,
        "max_seq_len": 1024,
        "vocab_size": 2048,
        "ffn_hidden_size": 2048,
        "init_device": "cpu",
        "tie_word_embeddings": True,
        "no_bias": True,
        "attn_config": {"attn_type": "torch"},
        "ffn_config": {
            "ffn_type": "mptmlp",
            "ffn_act_fn": {"name": "silu"},
            "fc_type": {"name": "torch"},
        },
        "mup_config": asdict(MuPConfig()),
        "init_config": asdict(ModelInitConfig(init_std=0.02, emb_init_std=0.02)),
    }
}

__all__ = [
    "HAS_MPT_MUP_SUPPORT",
    "ModelInitConfig",
    "MuPConfig",
    "MPTMuPConfig",
    "MPTMuPModelArgs",
    "TitanComposerMPTMuP",
    "MPT_MUP_CONFIGS",
]
