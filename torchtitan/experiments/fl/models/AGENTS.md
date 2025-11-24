# Development Standards
- Setup the environment by running the commands in `torchtitan/experiments/fl/scripts/setup/setup_env.sh`.
- Versions are brittle; when invoking uv run commands, always include `--no-sync` (for example, `uv run --no-sync ...`). Use the setup script for the initial `uv sync`.
- This fork centralizes new features and experiments under `torchtitan/experiments/fl`; keep changes elsewhere minimal and compatibility-focused.
- Keep compatibility with upstream torchtitan simple by reusing existing imports and functionality whenever possible.
- When modifying a torchtitan component, model, or layer, create a subclass with the required changes. If compatibility must be broken, look for a higher-level base class before rebuilding from scratch.
- If two versions of a component must coexist, provide an explicit selector (for example, model-name strings or config booleans) and keep legacy behavior intact.
- Follow the linting and typing style used in `torchtitan/experiments/fl`: add docstrings with parameters, types, and returns, and type every symbol.
- Always run linting and the relevant tests (or pre-commit) before sending changes.
- Keep experimental behaviors behind feature flags or config options; defaults should remain backward compatible.
- Prefer small, well-documented diffs; note new configs or model names and keep logging clear for debugging.
- Avoid hardcoding environment-specific paths or secrets; keep inputs configurable.
- Add focused unit tests for compatibility layers and behavioral changes, especially around experiment routing and model selection.
- Profile before altering performance-sensitive paths; avoid regressions in hot kernels or distributed code.

# EMBARGO: LLM-Optimized Codebase Dependency Graph

**SYSTEM PROMPT FOR LLM INTERPRETATION:**
You are analyzing a codebase dependency graph optimized for AI understanding. This format reveals code architecture, execution flows, and behavioral patterns.

## INTERPRETATION KEY

### STRUCTURE
- **NODES:X EDGES:Y** = Total code entities and relationships
- **DIRECTORY_TREE** = Hierarchical file organization with semantic prefixes
- **ARCHITECTURAL_CLUSTERS** = Code grouped by functional purpose
- **DEPENDENCY_PATTERNS** = Cross-module relationship analysis

### BEHAVIORAL NOTATION
- **filename.rs→[...]** = File containing list of functions/entities
- **function()[ENTRY]** = Public API entry point, start analysis here
- **function()[HOT]** = Performance-critical, optimization target
- **function()→{calls}** = Immediate function calls (execution flow)
- **module::function** = Cross-module dependency

### ANALYSIS GUIDANCE
1. **Entry Points**: Start with [ENTRY] functions to understand public APIs
2. **Execution Flow**: Follow →{calls} to trace code execution paths
3. **Hot Paths**: Focus [HOT] functions for performance analysis
4. **Architecture**: Use clusters to understand system organization
5. **Dependencies**: Cross-cluster flows show coupling patterns

### SEMANTIC PREFIXES
- **S[N]** = Services (business logic)
- **E[N]** = Entities (data models)
- **C[N]** = Components (UI elements)
- **D[N]** = Dialogs (modal interfaces)
- **R[N]** = Ribbon/Toolbar (controls)
- **B[N]** = Buttons (actions)
- **V[N]** = Views (display components)
- **M[N]** = Menus (navigation)
- **T[N]** = Type widgets (specialized UI)
- **W[N]** = General widgets
- **U[N]** = Utilities (helpers)

### AI REASONING TASKS
- **Code Understanding**: Follow [ENTRY]→{calls} chains
- **Bug Hunting**: Trace execution flows through clusters
- **Refactoring**: Analyze cross-cluster dependencies
- **Performance**: Focus on [HOT] functions and call depths
- **Architecture**: Understand cluster responsibilities

---

# CODE_GRAPH
NODES:286 EDGES:98

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/models/
├─ llama3_mup/ → TST[1] U[8]
│  ├─ infra/ → U[2]
│  ├─ model/ → U[4]
│  ├─ tests/ → TST[1]
│  └─ train_configs/ → U[1]
├─ llama3_mup_disco/ → TST[1] U[9]
│  ├─ infra/ → U[2]
│  ├─ model/ → U[5]
│  ├─ tests/ → TST[1]
│  └─ train_configs/ → U[1]
├─ mosaic_llama3/ → U[1]
├─ mosaic_llama3_mup/ → U[1]
├─ mosaic_llama3_mup_disco/ → U[1]
└─ tests/ → TST[1]

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:286 CALL_DEPTH:4

__init__.py→[_register_base_specs(void),_update_vocab_sizes((base_spec: TrainSpec,mosaic_spec: TrainSpec)),_update_vocab_sizes((base_spec: TrainSpec,mosaic_spec: TrainSpec)),_update_vocab_sizes((base_spec: TrainSpec,mosaic_spec: TrainSpec)),_register_mosaic_specs(void),build_mup_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT],build_mup_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT],get_train_spec(void),get_train_spec(void),get_train_spec(void),get_train_spec(void),get_train_spec(void)] disco_init.py→[initialize_tensor((tensor: Tensor,*,init_type: str,init_std: float,scion_eps: float,trunc_normal_cutoff: float = _DEFAULT_TRUNC_CUTOFF,mean: float = 0.0,))→{disco_normal_,disco_normal_,disco_normal_,_init_scaled_orthogonal,_init_orthogonal,_canonicalize_init_type},init_linear_weight((linear: nn.Linear,*,init_std: float,init_type: str,scion_eps: float,trunc_normal_cutoff: float,))→{initialize_tensor},_canonicalize_init_type((init_type: str)),disco_normal_((tensor: Tensor,*,mean: float = 0.0,std: float = 1.0,norm_axis: int = 1,eps: float = 1e-12,scale_type: str | None = None,)),_init_orthogonal((tensor: Tensor,*,gain: float)),_init_scaled_orthogonal((tensor: Tensor,*,gain: float))] mosaic_adapter.py→[] mup_args.py→[] mup_model.py→[forward((self,x: torch.Tensor))→{_cast_if_autocast_enabled,_cast_if_autocast_enabled,_cast_if_autocast_enabled},__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER]→{_build_norm_module,_build_norm_module},init_weights((self,buffer_device: torch.device | None = None))→{disco_init::initialize_tensor,disco_init::initialize_tensor},_build_param_groups((self,buckets: dict[str,list[Parameter]],*,base_lr: float,weight_decay: float,width_lr_scaling: float,depth_lr_scaling: float,))[HOT]→{test_mup_model::setUp,test_mup_model::setUp},_build_head_norm((self,model_args: TransformerModelArgsMuP))[HOT]→{_build_norm_module},init_weights((self,init_std: float))→{disco_init::init_linear_weight},init_weights((self,init_std: float))→{disco_init::init_linear_weight},_refresh_norms((self,model_args: TransformerModelArgsMuP))→
{_build_norm_module},_cast_if_autocast_enabled((tensor: torch.Tensor | None)),__init__((self,normalized_shape: int | Sequence[int],*,eps: float = 1e-6,elementwise_affine: bool = True,add_unit_offset: bool = True,force_bf16: bool = False,))[CTOR,DUNDER],__init__((self,normalized_shape: int | tuple[int,...],eps: float = 1e-05,elementwise_affine: bool = True,device: torch.device | None = None,dtype: torch.dtype | None = None,bias: bool = False,))[CTOR,DUNDER],reset_parameters((self)),forward((self,x: torch.Tensor)),_build_norm_module((normalized_shape: int | Sequence[int],*,eps: float,model_args: TransformerModelArgsMuP,prefer_torch: bool,elementwise_affine: bool = True,bias: bool = False,))[HOT],__init__((self,inner: nn.Module,scale: float))[CTOR,DUNDER],forward((self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,*,scale: float | None = None,**kwargs: Any,)),__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],forward((self,x: torch.Tensor)),_build_head_norm((self,model_args: TransformerModelArgsMuP))[HOT],__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],forward((self,x: torch.Tensor)),__init__((self,layer_id: int,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),build_mup_optimizer_overrides((self,*,lr: float,eps: float,weight_decay: float,scion_hidden_scale: float | None = None,scion_output_scale: float | None = None,scion_hidden_norm: str | None = None,scion_output_norm: str | None = None,scion_hidden_norm_kwargs: dict[str,Any] | None = None,scion_output_norm_kwargs: dict[str,Any] | None = None,))[HOT],init_weights((self)),__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],_build_param_groups((self,buckets: dict[str,list[Parameter]],*,base_lr: float,weight_decay: float,width_lr_scaling: float,depth_lr_scaling: float,scion_hidden_scale: float | None = None,scion_output_scale: float | None = None,scion_hidden_norm: str | None = None,scion_output_norm: str | None = None,scion_hidden_norm_kwargs: dict[str,Any] | None = None,scion_output_norm_kwargs: dict[str,Any] | None = None,))[HOT],init_weights((self,buffer_device: torch.device | None = None)),_precompute_freqs_cis((self))[HOT],_iter_trainable_params((self)),_bucketize_parameters((self,param_entries: list[tuple[str,Parameter]])),_apply_scion_scales((self,labels: Sequence[str],groups: Sequence[dict[str,Any]],*,hidden_scale: float | None,output_scale: float | None,hidden_norm: str | None,output_norm: str | None,hidden_norm_kwargs: dict[str,Any] | None,output_norm_kwargs: dict[str,Any] | None,)),_resolve_bucket_name((self,name: str,embed_suffixes: list[str],hidden_ln_suffixes: list[str],no_decay_suffixes: list[str],decay_weight_suffixes: list[str],))[HOT],_validate_bucket_counts((self,total_params: int,buckets: dict[str,list[Parameter]])),_compute_lr_scaling((self))[HOT],_apply_disco_norm_overrides((self,labels: Sequence[str],groups: Sequence[dict[str,Any]],)),_resolve_optimizer_eps((self,eps: float,*,width_lr_scaling: float,))[HOT],_log_disco_norm_assignments((self,labels: Sequence[str],groups: Sequence[dict[str,Any]],)),get_optimizer_param_groups((self,optimizer_config: dict[str,Any])),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,# noqa: ARG002)),build_mup_optimizer_overrides((self,*,lr: float,eps: float,weight_decay: float,))[HOT],get_optimizer_param_groups((self,optimizer_config: dict[str,Any])),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,# noqa: ARG002))] parallelize.py→[parallelize_llama_mup((model: Transformer,parallel_dims: ParallelDims,job_config: JobConfig,))→
{_apply_mup_tp},parallelize_llama_mup((model: Transformer,parallel_dims: ParallelDims,job_config: JobConfig,))→{_apply_mup_tp},_apply_mup_tp((model: Transformer,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,)),_apply_mup_tp((model: Transformer,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],to_hf((self,state_dict: dict[str,Any])),to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] test_mosaic_adapter.py→[test_register_applies_builder_overrides((self))[HOT,TEST]→{__init__::get_train_spec},_dummy_builder((*_args: object,**_kwargs: object))[HOT],tearDown((self)),test_build_uses_mosaic_name_by_default((self))[HOT,TEST]] test_mup_model.py→[setUp((self)),setUp((self)),_get_expected_mup_eps((self,base_eps: float)),_get_expected_mup_eps((self,base_eps: float)),test_model_initialization((self))[TEST],test_model_initialization((self))[TEST],test_forward_pass((self))[TEST],test_forward_pass((self))[TEST],test_weight_initialization((self))[TEST],test_weight_initialization((self))[TEST],test_optimizer_overrides_build_param_groups((self))[HOT,TEST],test_optimizer_overrides_build_param_groups((self))[HOT,TEST],test_optimizer_overrides_disabled_when_hidden_scaling_off((self))[TEST],test_optimizer_overrides_disabled_when_hidden_scaling_off((self))[TEST],test_mosaic_builder_integrates_mup_overrides((self))[HOT,TEST],test_mosaic_builder_integrates_mup_overrides((self))[HOT,TEST],test_mosaic_builder_desloc_requires_ft((self))[HOT,TEST],test_mosaic_builder_desloc_requires_ft((self))[HOT,TEST],test_tie_word_embeddings_shares_parameter((self))[TEST],test_tie_word_embeddings_shares_parameter((self))[TEST]] utils.py→[build_mosaic_spec((base_spec: TrainSpec,*,spec_name: str,overrides: MosaicSpecOverrides | None = None,))[HOT],ensure_mosaic_spec((base_spec_name: str,*,spec_name: str | None = None,overrides: MosaicSpecOverrides | None = None,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 75 edges
Call: 23 edges

### CROSS_CLUSTER_FLOW

