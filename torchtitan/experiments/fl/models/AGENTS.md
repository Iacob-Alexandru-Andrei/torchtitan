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
NODES:230 EDGES:63

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/models/
├─ llama3_mup/ → TST[1] U[8]
│  ├─ infra/ → U[2]
│  ├─ model/ → U[4]
│  ├─ tests/ → TST[1]
│  └─ train_configs/ → U[1]
├─ mosaic_llama3/ → U[1]
├─ mosaic_llama3_mup/ → U[1]
├─ mosaic_mpt_mup/ → U[1]
├─ mpt_mup/ → U[5]
│  ├─ infra/ → U[1]
│  ├─ model/ → U[2]
│  └─ train_configs/ → U[1]
└─ tests/ → TST[1]

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:230 CALL_DEPTH:2

__init__.py→[build_mup_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT]→{_maybe_override_optimizer_config},_get_llama3_mup_spec(void),_update_vocab_sizes((base_spec: TrainSpec,mosaic_spec: TrainSpec)),_update_vocab_sizes((base_spec: TrainSpec,mosaic_spec: TrainSpec)),get_train_spec(void),_get_mpt_mup_spec(void),build_mup_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT],get_train_spec(void),get_train_spec(void),_maybe_override_optimizer_config((overrides: MuPOptimizerOverride | None,optimizer_config: OptimizerConfig,)),get_train_spec(void),get_train_spec(void)] mosaic_adapter.py→[] mup_args.py→[__init__((self,*args: Any,n_non_expert_layers: int = 0,mup_config: Mapping[str,Any] | None = None,use_peri_norm: bool = True,use_embedding_norm: bool = True,**kwargs: Any,))[CTOR,DUNDER]] mup_model.py→[get_optimizer_param_groups((self,optimizer_config: Mapping[str,Any],))→
{test_mup_model::setUp,test_mup_model::setUp,test_mup_model::setUp,test_mup_model::setUp,test_mup_model::setUp},_partial_llama_getattribute((self: PartialLlamaConfig,key: str)),__init__((self,inner: nn.Module,scale: float))[CTOR,DUNDER],_partial_llama_getitem((self: PartialLlamaConfig,key: str)),forward((self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,)),__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],init_weights((self,init_std: float)),init_weights((self,init_std: float)),__init__((self,vocab_size: int,d_model: int,*,scale: float = 1.0,use_embedding_norm: bool = False,norm_type: str = "low_precision_layernorm",norm_eps: float = 1e-5,device: str | None = None,**kwargs: Any,))[CTOR,DUNDER],__init__((self,layer_id: int,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],forward((self,input: torch.Tensor,unembed: bool = False)),__init__((self,*,depth_multiplier: float = 1.0,depth_alpha_enabled: bool = False,depth_alpha_exp: float = 1.0,use_peri_norm: bool = False,norm_type: str = "low_precision_layernorm",norm_eps: float = 1e-5,**kwargs: Any,))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),forward((# noqa: PLR0913 self,x: torch.Tensor,past_key_value: tuple[torch.Tensor,torch.Tensor] | None = None,attn_bias: torch.Tensor | None = None,rotary_emb_w_meta_info: dict | None = None,attention_mask: torch.ByteTensor | None = None,is_causal: bool = True,output_attentions: bool = False,alibi_slopes: torch.Tensor | None = None,flash_attn_padding_info: dict[str,torch.Tensor] | None = None,prev_layer_key_value: tuple[torch.Tensor,torch.Tensor] | None = None,key_value_states: torch.Tensor | None = None,x_prev: torch.Tensor | None = None,pos_id_within_seq: torch.Tensor | None = None,)),init_weights((self)),__init__((self,model_args: TransformerModelArgsMuP))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None)),_iter_trainable_params((self)),__init__((self,config: MPTMuPConfig))[CTOR,DUNDER],_bucketize_parameters((self,param_entries: list[tuple[str,Parameter]])),extract_block_args((self,block_args: dict[str,Any])),forward((self,*args: Any,**kwargs: Any)),_resolve_bucket_name((self,name: str,embed_suffixes: list[str],hidden_ln_suffixes: list[str],no_decay_suffixes: list[str],decay_weight_suffixes: list[str],))[HOT],__init__((self,config: MPTMuPConfig))[CTOR,DUNDER],reset_parameters((self)),_validate_bucket_counts((self,total_params: int,buckets: dict[str,list[Parameter]])),param_init_fn((self,module: nn.Module)),_compute_lr_scaling((self))[HOT],_resolve_optimizer_eps((self,eps: float,*,width_lr_scaling: float,))[HOT],_build_param_groups((self,buckets: dict[str,list[Parameter]],*,base_lr: float,weight_decay: float,width_lr_scaling: float,depth_lr_scaling: float,))[HOT],build_mup_optimizer_overrides((self,*,lr: float,eps: float,weight_decay: float,))[HOT],get_optimizer_param_groups((self,optimizer_config: dict[str,Any])),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,# noqa: ARG002)),forward((self,input_ids: torch.LongTensor | None = None,attention_mask: torch.ByteTensor | None = None,sequence_id: torch.LongTensor | None = None,labels: torch.LongTensor | None = None,**kwargs: Any,)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None)),forward((self,tokens: torch.Tensor,input_batch: MutableMapping[str,torch.Tensor] | None = None,)),_call_get_param_groups((self,*,lr: float,eps: float,weight_decay: float,)),build_mup_optimizer_overrides((self,*,lr: float,eps: float,weight_decay: float,))[HOT],get_optimizer_param_groups((self,optimizer_config: Mapping[str,Any],))] parallelize.py→[parallelize_llama_mup((model: Transformer,parallel_dims: ParallelDims,job_config: JobConfig,))→{_apply_mup_tp},parallelize_mpt_mup((model: Transformer,parallel_dims: ParallelDims,job_config: JobConfig,))→
{_apply_mpt_tp},_apply_mpt_tp((model: Transformer,tp_mesh: DeviceMesh,*,loss_parallel: bool,enable_float8_tensorwise_tp: bool,)),_apply_mup_tp((model: Transformer,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],to_hf((self,state_dict: dict[str,Any]))] test_mosaic_adapter.py→[test_register_applies_builder_overrides((self))[HOT,TEST]→{__init__::get_train_spec},_dummy_builder((*_args: object,**_kwargs: object))[HOT],tearDown((self)),test_build_uses_mosaic_name_by_default((self))[HOT,TEST]] test_mup_model.py→[setUp((self)),_get_expected_mup_eps((self,base_eps: float)),test_model_initialization((self))[TEST],test_forward_pass((self))[TEST],test_weight_initialization((self))[TEST],test_optimizer_overrides_build_param_groups((self))[HOT,TEST],test_optimizer_overrides_disabled_when_hidden_scaling_off((self))[TEST],test_mosaic_builder_integrates_mup_overrides((self))[HOT,TEST],test_mosaic_builder_desloc_requires_ft((self))[HOT,TEST],test_tie_word_embeddings_shares_parameter((self))[TEST]] utils.py→[build_mosaic_spec((base_spec: TrainSpec,*,spec_name: str,overrides: MosaicSpecOverrides | None = None,))[HOT],ensure_mosaic_spec((base_spec_name: str,*,spec_name: str | None = None,overrides: MosaicSpecOverrides | None = None,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 54 edges
Call: 9 edges

### CROSS_CLUSTER_FLOW

