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
NODES:227 EDGES:82

## DIRECTORY_TREE
ROOT: torchtitan/models/
├─ deepseek_v3/ → U[6]
│  ├─ infra/ → U[1]
│  └─ model/ → U[4]
├─ llama3/ → U[6]
│  ├─ infra/ → U[2]
│  └─ model/ → U[3]
└─ llama3_ft/ → U[1]

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:227 CALL_DEPTH:3

__init__.py→[get_train_spec(void),get_train_spec(void),get_train_spec(void)] args.py→[] attention.py→[__init__((self,attn_mask_type: str,fixed_block_size: int | None = None))[CTOR,DUNDER],forward((self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,scale: float | None = None,)),__init__((self,attn_mask_type: str))[CTOR,DUNDER],forward((self,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,scale: float | None = None,)),build_attention((use_flex_attn: bool,attn_mask_type: str,fixed_block_size: int | None = None))[HOT],init_attention_mask((batch: torch.Tensor,eos_id: int | None))] model.py→[forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,))→{repeat_kv,repeat_kv,apply_rotary_emb},forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,))→{apply_rotary_emb,apply_rotary_emb},apply_rotary_emb((xq: torch.Tensor,xk: torch.Tensor,freqs_cis: torch.Tensor,))→{reshape_for_broadcast},__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER]→{attention::build_attention},__init__((self,model_args: DeepSeekV3ModelArgs))[CTOR,DUNDER]→{attention::build_attention},__init__((self,model_args: DeepSeekV3ModelArgs))[CTOR,DUNDER]→{precompute_freqs_cis},init_weights((self,buffer_device: torch.device | None = None))→{precompute_freqs_cis},_precompute_freqs_cis((self))[HOT]→{precompute_freqs_cis},precompute_freqs_cis((dim: int,end: int,theta: float = 10000.0))[HOT],precompute_freqs_cis((args: DeepSeekV3ModelArgs))[HOT],reshape_for_broadcast((freqs_cis: torch.Tensor,x: torch.Tensor)),repeat_kv((x: torch.Tensor,n_rep: int)),apply_rotary_emb((x: torch.Tensor,freqs_cis: torch.Tensor)),init_weights((self,init_std: float)),__init__((self,dim: int,hidden_dim: int,multiple_of: int,ffn_dim_multiplier: float | None,))[CTOR,DUNDER],forward((self,x)),init_weights((self,init_std: float)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgs))[CTOR,DUNDER],__init__((self,layer_id: int,model_args: DeepSeekV3ModelArgs))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),forward((self,x: torch.Tensor,freqs_cis: torch.Tensor)),init_weights((self)),init_weights((self,buffer_device: torch.device)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None,)),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,)),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,))] moe.py→[__init__((self,dim: int,hidden_dim: int,))[CTOR,DUNDER],forward((self,x: torch.Tensor)),init_weights((self,init_std: float = 0.02)),__init__((self,dim: int,hidden_dim: int,num_experts: int,use_grouped_mm: bool,))[CTOR,DUNDER],forward((self,x: torch.Tensor,num_tokens_per_expert: torch.Tensor,)),init_weights((self,init_std: float)),__init__((self,dim: int,num_experts: int,top_k: int,score_func: Literal["softmax","sigmoid"],route_norm: bool,route_scale: float,_debug_force_load_balance: bool = False,))[CTOR,DUNDER],_debug_force_load_balance_routing((self,scores: torch.Tensor)),forward((self,x: torch.Tensor,expert_bias: torch.Tensor | None = None)),init_weights((self,init_std: float)),__init__((self,num_experts: int,top_k: int))[CTOR,DUNDER],forward((self,top_scores: torch.Tensor,selected_experts_indices: torch.Tensor,)),__init__((self,moe_args: MoEArgs,dim: int,hidden_dim: int))[CTOR,DUNDER],forward((self,x: torch.Tensor)),init_weights((self,init_std: float,buffer_device: torch.device,))] parallelize.py→[parallelize_llama((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→{apply_ddp,apply_fsdp,apply_compile,apply_tp,apply_tp},parallelize_deepseekv3((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→
{apply_ddp,apply_fsdp,apply_compile,apply_tp,apply_non_moe_tp},apply_tp((model: nn.Module,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,)),apply_non_moe_tp((model: nn.Module,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,)),apply_compile((model: nn.Module,compile_config: CompileConfig)),apply_fsdp((model: nn.Module,dp_mesh: DeviceMesh,param_dtype: torch.dtype,reduce_dtype: torch.dtype,pp_enabled: bool,cpu_offload: bool = False,reshard_after_forward_policy: str = "default",)),apply_ddp((model: nn.Module,dp_mesh: DeviceMesh,enable_compile: bool,enable_compiled_autograd: bool,))] pipeline.py→[pipeline_llama((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,device: torch.device,model_args: BaseModelArgs,parallelize_fn: ParallelizeFunction,loss_fn: LossFunction,))] quantization.py→[dequantize_from_fp8((weight: torch.Tensor,scale_inv: torch.Tensor,dtype=torch.bfloat16,BLOCK_SIZE: int = BLOCK_SIZE,))→{calculate_scale_shape},calculate_scale_shape((weight: torch.Tensor,BLOCK_SIZE: int = BLOCK_SIZE))] state_dict_adapter.py→[_dequantize((self,state_dict: dict[str,Any]))→{quantization::dequantize_from_fp8},_add_quantization_scale_inv_tensors((self,state_dict: dict[str,Any]))→{quantization::calculate_scale_shape},__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],__init__((self,model_args: DeepSeekV3ModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],_permute((self,w,n_heads_arg,dim1=None,dim2=None)),_reverse_permute((self,w,n_heads_arg,dim1=None,dim2=None)),to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any])),to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] utils.py→[__init__((self,model_args: BaseModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],_calculate_strided_shard_shard_indices((self,strided_shard_dim_degree: int,strided_shard_dim_rank: int,shard_dim_degree: int,shard_dim_rank: int,dim_size_to_split: int,)),_caculate_indices_from_placements((self,dim: int,dim_size: int,dtensor_placements: tuple,device_mesh: DeviceMesh,)),_get_local_experts_weights((self,abstract_key: str,titan_abstract_key: str,layer_id: str,grouped_expert_weight: torch.Tensor,)),_concatenate_expert_weights_dtensor((self,expert_weights_by_layer: dict[str,dict[str,dict[int,torch.Tensor]]],abstract_key: str,layer_num: str,device_mesh: DeviceMesh,)),_split_experts_weights((self,weight: torch.Tensor,n_experts: int)),_concatenate_expert_weights((self,expert_weights_by_layer: dict[str,dict[str,dict[int,torch.Tensor]]],abstract_key: str,layer_num: str,n_experts: int,)),get_dense_model_nparams_and_flops((model_args: BaseModelArgs,model: nn.Module,seq_len: int)),get_moe_model_nparams_and_flops((model_args: BaseModelArgs,model: nn.Module,seq_len: int))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 58 edges
Call: 24 edges

### CROSS_CLUSTER_FLOW

