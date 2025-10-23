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
NODES:140 EDGES:49

## DIRECTORY_TREE
ROOT: torchtitan/experiments/llama4/
├─ infra/ → U[1]
├─ model/ → U[3]
└─ scripts/ → U[2]

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:140 CALL_DEPTH:3

__init__.py→[get_train_spec(void)] args.py→[] convert_hf_to_dcp_with_gpus.py→[convert_to_titan_fqns((fqn: str))→{extract_layer_number},_create_fqn_mappings((self,state_dict: dict[str,torch.Tensor]))→{convert_to_titan_fqns},_get_load_assignments((self,state_dict: dict[str,Any]))→{convert_to_hf_shape},extract_layer_number((s)),convert_to_hf_shape((fqn: str,titan_fqns: list[str],dtensor: DTensor)),convert_to_titan_tensors((fqn: str,full_tensor: torch.Tensor)),__init__((self,process_group: dist.ProcessGroup,path: str,token: Optional[str] = None,loader_every_n_ranks: int = 8,))[CTOR,DUNDER],convert((self,state_dict: dict[str,torch.Tensor])),_load_metadata((self)),_load_round((self,assignment: _Assignment)),_reshard_send((self,assignment: _Assignment,loaded_state_dict: dict[str,torch.Tensor],)),_reshard_receive((self,assignment: _Assignment,state_dict: dict[str,torch.Tensor])),_reshard((self,result: dict[str,torch.Tensor],state_dict: dict[str,torch.Tensor],)),_create_verified_state_dict((pg: dist.ProcessGroup,mesh: DeviceMesh)),_verify_state_dict((state_dict: dict[str,torch.Tensor],path: str,rank: int))] convert_meta_to_dcp_with_gpus.py→[_create_fqn_mappings((self,state_dict: dict[str,torch.Tensor]))→{convert_hf_to_dcp_with_gpus::convert_to_titan_fqns},convert_to_titan_fqns((fqn: str)),get_shard_dim((fqn: str)),split_fused_qkv((shards: list[torch.Tensor])),__init__((self,process_group: dist.ProcessGroup,path: str,loader_every_n_ranks: int = 8,))[CTOR,DUNDER],convert((self,state_dict: dict[str,torch.Tensor])),_get_file_path((self,loader_id: int)),_load_metadata((self)),_get_load_assignments((self,state_dict: dict[str,torch.Tensor])),_load_round((self,assignment: _Assignment)),_reshard_send((self,assignment: _Assignment,loaded_state_dict: dict[str,torch.Tensor],)),_reshard_receive((self,assignment: _Assignment,state_dict: dict[str,torch.Tensor])),_reshard((self,results: list[dict[str,torch.Tensor]],state_dict: dict[str,torch.Tensor],)),_create_verified_state_dict((pg: dist.ProcessGroup,mesh: DeviceMesh)),_verify_state_dict((state_dict: dict[str,torch.Tensor],path: str,rank: int))] model.py→[forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,))→{repeat_kv,repeat_kv,apply_rotary_emb},apply_rotary_emb((xq: torch.Tensor,xk: torch.Tensor,freqs_cis: torch.Tensor,))→{reshape_for_broadcast},_precompute_freqs_cis((self))[HOT]→{precompute_freqs_cis},precompute_freqs_cis((dim: int,end: int,theta: float = 10000.0))[HOT],reshape_for_broadcast((freqs_cis: torch.Tensor,x: torch.Tensor)),repeat_kv((x: torch.Tensor,n_rep: int)),__init__((self,model_args: TransformerModelArgs,use_rope: bool = True,fixed_block_size: int | None = None,))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,dim: int,hidden_dim: int,multiple_of: int,ffn_dim_multiplier: float | None,))[CTOR,DUNDER],forward((self,x)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgs,))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),init_weights((self,buffer_device: torch.device)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None,)),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,))] parallelize.py→[parallelize_llama((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→
{apply_fsdp,apply_fsdp,apply_compile,apply_moe_ep_tp,apply_non_moe_tp},apply_non_moe_tp((model: nn.Module,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,)),apply_fsdp((model: nn.Module,dp_mesh: DeviceMesh,param_dtype: torch.dtype,reduce_dtype: torch.dtype,pp_enabled: bool,cpu_offload: bool = False,reshard_after_forward_policy: str = "default",ep_degree: int = 1,dp_mod_ep_mesh: DeviceMesh | None = None,gradient_divide_factor: int | None = None,)),apply_moe_ep_tp((model: nn.Module,tp_mesh: DeviceMesh | None,ep_mesh: DeviceMesh | None,ep_tp_mesh: DeviceMesh | None,etp_enabled: bool,)),apply_compile((model: nn.Module,compile_config: CompileConfig))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None))[CTOR,DUNDER],to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 35 edges
Call: 14 edges

### CROSS_CLUSTER_FLOW

