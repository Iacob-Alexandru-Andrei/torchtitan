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
NODES:71 EDGES:24

## DIRECTORY_TREE
ROOT: torchtitan/experiments/qwen3/
├─ infra/ → U[1]
└─ model/ → U[3]

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:71 CALL_DEPTH:3

__init__.py→[get_train_spec(void)] args.py→[] model.py→[apply_rotary_emb((xq: torch.Tensor,xk: torch.Tensor,rope_cache: torch.Tensor))→{rotate_half,rotate_half,reshape_for_broadcast},forward((self,x: torch.Tensor,rope_cache: torch.Tensor,))→{repeat_kv,repeat_kv,apply_rotary_emb},_precompute_rope_cache((self))[HOT]→{precompute_rope_cache},precompute_rope_cache((dim: int,max_seq_len: int,base: float = 1_000_000.0))[HOT],rotate_half((x: torch.Tensor)),reshape_for_broadcast((rope_cache: torch.Tensor,x: torch.Tensor)),repeat_kv((x: torch.Tensor,n_rep: int)),__init__((self,model_args: Qwen3ModelArgs))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,dim: int,hidden_dim: int,))[CTOR,DUNDER],forward((self,x)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: Qwen3ModelArgs))[CTOR,DUNDER],forward((self,x: torch.Tensor,rope_cache: torch.Tensor,)),init_weights((self,buffer_device: torch.device)),__init__((self,model_args: Qwen3ModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None,)),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,))] parallelize.py→[parallelize_qwen3((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→{apply_non_moe_tp},apply_non_moe_tp((model: nn.Module,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,enable_async_tp: bool,))] state_dict_adapter.py→[__init__((self,model_args: Qwen3ModelArgs,hf_assets_path: str | None))[CTOR,DUNDER],to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 16 edges
Call: 8 edges

### CROSS_CLUSTER_FLOW

