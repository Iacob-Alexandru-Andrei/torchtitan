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
NODES:29 EDGES:5

## DIRECTORY_TREE
ROOT: torchtitan/models/llama3/infra/

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:29 CALL_DEPTH:2

parallelize.py→[parallelize_llama((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→{apply_ddp,apply_fsdp,apply_compile,apply_tp,apply_tp},apply_tp((model: nn.Module,tp_mesh: DeviceMesh,loss_parallel: bool,enable_float8_tensorwise_tp: bool,)),apply_compile((model: nn.Module,compile_config: CompileConfig)),apply_fsdp((model: nn.Module,dp_mesh: DeviceMesh,param_dtype: torch.dtype,reduce_dtype: torch.dtype,pp_enabled: bool,cpu_offload: bool = False,reshard_after_forward_policy: str = "default",)),apply_ddp((model: nn.Module,dp_mesh: DeviceMesh,enable_compile: bool,enable_compiled_autograd: bool,))] pipeline.py→[pipeline_llama((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,device: torch.device,model_args: BaseModelArgs,parallelize_fn: ParallelizeFunction,loss_fn: LossFunction,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 5 edges

### CROSS_CLUSTER_FLOW

