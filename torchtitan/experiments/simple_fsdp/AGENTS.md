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
NODES:94 EDGES:19

## DIRECTORY_TREE
ROOT: torchtitan/experiments/simple_fsdp/
├─ deepseek_v3/ → U[3]
├─ llama3/ → U[3]
└─ tests/ → TST[2]

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:21 CALL_DEPTH:3

integration_tests.py→[main(void)[ENTRY],build_simple_fsdp_test_list(void)[HOT]] test_numerics.py→[run_simple_fsdp((self,model,inputs,labels,epoch=20))→{simple_fsdp::data_parallel},init_test((self)),get_input((self)),run_fsdp2((self,model,inputs,labels,epoch=20)),test_replicate_convergence((self))[TEST],test_fullyshard_convergence((self))[TEST],test_hybridshard_convergence((self))[TEST]] 
### UTILITY_LAYER
NODES:73 CALL_DEPTH:3

__init__.py→[get_train_spec(void),get_train_spec(void)] model.py→[__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],__init__((self,model_args: DeepSeekV3ModelArgs))[CTOR,DUNDER],init_weights((self,*args,**kwargs)),init_weights((self,*args,**kwargs))] parallelize.py→[parallelize_deepseekv3((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→{simple_fsdp::data_parallel,simple_fsdp::data_parallel},parallelize_llama((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→{simple_fsdp::data_parallel}] simple_fsdp.py→[data_parallel((model,device_mesh,mode="replicate",ac_mode: str = "none",mp_policy: Optional[MixedPrecisionPolicy] = None,shard_dim: int = 0,))→{_register_parametrization},_distribute_dtensor((tensor: DTensor,device_mesh: DeviceMesh,dp_placements: Sequence[Placement],)),_register_parametrization((module: nn.Module,param_names: List[str],parametrization: nn.Module)),fsdp_policy(void),__init__((self,device_mesh,param_sharding,mode,regional_ac,mp_policy,))[CTOR,DUNDER],replicate_compute((self,x))[HOT],forward((self,x))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 14 edges
Call: 5 edges

### CROSS_CLUSTER_FLOW
TESTS→UTILITY_LAYER: 1

