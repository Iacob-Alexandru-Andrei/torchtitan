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
NODES:88 EDGES:16

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/models/
├─ llama3_mup/ → TST[1] U[8]
│  ├─ infra/ → U[2]
│  ├─ model/ → U[4]
│  ├─ tests/ → TST[1]
│  └─ train_configs/ → U[1]
├─ mosaic_llama3/ → U[1]
└─ mosaic_llama3_mup/ → U[1]

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:88 CALL_DEPTH:1

__init__.py→[_get_llama3_mup_spec(void),get_train_spec(void),get_train_spec(void),build_mup_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT],get_train_spec(void)] mup_args.py→[] mup_model.py→[_precompute_freqs_cis((self))[HOT]→{_precompute_freqs_cis},init_weights((self,init_std: float)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgs # noqa: ARG002))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),init_weights((self)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None)),get_optimizer_param_groups((self,optimizer_config: dict[str,Any])),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,# noqa: ARG002))] parallelize.py→[parallelize_llama_mup((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER]] test_mup_model.py→[setUp((self)),test_model_initialization((self))[TEST],test_forward_pass((self))[TEST],test_weight_initialization((self))[TEST]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 1 edges
Contains: 15 edges

### CROSS_CLUSTER_FLOW

