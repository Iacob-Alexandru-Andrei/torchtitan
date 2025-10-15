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
NODES:40 EDGES:20

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/models/llama3_mup/model/

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:40 CALL_DEPTH:1

__init__.py→[] mup_args.py→[] mup_model.py→[_precompute_freqs_cis((self))[HOT]→{_precompute_freqs_cis},init_weights((self,init_std: float)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgs # noqa: ARG002))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),init_weights((self)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None)),_iter_trainable_params((self)),_bucketize_parameters((self,param_entries: list[tuple[str,Parameter]])),_resolve_bucket_name((self,name: str,embed_suffixes: list[str],hidden_ln_suffixes: list[str],no_decay_suffixes: list[str],decay_weight_suffixes: list[str],))[HOT],_validate_bucket_counts((self,total_params: int,buckets: dict[str,list[Parameter]])),_compute_lr_scaling((self))[HOT],_resolve_optimizer_eps((self,eps: float,*,width_lr_scaling: float,))[HOT],_build_param_groups((self,buckets: dict[str,list[Parameter]],*,base_lr: float,weight_decay: float,width_lr_scaling: float,depth_lr_scaling: float,))[HOT],build_mup_optimizer_overrides((self,*,lr: float,eps: float,weight_decay: float,))[HOT],get_optimizer_param_groups((self,optimizer_config: dict[str,Any])),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,# noqa: ARG002))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 19 edges
Call: 1 edges

### CROSS_CLUSTER_FLOW

