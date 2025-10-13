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
NODES:44 EDGES:23

## DIRECTORY_TREE
ROOT: torchtitan/models/llama3/model/

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:44 CALL_DEPTH:3

args.py→[] model.py→[forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,))→{repeat_kv,repeat_kv,apply_rotary_emb},apply_rotary_emb((xq: torch.Tensor,xk: torch.Tensor,freqs_cis: torch.Tensor,))→{reshape_for_broadcast},_precompute_freqs_cis((self))[HOT]→{precompute_freqs_cis},precompute_freqs_cis((dim: int,end: int,theta: float = 10000.0))[HOT],reshape_for_broadcast((freqs_cis: torch.Tensor,x: torch.Tensor)),repeat_kv((x: torch.Tensor,n_rep: int)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,dim: int,hidden_dim: int,multiple_of: int,ffn_dim_multiplier: float | None,))[CTOR,DUNDER],forward((self,x)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgs))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),init_weights((self)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None,)),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER],_permute((self,w,n_heads_arg,dim1=None,dim2=None)),_reverse_permute((self,w,n_heads_arg,dim1=None,dim2=None)),to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 5 edges
Contains: 18 edges

### CROSS_CLUSTER_FLOW

