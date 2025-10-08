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
NODES:46 EDGES:23

## DIRECTORY_TREE
ROOT: torchtitan/experiments/vlm/model/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:46 CALL_DEPTH:2

args.py→[] model.py→[forward((self,tokens: torch.Tensor,pixel_values: torch.Tensor,grid_thw: torch.Tensor,special_tokens: SpecialTokens,input_batch: torch.Tensor | None = None,))→{_scatter_img_tokens},_scatter_img_tokens((h_BSD,tokens_BS,i_NLD,i_mask_NL,img_id)),__init__((self,in_dim: int,out_dim: int))[CTOR,DUNDER],forward((self,x_NLD: torch.Tensor)),init_weights((self)),__init__((self,model_args: Llama3Siglip2ModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device=None))] siglip2.py→[forward((self,pixels_NLD: torch.Tensor,grid_hw: torch.Tensor))→{resize_positional_embeddings},resize_positional_embeddings((pos_embs_HWD: torch.Tensor,spatial_shapes_N2: torch.Tensor,max_length: int,)),__init__((self,args: Siglip2ModelArgs))[CTOR,DUNDER],init_weights((self)),__init__((self,args: Siglip2ModelArgs))[CTOR,DUNDER],forward((self,x: torch.Tensor)),init_weights((self)),__init__((self,args: Siglip2ModelArgs))[CTOR,DUNDER],forward((self,x: torch.Tensor)),init_weights((self)),__init__((self,args: Siglip2ModelArgs))[CTOR,DUNDER],forward((self,x: torch.Tensor)),init_weights((self)),__init__((self,args: Siglip2ModelArgs))[CTOR,DUNDER],forward((self,pixel_values_NLD: torch.FloatTensor,pixel_masks_NL: torch.BoolTensor,grid_hw: torch.LongTensor,)),init_weights((self))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 2 edges
Contains: 21 edges

### CROSS_CLUSTER_FLOW

