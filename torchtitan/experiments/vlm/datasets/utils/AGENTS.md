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
NODES:29 EDGES:9

## DIRECTORY_TREE
ROOT: torchtitan/experiments/vlm/datasets/utils/

## ARCHITECTURAL_CLUSTERS

### UTILS
NODES:29 CALL_DEPTH:3

image.py→[_resize_image_by_patch_count((image: Image.Image,max_patch_per_image: int,patch_size: int,merge_size: int,min_patch_per_image: int = 1,))→{_smart_resize,_smart_resize,_smart_resize},process_image((image: str | bytes | Image.Image,patch_size: int = 16,merge_size: int = 1,max_patch_per_image: int = 256,min_patch_per_image: int = 1,))→{_resize_image_by_patch_count},_smart_resize((height: int,width: int,factor: int,# should be equal patch_size * merge_size max_patch_per_image: int,min_patch_per_image: int = 1,)),calculate_image_tokens((image: Image.Image | torch.Tensor,patch_size: int,spatial_merge_size: int,)),convert_to_patches((pixel_values: torch.Tensor,patch_size: int,temporal_patch_size: int = 1,)),pad_patches((patches: torch.Tensor,grids: torch.Tensor,max_patches: int,)),pad_empty_images_to_target_batch_size((patches: torch.Tensor,grids: torch.Tensor,max_images: int,))] packing.py→[__init__((self,max_seq_length: int,buffer_size: int = 100,batch_size: int = 8,))[CTOR,DUNDER],_pack_buffered_samples((self)),add_sample((self,sample: dict[str,Any])),has_batch_ready((self)),get_next_batch((self))] text.py→[pad_text_batch((input_ids: torch.Tensor,labels: torch.Tensor,seq_len: int,padding_idx: int = 0,ignore_idx: int = -100,)),pad_input_ids_and_labels_to_target_batch_size((input_ids: torch.Tensor,labels: torch.Tensor,target_batch_size: int,padding_idx: int = 0,ignore_idx: int = -100,)),process_text_with_images((text: list[str],image_tokens: list[tuple[int,int,int]],# [(total,width,height),...] tokenizer,special_tokens,add_eos: bool = True,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 5 edges
Call: 4 edges

### CROSS_CLUSTER_FLOW

