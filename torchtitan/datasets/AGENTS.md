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
NODES:25 EDGES:6

## DIRECTORY_TREE
ROOT: torchtitan/datasets/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:25 CALL_DEPTH:2

__init__.py→[] hf_datasets.py→[__init__((self,dataset_name: str,dataset_path: str | None,tokenizer: BaseTokenizer,seq_len: int = 2048,dp_rank: int = 0,dp_world_size: int = 1,infinite: bool = False,))[CTOR,DUNDER]→{_validate_dataset},_load_c4_dataset((dataset_path: str,split: str)),_process_c4_text((sample: dict[str,Any])),_validate_dataset((dataset_name: str,dataset_path: str | None = None)),_get_data_iter((self)),__iter__((self))[DUNDER],load_state_dict((self,state_dict)),state_dict((self)),build_hf_dataloader((dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: JobConfig,infinite: bool = True,))[HOT],build_hf_validation_dataloader((dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: JobConfig,infinite: bool = False,))[HOT]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 1 edges
Contains: 5 edges

### CROSS_CLUSTER_FLOW

