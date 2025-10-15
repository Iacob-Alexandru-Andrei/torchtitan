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
NODES:54 EDGES:9

## DIRECTORY_TREE
ROOT: torchtitan/experiments/forge/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:54 CALL_DEPTH:2

__init__.py→[] engine.py→[close((self))] example_train.py→[batch_generator((self,data_iterable: Iterable[tuple[dict[str,torch.Tensor],torch.Tensor]])),forward_backward_step((self,input_dict: dict[str,torch.Tensor],labels: torch.Tensor)),train_step((self,data_iterator: Iterable[tuple[dict[str,torch.Tensor],torch.Tensor]])),state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),close((self))] job_config.py→[] train_spec.py→[get_train_spec((name: str))→{_transform_train_spec,_transform_train_spec},_transform_train_spec((original_spec: TrainSpec)),register_train_spec((train_spec: ForgeTrainSpec))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 2 edges
Contains: 7 edges

### CROSS_CLUSTER_FLOW

