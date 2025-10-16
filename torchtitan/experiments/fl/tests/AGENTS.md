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
NODES:26 EDGES:7

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/tests/

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:26 CALL_DEPTH:1

test_unigram_metrics.py→[__init__((self,*_args: object,**_kwargs: object))[CTOR,DUNDER],add_state((self,name: str,default: torch.Tensor,dist_reduce_fx: str | None = None,)),register_buffer((self,name: str,tensor: torch.Tensor)),__init__((self,*_args: object,**_kwargs: object))[CTOR,DUNDER],__init__((self,*_args: object,**_kwargs: object))[CTOR,DUNDER],get_peak_stats((self)),reset_peak_stats((self)),test_unigram_manager_aggregation_and_reset(void)[TEST],test_unigram_manager_teardown_removes_metric(void)[TEST],test_fl_metrics_processor_registers_expected_callbacks(void)[TEST],test_unigram_payload_reports_local_and_global_metrics(void)[TEST],test_unigram_local_metric_logged_before_global(void)[TEST]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 7 edges

### CROSS_CLUSTER_FLOW

