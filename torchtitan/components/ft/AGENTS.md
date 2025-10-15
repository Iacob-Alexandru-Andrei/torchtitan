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
NODES:32 EDGES:4

## DIRECTORY_TREE
ROOT: torchtitan/components/ft/
├─ config/ → U[2]
└─ diloco/ → U[3]

## ARCHITECTURAL_CLUSTERS

### UI_COMPONENTS
NODES:32 CALL_DEPTH:2

__init__.py→[] job_config.py→[] manager.py→[__init__((self,ft_config: FTConfig,))[CTOR,DUNDER],get_dp_info((self,dp_degree: int,dp_rank: int)),maybe_set_all_reduce_hook((self,model_parts: list[torch.nn.Module])),maybe_semi_sync_training((ft_config: FTConfig,ft_manager: FTManager,model: torch.nn.Module,n_layers: int,optimizer: torch.optim.Optimizer,fragment_fn: Optional[Callable[...,list[nn.Module]]] = None,))] protocol.py→[] utils.py→[fragment_llm((model: nn.Module,ft_config: FTConfig,n_layers: int,))→{module_split},module_split((model: nn.Module,module_fqns_per_model_fragment: list[list[str]],))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 1 edges
Contains: 3 edges

### CROSS_CLUSTER_FLOW

