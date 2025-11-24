# Development Standards
- Setup the environment by running the commands in `torchtitan/experiments/fl/scripts/setup/setup_env.sh`.
- Versions are brittle; when invoking uv run commands, always include `--no-sync` (for example, `uv run --no-sync ...`). Use the setup script for the initial `uv sync`.
- This fork centralizes new features and experiments under `torchtitan/experiments/fl`; keep changes elsewhere minimal and compatibility-focused.
- Keep compatibility with upstream torchtitan simple by reusing existing imports and functionality whenever possible.
- When modifying a torchtitan component, model, or layer, create a subclass with the required changes. If compatibility must be broken, look for a higher-level base class before rebuilding from scratch.
- If two versions of a component must coexist, provide an explicit selector (for example, model-name strings or config booleans) and keep legacy behavior intact.
- Follow the linting and typing style used in `torchtitan/experiments/fl`: add docstrings with parameters, types, and returns, and type every symbol.
- Always run linting and the relevant tests (or pre-commit) before sending changes.
- Keep experimental behaviors behind feature flags or config options; defaults should remain backward compatible.
- Prefer small, well-documented diffs; note new configs or model names and keep logging clear for debugging.
- Avoid hardcoding environment-specific paths or secrets; keep inputs configurable.
- Add focused unit tests for compatibility layers and behavioral changes, especially around experiment routing and model selection.
- Profile before altering performance-sensitive paths; avoid regressions in hot kernels or distributed code.

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
NODES:56 EDGES:8

## DIRECTORY_TREE
ROOT: torchtitan/protocols/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:56 CALL_DEPTH:1

__init__.py→[] model.py→[__init__((self,model_args: BaseModelArgs))[CTOR,DUNDER]] model_converter.py→[__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER],convert((self,model: nn.Module)),post_optimizer_hook((self,model: Union[nn.Module,List[nn.Module]])),register_model_converter((converter_cls: type[ModelConverter],name: str)),__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER],convert((self,model: nn.Module)),post_optimizer_hook((self,model: Union[nn.Module,List[nn.Module]])),build_model_converters((job_config: JobConfig,parallel_dims: ParallelDims))[HOT]] state_dict_adapter.py→[__init__((self,model_args: BaseModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER]] train_spec.py→[register_train_spec((train_spec: TrainSpec)),update_train_spec((train_spec: TrainSpec)),unregister_train_spec((name: str)),get_train_spec((name: str))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 8 edges

### CROSS_CLUSTER_FLOW

