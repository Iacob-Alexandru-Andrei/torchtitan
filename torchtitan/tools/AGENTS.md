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
NODES:59 EDGES:14

## DIRECTORY_TREE
ROOT: torchtitan/tools/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:59 CALL_DEPTH:4

checkpoint_conversion.py→[download_dist_cp_checkpoint((location: S3CheckpointLocation,output_root: Path,))→{_remote_key,_enumerate_remote_step_files,_remote_key,_remote_key,_read_latest_step,_ensure_remote},_enumerate_remote_step_files((location: S3CheckpointLocation,candidate_relatives: list[Path],))→{_listing_prefix,_build_listing_uri,_remote_key},_read_latest_step((remote: RemoteUploaderDownloader,location: S3CheckpointLocation,scratch_dir: Path,))→{_remote_key},_ensure_tokenizer((output_dir: Path,tokenizer: str | None,*,revision: str | None = None,))→{_materialize_tokenizer},prepare_torchtitan_checkpoint((*,bucket: str,remote_root: str,prefix: str = "",step: int | None = None,local_checkpoint_root: Path,hf_output_dir: Path,model_name: str,model_flavor: str,hf_assets_path: Path | None = None,client_config: Mapping[str,Any] | None = None,tokenizer: str | None = None,tokenizer_revision: str | None = None,num_attempts: int = 3,num_concurrent_transfers: int = 4,use_processes: bool = False,))→{download_dist_cp_checkpoint},_remote_key((relative_path: Path,remote_root: str | None)),_build_listing_uri((bucket: str,prefix: str,remote_key: str))[HOT],_listing_prefix((prefix: str,remote_key: str)),_ensure_remote((remote: RemoteUploaderDownloader,run_name: str)),_materialize_tokenizer((output_dir: Path,tokenizer_source: str,*,revision: str | None = None,)),_write_hf_config((output_dir: Path,model_args: Any,*,tokenizer_name: str | None,tokenizer_obj: Any | None,model_name: str,)),_verify_native_state_dict_layers((state_dict: Mapping[str,Any],expected_layers: int,))] logging.py→[init_logger(void),warn_once((logger: logging.Logger,msg: str))] profiling.py→[] utils.py→[has_cuda_capability((major: int,minor: int)),get_device_info(void),__init__((self,gc_freq: int = 1000,debug: bool = False))[CTOR,DUNDER],run((self,step_count: int)),get_peak_flops((device_name: str)),check_if_feature_in_pytorch((feature_name: str,pull_request: str,min_nightly_version: Optional[str] = None,)),_round_up((x: int,y: int))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 2 edges
Call: 12 edges

### CROSS_CLUSTER_FLOW

