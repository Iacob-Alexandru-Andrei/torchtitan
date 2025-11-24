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
NODES:98 EDGES:25

## DIRECTORY_TREE
ROOT: torchtitan/distributed/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:98 CALL_DEPTH:3

__init__.py→[__init__((self,*,input_layout: Placement | None = None,output_layout: Placement | None = None,use_local_output: bool = True,))[CTOR,DUNDER],_apply((self,module: nn.Module,device_mesh: DeviceMesh))] activation_checkpoint.py→[_apply_ac_to_transformer_block((module: nn.Module,ac_config: ACConfig,*,base_fqn: str | None = None,model_compile_enabled: bool = False,use_flex_attn: bool = False,op_sac_save_list: set[torch._ops.OpOverload] | None = None,))→{_apply_layer_sac,_apply_op_sac,_apply_op_sac_to_transformer_block_with_flex,_apply_full_ac},apply_ac((model: nn.Module,ac_config: ACConfig,*,model_compile_enabled: bool = False,use_flex_attn: bool = False,op_sac_save_list: set[torch._ops.OpOverload] | None = None,))→{_apply_ac_to_transformer_block},_apply_layer_sac((module: nn.Module,ac_config: ACConfig)),_apply_op_sac((module: nn.Module,ac_config: ACConfig,*,base_fqn: str | None = None,op_sac_save_list: set[torch._ops.OpOverload],)),_apply_full_ac((module: nn.Module,ac_config: ACConfig)),_apply_op_sac_to_transformer_block_with_flex((module: nn.Module,ac_config: ACConfig,*,base_fqn: str | None = None,model_compile_enabled: bool = False,op_sac_save_list: set[torch._ops.OpOverload],))] expert_parallel.py→[set_token_group_alignment_size_m((alignment_size: ValidTokenGroupAlignmentSize,)),_partition_fn((self,name,module,device_mesh)),_apply((self,module: nn.Module,device_mesh: DeviceMesh)),__init__((self))[CTOR,DUNDER],_token_dispatch((self,mod,inputs,device_mesh)),_token_combine((self,mod,routed_output,device_mesh)),_apply((self,module: nn.Module,device_mesh: DeviceMesh)),__init__((self,tp_mesh: DeviceMesh,ep_mesh: DeviceMesh,))[CTOR,DUNDER],_token_dispatch((self,mod,inputs,device_mesh)),_partition_fn_2d((self,name,mod,ep_tp_mesh)),_token_combine((self,mod,routed_output,device_mesh)),_apply((self,module: nn.Module,device_mesh: DeviceMesh)),expert_parallel((func: Callable)),__init__((self))[CTOR,DUNDER],_prepare_inputput_fn((self,mod,inputs,device_mesh)),_prepare_output_fn((self,mod,outputs,device_mesh)),_apply((self,module: nn.Module,device_mesh: DeviceMesh))] parallel_dims.py→[] pipeline_parallel.py→[build_pipeline_schedule((job_config: JobConfig,stages: list[PipelineStage],loss_fn: Callable))[HOT],stage_ids_this_rank((pp_rank: int,pp_size: int,num_stages: int,style: str = "loop")),generate_llm_fqn_per_model_part((num_stages: int,num_layers: int,input_weight: int = 1,output_weight: int = 1,)),pipeline_module_split((whole_model: nn.Module,pp_mesh: DeviceMesh,pp_schedule: str,device: torch.device,module_names_per_stage: list[list[str]],))] tensor_parallel.py→[maybe_enable_async_tp((job_config: JobConfig,tp_mesh: DeviceMesh))] utils.py→[dist_max((x: torch.Tensor,mesh: DeviceMesh,extra_pg: dist.ProcessGroup | None = None,))→{_dist_reduce},dist_sum((x: torch.Tensor,mesh: DeviceMesh,extra_pg: dist.ProcessGroup | None = None,))→{_dist_reduce},dist_mean((x: torch.Tensor,mesh: DeviceMesh,extra_pg: dist.ProcessGroup | None = None,))→{_dist_reduce},_dist_reduce((x: torch.Tensor,reduceOp: str,mesh: DeviceMesh,extra_pg: dist.ProcessGroup | None,)),set_determinism((world_mesh: DeviceMesh | None,device: torch.device,seed: int | None = None,deterministic: bool = False,distinct_seed_mesh_dim: str = "pp",)),create_context_parallel_ctx((cp_mesh: DeviceMesh,cp_buffers: list[torch.Tensor],cp_seq_dims: list[int],cp_no_restore_buffers: set[torch.Tensor],cp_rotate_method: str,)),get_train_context((enable_loss_parallel: bool,enable_compiled_autograd: bool)),maybe_enable_amp((parallel_dims: ParallelDims,mixed_precision_param: str,device_type: torch.device)),init_distributed((comm_config: CommConfig,enable_cpu_backend: bool = False,base_folder: str = "")),set_pg_timeouts((timeout,world_mesh))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 17 edges
Call: 8 edges

### CROSS_CLUSTER_FLOW

