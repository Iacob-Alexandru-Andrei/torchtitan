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
NODES:61 EDGES:41

## DIRECTORY_TREE
ROOT: torchtitan/experiments/kernels/triton_contiguous_group_gemm/

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:7 CALL_DEPTH:2

debug.py→[test_small(void)[TEST]→{debug::verify_results,cg_reference::pytorch_reference,cg_forward::cg_grouped_gemm_forward,debug::create_aligned_test_data},test_medium(void)[TEST]→{debug::verify_results,cg_reference::pytorch_reference,cg_forward::cg_grouped_gemm_forward,debug::create_aligned_test_data},test_large(void)[TEST]→{debug::verify_results,cg_reference::pytorch_reference,cg_forward::cg_grouped_gemm_forward,debug::create_aligned_test_data}] unit_test_cg.py→[test_forward_deepseek_shapes((self))[TEST],test_backward_deepseek_shapes((self))[TEST],test_forward_performance_deepseek((self))[TEST],test_backward_performance_deepseek((self))[TEST]] 
### UTILITY_LAYER
NODES:54 CALL_DEPTH:3

cg_backward.py→[verify_cg_gemm_backward((M_total=1024,N=512,K=512,num_experts=8,group_size_m=128,device="cuda",atol=1e-1,# Absolute tolerance for validation rtol=1e-1,# Relative tolerance for validation))→{cg_forward::cg_grouped_gemm},cg_grouped_gemm_backward_weights((grad_output: torch.Tensor,# [M_total,N] inputs: torch.Tensor,# [M_total,K] expert_indices: torch.Tensor,# [M_total] num_experts: int,group_size_m: int = 128,)),cg_grouped_gemm_backward_inputs((grad_output: torch.Tensor,# [M_total,N] expert_weights: torch.Tensor,# [num_experts,N,K] expert_indices: torch.Tensor,# [M_total] group_size_m: int = 128,)),cg_grouped_gemm((inputs: torch.Tensor,expert_weights: torch.Tensor,expert_indices: torch.Tensor,group_size_m: int = 128,)),benchmark_cg_gemm_backward((M_total=1024,N=512,K=512,num_experts=8,group_size_m=128,device="cuda",num_runs=10,))] cg_forward.py→[cg_grouped_gemm_forward((inputs: torch.Tensor,# [M_total,K] expert_weights: torch.Tensor,# [num_experts,N,K] expert_indices: torch.Tensor,# [M_total] group_size_m: int = 128,)),cg_grouped_gemm_forward_dynamic((inputs: torch.Tensor,# [M_total,K] expert_weights: torch.Tensor,# [num_experts,N,K] expert_indices: torch.Tensor,# [M_total] group_size_m: int = 128,)),cg_grouped_gemm((inputs: torch.Tensor,expert_weights: torch.Tensor,expert_indices: torch.Tensor,# use_tma: bool = True,group_size_m: int = 128,))] cg_reference.py→[pytorch_reference((inputs: torch.Tensor,expert_weights: torch.Tensor,expert_indices: torch.Tensor,group_size_m: int = 128,))] debug.py→[benchmark_performance(void)→{create_aligned_test_data,cg_reference::pytorch_reference,cg_forward::cg_grouped_gemm_forward,cg_reference::pytorch_reference,cg_forward::cg_grouped_gemm_forward},run_all_tests(void)→{benchmark_performance,debug::test_large,debug::test_medium,debug::test_small},create_aligned_test_data((batch_size: int,seq_len: int,hidden_dim: int,output_dim: int,num_experts: int,group_size_m: int = 128,device: str = "cuda",dtype: torch.dtype = torch.float16,)),pytorch_reference((inputs: torch.Tensor,expert_weights: torch.Tensor,expert_indices: torch.Tensor,group_size_m: int = 128,)),verify_results((output_triton: torch.Tensor,output_reference: torch.Tensor,rtol: float = 1e-2,atol: float = 1e-2,))] tma_cuda_autotune.py→[__init__((self,tma_size: int = 128))[CTOR,DUNDER],init_tma_descriptor((self,name: str)),fill_1d_tma_descriptor((self,name: str,ptr: int,dim: int,block_dim: int,element_size: int)),fill_2d_tma_descriptor((self,name: str,ptr: int,dim1: int,dim0: int,block_dim1: int,block_dim0: int,element_size: int,)),get_tma_descriptor_kernel_param((self,name: str)),early_config_prune((configs,args,**kwargs))] unit_test_cg.py→[benchmark_forward((self,M_total,K,N,num_experts,group_size_m,num_runs=10))→{cg_forward::cg_grouped_gemm_forward,cg_forward::cg_grouped_gemm_forward},benchmark_backward((self,M_total,K,N,num_experts,group_size_m,num_runs=5))→{cg_forward::cg_grouped_gemm,cg_forward::cg_grouped_gemm},verify_forward((self,M_total,K,N,num_experts,group_size_m,print_stats=False))→{cg_forward::cg_grouped_gemm_forward},verify_backward((self,M_total,K,N,num_experts,group_size_m,print_stats=False))→{cg_forward::cg_grouped_gemm},run_tests((run_benchmarks=False))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 13 edges
Call: 28 edges

### CROSS_CLUSTER_FLOW
TESTS→UTILITY_LAYER: 12
UTILITY_LAYER→TESTS: 7

