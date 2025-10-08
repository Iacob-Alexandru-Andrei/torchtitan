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
NODES:64 EDGES:49

## DIRECTORY_TREE
ROOT: torchtitan/experiments/kernels/triton_mg_group_gemm/torchao_pr/

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:9 CALL_DEPTH:3

fast_debug_ao.py→[test_multiple_deepseek_configs(void)[TEST]→{reference_utils::analyze_tensor_differences,reference_utils::analyze_tensor_differences,reference_utils::compute_reference_backward,mg_grouped_gemm::grouped_gemm_backward,reference_utils::analyze_tensor_differences,reference_utils::compute_reference_forward},test_backward_pass(void)[TEST]→{reference_utils::analyze_tensor_differences,reference_utils::analyze_tensor_differences,reference_utils::compute_reference_backward,mg_grouped_gemm::grouped_gemm_backward,mg_grouped_gemm::grouped_gemm_forward},test_forward_pass(void)[TEST]→{reference_utils::analyze_tensor_differences,reference_utils::compute_reference_forward,mg_grouped_gemm::grouped_gemm_forward}] unit_test_backwards.py→[test_mg_dx((self))[TEST]→{reference_utils::analyze_tensor_differences,reference_utils::compute_reference_backward,mg_grouped_gemm::grouped_gemm_backward,mg_grouped_gemm::grouped_gemm_forward},test_mg_dw((self))[TEST]→{reference_utils::analyze_tensor_differences,reference_utils::compute_reference_backward,mg_grouped_gemm::grouped_gemm_backward,mg_grouped_gemm::grouped_gemm_forward},test_mg_grouped_gemm_backward_bf16((self))[TEST],test_mg_grouped_gemm_backward_deepseek_shapes((self))[TEST]] unit_test_forwards.py→[test_mg_grouped_gemm_bf16((self))[TEST],test_mg_grouped_gemm_deepseek_shapes((self))[TEST]] 
### UTILITY_LAYER
NODES:55 CALL_DEPTH:3

__init__.py→[] fast_debug_ao.py→[] mg_grouped_gemm.py→[grouped_gemm_backward((grad_output: torch.Tensor,x: torch.Tensor,w: torch.Tensor,m_sizes: torch.Tensor,use_tma: bool = True,tma_size: int = 128,))→{grouped_gemm_dw_tma,grouped_gemm_dx_tma},grouped_gemm_forward((x: torch.Tensor,w: torch.Tensor,m_sizes: torch.Tensor,tma_size: int = 128,using_fp8: bool = False,)),grouped_gemm_dx_tma((grad_output: torch.Tensor,w: torch.Tensor,m_sizes: torch.Tensor,num_sms: int = 132,tma_size: int = 128,)),grouped_gemm_dw_tma((x: torch.Tensor,grad_output: torch.Tensor,m_sizes: torch.Tensor,num_sms: int = 132,tma_size: int = 128,)),mg_grouped_gemm((x: torch.Tensor,w: torch.Tensor,m_sizes: torch.Tensor,use_tma: bool = True,tma_size: int = 128,using_fp8: bool = False,))] reference_utils.py→[compute_reference_backward((x,w,m_sizes,grad_output))[HOT]→{compute_reference_forward},compute_reference_forward((x,w,m_sizes))[HOT],analyze_tensor_differences((actual,expected,name))] tma_autotuning.py→[__init__((self,tma_size: int = 128))[CTOR,DUNDER],init_tma_descriptor((self,name: str)),fill_1d_tma_descriptor((self,name: str,ptr: int,dim: int,block_dim: int,element_size: int)),fill_2d_tma_descriptor((self,name: str,ptr: int,dim1: int,dim0: int,block_dim1: int,block_dim0: int,element_size: int,)),get_tma_descriptor_kernel_param((self,name: str)),early_config_prune((configs,named_args,dtsize=None,dtype=None,**kwargs))] unit_test_backwards.py→[_run_grouped_gemm_backward_test((self,shape: Tuple[int,int,int,int],device: torch.device,dtype: torch.dtype = torch.bfloat16,atol: float = 1e-5,rtol: float = 1.6e-2,))→{reference_utils::analyze_tensor_differences,reference_utils::analyze_tensor_differences,reference_utils::compute_reference_backward,mg_grouped_gemm::grouped_gemm_backward,reference_utils::analyze_tensor_differences,reference_utils::compute_reference_forward},setUp((self))] unit_test_forwards.py→[_run_grouped_gemm_test((self,shape: Tuple[int,int,int,int],device: torch.device,dtype: torch.dtype = torch.bfloat16,atol: float = 1e-5,rtol: float = 1.6e-2,))→{mg_grouped_gemm::grouped_gemm_forward},setUp((self))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 15 edges
Call: 34 edges

### CROSS_CLUSTER_FLOW
TESTS→UTILITY_LAYER: 23
UTILITY_LAYER→TESTS: 6

