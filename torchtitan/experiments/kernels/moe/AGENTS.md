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
NODES:38 EDGES:27

## DIRECTORY_TREE
ROOT: torchtitan/experiments/kernels/moe/
└─ unit_tests/ → TST[1]

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:8 CALL_DEPTH:3

combine.py→[test_token_combine(void)[TEST]] dispatch.py→[test_token_dispatch(void)[TEST]] indices.py→[test_with_zero_tokens(void)[TEST]→{indices::generate_permute_indices,indices::generate_permute_indices}] permute_indices_testing.py→[test_fixed_total_experts_varying_ranks((self))[TEST]→{indices::fill_indices_wrapper,indices::fill_indices_cpu},test_different_block_sizes_with_fixed_experts((self))[TEST]→{indices::fill_indices_wrapper,indices::fill_indices_cpu},test_edge_cases_with_fixed_experts((self))[TEST]→{indices::fill_indices_wrapper,indices::fill_indices_cpu},test_max_blocks_with_large_experts((self))[TEST]→{indices::fill_indices_wrapper,indices::fill_indices_cpu},test_extreme_max_blocks_limit((self))[TEST]→{indices::fill_indices_wrapper,indices::fill_indices_cpu}] 
### UTILITY_LAYER
NODES:30 CALL_DEPTH:3

combine.py→[__init__((self,group_name: str,align: int,in_len,out_len,token_shape,num_ranks,num_local_experts,dtype,device: torch.device,))[CTOR,DUNDER],forward((self,inp: torch.Tensor,out: torch.Tensor,in_splits_offsets: torch.Tensor,out_splits_offsets: torch.Tensor,))] dispatch.py→[__init__((self,group_name: str,align: int,in_len,out_len,token_shape,num_ranks,num_local_experts,dtype,device: torch.device,))[CTOR,DUNDER],forward((self,inp: torch.Tensor,out: torch.Tensor,in_splits: torch.Tensor,out_splits_offsets: torch.Tensor,))] indices.py→[generate_permute_indices((tokens_per_expert_group: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,alignment: int,use_cpu: bool = False,))→{fill_indices_wrapper,fill_indices_cpu},simple_test(void)→{generate_permute_indices,generate_permute_indices},fill_indices_wrapper((tokens_per_expert_group: torch.Tensor,start_index_values: torch.Tensor,write_offsets: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,block_size: int = 128,max_blocks: int = 1024,# cap on total number of blocks to launch)),fill_indices_cpu((tokens_per_expert_group: torch.Tensor,start_index_values: torch.Tensor,write_offsets: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,))] permute_indices_testing.py→[fill_indices_cpu((tokens_per_expert_group: torch.Tensor,start_index_values: torch.Tensor,write_offsets: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,)),setUp((self)),create_test_data((self,experts_per_rank: int,num_ranks: int,token_range: Tuple[int,int] = (1,16),alignment: int = 32,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 16 edges
Contains: 11 edges

### CROSS_CLUSTER_FLOW
TESTS→UTILITY_LAYER: 12
UTILITY_LAYER→TESTS: 5

