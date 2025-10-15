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
NODES:282 EDGES:138

## DIRECTORY_TREE
ROOT: torchtitan/experiments/deepseek_v3/
├─ infra/ → U[1]
├─ symm_mem_recipes/ → U[4]
├─ tokenizers/ → U[1]
├─ train_configs/ → U[1]
└─ unit_testing/ → U[4]

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:9 CALL_DEPTH:4

dsgemm_unit_testing.py→[test_m_grouped_gemm_contiguous_with_empty_groups(void)[TEST]→{dsgemm_unit_testing::compute_reference_with_scaling,dsgemm_utils::get_col_major_tma_aligned_tensor,dsgemm_unit_testing::per_block_cast_to_fp8,dsgemm_unit_testing::per_token_cast_to_fp8,dsgemm_unit_testing::create_m_indices_fast},test_m_grouped_gemm_contiguous_all_empty_but_one(void)[TEST]→{dsgemm_unit_testing::compute_reference_with_scaling,dsgemm_utils::get_col_major_tma_aligned_tensor,dsgemm_unit_testing::per_block_cast_to_fp8,dsgemm_unit_testing::per_token_cast_to_fp8,dsgemm_unit_testing::create_m_indices_fast},test_m_grouped_gemm_contiguous_with_scaling_edge_cases(void)[TEST]→{dsgemm_unit_testing::compute_reference_with_scaling,dsgemm_utils::get_col_major_tma_aligned_tensor,dsgemm_unit_testing::per_block_cast_to_fp8,dsgemm_unit_testing::per_token_cast_to_fp8,dsgemm_unit_testing::create_m_indices_fast}] permute_indices_testing.py→[test_fixed_total_experts_varying_ranks((self))[TEST]→{moe_kernels::fill_indices_wrapper,moe_kernels::fill_indices_cpu},test_different_block_sizes_with_fixed_experts((self))[TEST]→{moe_kernels::fill_indices_wrapper,moe_kernels::fill_indices_cpu},test_edge_cases_with_fixed_experts((self))[TEST]→{moe_kernels::fill_indices_wrapper,moe_kernels::fill_indices_cpu},test_max_blocks_with_large_experts((self))[TEST]→{moe_kernels::fill_indices_wrapper,moe_kernels::fill_indices_cpu},test_extreme_max_blocks_limit((self))[TEST]→{moe_kernels::fill_indices_wrapper,moe_kernels::fill_indices_cpu}] test_create_m_indices.py→[test_create_indices_from_offsets_nosync(void)[TEST]→{dsgemm_utils::create_indices_from_offsets_nosync,dsgemm_utils::create_indices_from_offsets_nosync}] 
### UTILITY_LAYER
NODES:273 CALL_DEPTH:5

__init__.py→[] attn_mask_utils.py→[_prepare_4d_causal_attention_mask((attention_mask: Optional[torch.Tensor],input_shape: Union[torch.Size,Tuple,List],inputs_embeds: torch.Tensor,past_key_values_length: int,sliding_window: Optional[int] = None,))] benchmark_kernels.py→[benchmark_quant_kernels((shapes,dtype=torch.bfloat16,warmup=10,iters=100)),print_results_table((results))] checkpoint.py→[load_safetensor_weights((model: torch.nn.Module,weight_map: Dict[str,str],file_location: str,device: torch.device,))→{load_safetensor_file,get_needed_files,permute_indices_testing::setUp,permute_indices_testing::setUp,permute_indices_testing::setUp,permute_indices_testing::setUp},load_weights_from_hf((model: torch.nn.Module,distribution: str,device: torch.device,))→{load_safetensor_weights,get_hf_weight_map_and_path},get_hf_weight_map_and_path((model_id: str,))→{read_weights_from_json},get_needed_files((state_dict: Dict[str,torch.Tensor],weight_map: Dict[str,str]))→{permute_indices_testing::setUp},read_weights_from_json((file_path: str)),load_safetensor_file((full_path: str,device: torch.device))] custom_args.py→[] download.py→[print_usage(void)] dsgemm_kernels.py→[groupwise_activation_quant((x: torch.Tensor,block_size: int = 128,switching_size=2048,))] dsgemm_unit_testing.py→[create_m_indices_fast((m_sizes: torch.Tensor)),per_token_cast_to_fp8((x: torch.Tensor)),per_block_cast_to_fp8((x: torch.Tensor)),compute_reference_with_scaling((lhs: torch.Tensor,lhs_scales: torch.Tensor,rhs: torch.Tensor,rhs_scales: torch.Tensor,m_indices: torch.Tensor,num_groups: int,))[HOT]] dsgemm_utils.py→[make_grouped_layout((num_groups: int,x: torch.Tensor,y: torch.Tensor))→{get_col_major_tma_aligned_tensor,dsgemm_unit_testing::per_token_cast_to_fp8,dsgemm_unit_testing::per_block_cast_to_fp8,dsgemm_unit_testing::per_token_cast_to_fp8},construct_grouped((num_groups: int,x: torch.Tensor,y: torch.Tensor,is_masked: bool))→{get_col_major_tma_aligned_tensor,dsgemm_unit_testing::per_token_cast_to_fp8,dsgemm_unit_testing::per_block_cast_to_fp8,dsgemm_unit_testing::per_token_cast_to_fp8},prepare_fp8_input((x))→{get_col_major_tma_aligned_tensor,dsgemm_unit_testing::per_token_cast_to_fp8},per_block_cast_to_fp8((x: torch.Tensor))→{ceil_div,ceil_div},prepare_fp8_weight((w))→{dsgemm_unit_testing::per_block_cast_to_fp8},get_tma_aligned_size((x: int,element_size: int))→{ceil_div},get_col_major_tma_aligned_tensor((x: torch.Tensor))→{get_tma_aligned_size},compare_fp8_tensors((a: torch.Tensor,b: torch.Tensor)),create_indices_from_offsets_nosync((m_offsets: torch.Tensor)),create_m_indices_from_offsets((m_offsets: torch.Tensor)),create_m_indices_from_sizes((m_sizes: torch.Tensor)),get_m_indices((num_groups: int,m: int)),set_num_sms((num_sms: int)),get_num_sms(void),ceil_div((x: int,y: int)),get_m_alignment_for_contiguous_layout(void),per_token_cast_to_fp8((x: torch.Tensor))] generate.py→[create_model((dist_config: DistConfig))→{checkpoint::load_weights_from_hf},decode((tokenizer,x))→{colorize_chat},colorize_chat((text,user_color=None,assistant_color=None,output_color=None)),create_dist_config((mesh: DeviceMesh)),time_generation((func))] group_gemms.py→
[__init__((self,custom_activation))[CTOR,DUNDER],arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module)),arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module)),arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module)),arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module)),arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module)),arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module)),__init__((self,custom_activation,use_triton_quant=True))[CTOR,DUNDER],arrange_expert_weights((self,all_weights,submod_name,module)),execute((self,contig_tokens,m_sizes,m_offsets,module))] hf_tokenizer.py→[remove_notset_root_handlers(void),__init__((self,tokenizer))[CTOR,DUNDER],encode((self,text,bos=False,eos=False,**kwargs)),__getattr__((self,name))[DUNDER],get_hf_tokenizer((model_id: str))] model.py→[_set_cos_sin_cache((self,seq_len,device,dtype))→{yarn_get_mscale,yarn_get_mscale,yarn_linear_ramp_mask,yarn_find_correction_range},yarn_find_correction_range((low_rot,high_rot,dim,base=10000,max_position_embeddings=2048))→{yarn_find_correction_dim,yarn_find_correction_dim},apply_rotary_pos_emb((q,k,cos,sin,position_ids,unsqueeze_dim=1))→{rotate_half,rotate_half},forward((self,hidden_states: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,position_ids: Optional[torch.LongTensor] = None,))→{apply_rotary_pos_emb,attn_mask_utils::_prepare_4d_causal_attention_mask},__init__((self,config))[CTOR,DUNDER]→{get_group},moe_on_device((self,x,topk_ids,topk_weight))→{moe_kernels::generate_permute_indices},__init__((self,config: ModelArgs,layer_idx: Optional[int] = None))[CTOR,DUNDER]→
{yarn_get_mscale},get_group((dim_name: Optional[str] = None)),__init__((self,hidden_size,eps=1e-6))[CTOR,DUNDER],forward((self,hidden_states)),__init__((self,dim,max_position_embeddings=2048,base=10000,device=None))[CTOR,DUNDER],_set_cos_sin_cache((self,seq_len,device,dtype)),forward((self,x,seq_len=None)),__init__((self,dim,max_position_embeddings=2048,base=10000,device=None,scaling_factor=1.0,))[CTOR,DUNDER],_set_cos_sin_cache((self,seq_len,device,dtype)),__init__((self,dim,max_position_embeddings=2048,base=10000,device=None,scaling_factor=1.0,))[CTOR,DUNDER],_set_cos_sin_cache((self,seq_len,device,dtype)),yarn_find_correction_dim((num_rotations,dim,base=10000,max_position_embeddings=2048)),yarn_get_mscale((scale=1,mscale=1)),yarn_linear_ramp_mask((min,max,dim)),__init__((self,dim,max_position_embeddings=2048,base=10000,device=None,scaling_factor=1.0,original_max_position_embeddings=4096,beta_fast=32,beta_slow=1,mscale=1,mscale_all_dim=0,))[CTOR,DUNDER],rotate_half((x)),__init__((self,config,hidden_size=None,intermediate_size=None))[CTOR,DUNDER],forward((self,x)),__init__((self,config))[CTOR,DUNDER],reset_parameters((self)),forward((self,hidden_states)),combine_experts((self,submod_name: str)),setup_symm_mem((self,dtype: torch.dtype,device: torch.device)),get_send_buf((self)),get_gather_buf((self)),forward((self,hidden_states)),moe_forward((self,x,topk_ids,topk_weight)),sort_tokens((self,x,topk_ids,topk_weights)),_run_group_gemm((self,contig_tokens,m_sizes,m_offsets)),_init_rope((self)),__init__((self,config: ModelArgs,layer_idx: int))[CTOR,DUNDER],forward((self,hidden_states: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,position_ids: Optional[torch.LongTensor] = None,)),__init__((self,config: ModelArgs))[CTOR,DUNDER],_init_weights((self,module)),forward((self,tokens: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,position_ids: Optional[torch.LongTensor] = None,)),__init__((self,config))[CTOR,DUNDER],forward((self,tokens: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,position_ids: Optional[torch.LongTensor] = None,)),prepare_inputs_for_generation((self,input_ids,past_key_values=None,attention_mask=None,**kwargs,)),setup_symm_mem((self,dtype: torch.dtype,device: torch.device))] model_args.py→[] model_config.py→[] moe_kernels.py→[generate_permute_indices((tokens_per_expert_group: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,alignment: int,use_cpu: bool = False,))→{fill_indices_wrapper,fill_indices_cpu},simple_test(void)→{generate_permute_indices,generate_permute_indices},fill_indices_wrapper((tokens_per_expert_group: torch.Tensor,start_index_values: torch.Tensor,write_offsets: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,block_size: int = 128,max_blocks: int = 1024,# cap on total number of blocks to launch)),fill_indices_cpu((tokens_per_expert_group: torch.Tensor,start_index_values: torch.Tensor,write_offsets: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,))] parallelize_deepseek.py→[get_group((dim_name: Optional[str] = None)),parallelize_deepseek((# model: nn.Module,world_mesh: DeviceMesh,device: torch.device,model_args,rank: int,# parallel_dims: ParallelDims,# job_config: JobConfig,))] permute_indices_testing.py→[fill_indices_cpu((tokens_per_expert_group: torch.Tensor,start_index_values: torch.Tensor,write_offsets: torch.Tensor,experts_per_rank: int,num_ranks: int,max_len: int,)),setUp((self)),create_test_data((self,experts_per_rank: int,num_ranks: int,token_range: Tuple[int,int] = (1,16),alignment: int = 32,))] test_create_m_indices.py→[] train_ds_dev.py→[run_full_model((mesh: DeviceMesh,))] train_ds_real.py→[run_full_model((config: JobConfig,))→{next_batch,hf_tokenizer::get_hf_tokenizer,parallelize_deepseek::parallelize_deepseek},cross_entropy_loss((pred: torch.Tensor,labels: torch.Tensor)),next_batch((data_iterator: Iterable,metrics_processor))] triton_barrier.py→[] triton_on_device_all_to_all_v.py→
[_on_device_all_to_all_v((output: torch.Tensor,output_splits: torch.Tensor,input: torch.Tensor,input_splits: torch.Tensor,group: dist.ProcessGroup = dist.group.WORLD,BLOCKS_PER_REMOTE_RANK=8,UNROLL_FACTOR: int = 8,BLOCK_SIZE: int = 16384,))] triton_utils.py→[] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 74 edges
Contains: 64 edges

### CROSS_CLUSTER_FLOW
TESTS→UTILITY_LAYER: 27
UTILITY_LAYER→TESTS: 5

