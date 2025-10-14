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
NODES:293 EDGES:121

## DIRECTORY_TREE
ROOT: torchtitan/experiments/flux/
├─ dataset/ → U[2]
├─ inference/ → U[1]
├─ infra/ → U[1]
├─ model/ → U[7]
├─ scripts/ → U[1]
└─ tests/ → TST[3]
   ├─ assets/ → TST[1]
   │  └─ cc12m_test/ → TST[1]
   └─ unit_tests/ → TST[1]

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:23 CALL_DEPTH:3

integration_tests.py→[run_single_test((test_flavor: OverrideDefinitions,full_path: str,output_dir: str))→{main},run_tests((args,test_list: list[OverrideDefinitions]))→{run_single_test},main(void)→{run_tests},build_flux_test_list(void)[HOT]] pack_test_dataset.py→[pack_wds_dataset((tar_destination,source_folder,number_of_samples))→{math::rope,math::rope,math::rope,math::rope}] test_flux_dataloader.py→[test_load_dataset((self))[TEST]→{flux_dataset::build_flux_dataloader,flux_dataset::build_flux_dataloader},setUp((self)),tearDown((self))] 
### UTILITY_LAYER
NODES:270 CALL_DEPTH:4

__init__.py→[get_train_spec(void)] args.py→[] autoencoder.py→[forward((self,x))→{swish,swish},__init__((self,params: AutoEncoderParams))[CTOR,DUNDER]→{tokenizer::decode,tokenizer::encode},forward((self,x: Tensor))→{swish},forward((self,z: Tensor))→{swish},swish((x: Tensor)),__init__((self,in_channels: int))[CTOR,DUNDER],attention((self,h_: Tensor)),forward((self,x: Tensor)),__init__((self,in_channels: int,out_channels: int))[CTOR,DUNDER],__init__((self,in_channels: int))[CTOR,DUNDER],forward((self,x: Tensor)),__init__((self,in_channels: int))[CTOR,DUNDER],forward((self,x: Tensor)),__init__((self,resolution: int,in_channels: int,ch: int,ch_mult: list[int],num_res_blocks: int,z_channels: int,))[CTOR,DUNDER],__init__((self,ch: int,out_ch: int,ch_mult: list[int],num_res_blocks: int,in_channels: int,resolution: int,z_channels: int,))[CTOR,DUNDER],__init__((self,sample: bool = True,chunk_dim: int = 1))[CTOR,DUNDER],forward((self,z: Tensor)),encode((self,x: Tensor)),decode((self,z: Tensor)),forward((self,x: Tensor)),load_ae((ckpt_path: str,autoencoder_params: AutoEncoderParams,device: str | torch.device = "cuda",dtype=torch.bfloat16,random_init=False,))] download_autoencoder.py→[hf_download((repo_id: str,file_path: str,local_dir: str,hf_token: Optional[str] = None))] flux_dataset.py→[_cc12m_wds_data_processor((sample: dict[str,Any],t5_tokenizer: FluxTokenizer,clip_tokenizer: FluxTokenizer,output_size: int = 256,))→{_process_cc12m_image},_coco_data_processor((sample: dict[str,Any],t5_tokenizer: FluxTokenizer,clip_tokenizer: FluxTokenizer,output_size: int = 256,))→{_process_cc12m_image},__init__((self,dataset_name: str,dataset_path: Optional[str],t5_tokenizer: BaseTokenizer,clip_tokenizer: BaseTokenizer,job_config: Optional[JobConfig] = None,dp_rank: int = 0,dp_world_size: int = 1,infinite: bool = False,))[CTOR,DUNDER]→{_validate_dataset},build_flux_dataloader((dp_world_size: int,dp_rank: int,job_config: JobConfig,# This parameter is not used,keep it for compatibility tokenizer: FluxTokenizer | None,infinite: bool = True,))[HOT]→{tokenizer::build_flux_tokenizer},build_flux_validation_dataloader((dp_world_size: int,dp_rank: int,job_config: JobConfig,# This parameter is not used,keep it for compatibility tokenizer: BaseTokenizer | None,generate_timestamps: bool = True,infinite: bool = False,))[HOT]→{tokenizer::build_flux_tokenizer},_process_cc12m_image((img: PIL.Image.Image,output_size: int = 256,)),_validate_dataset((dataset_name: str,dataset_path: Optional[str] = None)),_get_data_iter((self)),__iter__((self))[DUNDER],load_state_dict((self,state_dict)),state_dict((self)),__init__((self,dataset_name: str,dataset_path: Optional[str],t5_tokenizer: BaseTokenizer,clip_tokenizer: BaseTokenizer,job_config: Optional[JobConfig] = None,dp_rank: int = 0,dp_world_size: int = 1,generate_timesteps: bool = True,infinite: bool = False,))[CTOR,DUNDER],__iter__((self))[DUNDER]] hf_embedder.py→[__init__((self,version: str,random_init=False,**hf_kwargs))[CTOR,DUNDER],forward((self,batch_tokens: Tensor))] infer.py→[] job_config.py→[] layers.py→[forward((self,ids: Tensor))→{math::rope},forward((self,x: Tensor,pe: Tensor))→{math::attention},forward((self,img: Tensor,txt: Tensor,vec: Tensor,pe: Tensor))→{math::attention},forward((self,x: Tensor,vec: Tensor,pe: Tensor))→
{math::attention},__init__((self,dim: int,theta: int,axes_dim: list[int]))[CTOR,DUNDER],timestep_embedding((t: Tensor,dim,max_period=10000,time_factor: float = 1000.0)),__init__((self,in_dim: int,hidden_dim: int))[CTOR,DUNDER],init_weights((self,init_std: float = 0.02)),forward((self,x: Tensor)),__init__((self,dim: int))[CTOR,DUNDER],init_weights((self)),forward((self,q: Tensor,k: Tensor,v: Tensor)),__init__((self,dim: int,num_heads: int = 8,qkv_bias: bool = False))[CTOR,DUNDER],init_weights((self)),__init__((self,dim: int,double: bool))[CTOR,DUNDER],init_weights((self)),forward((self,vec: Tensor)),__init__((self,hidden_size: int,num_heads: int,mlp_ratio: float,qkv_bias: bool = False))[CTOR,DUNDER],init_weights((self)),__init__((self,hidden_size: int,num_heads: int,mlp_ratio: float = 4.0,qk_scale: float | None = None,))[CTOR,DUNDER],init_weights((self)),__init__((self,hidden_size: int,patch_size: int,out_channels: int))[CTOR,DUNDER],init_weights((self)),forward((self,x: Tensor,vec: Tensor))] loss.py→[mse_loss((pred: torch.Tensor,labels: torch.Tensor)),build_mse_loss((job_config: JobConfig))[HOT]] math.py→[attention((q: Tensor,k: Tensor,v: Tensor,pe: Tensor))→{apply_rope},rope((pos: Tensor,dim: int,theta: int)),apply_rope((xq: Tensor,xk: Tensor,freqs_cis: Tensor))] model.py→[forward((self,img: Tensor,img_ids: Tensor,txt: Tensor,txt_ids: Tensor,timesteps: Tensor,y: Tensor,))→{layers::timestep_embedding},__init__((self,model_args: FluxModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device=None))] parallelize.py→[parallelize_flux((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))→{apply_fsdp,apply_ac},apply_fsdp((model: nn.Module,dp_mesh: DeviceMesh,param_dtype: torch.dtype,reduce_dtype: torch.dtype,cpu_offload: bool = False,)),apply_ac((model: nn.Module,ac_config)),parallelize_encoders((t5_model: nn.Module,clip_model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))] sampling.py→[denoise((device: torch.device,dtype: torch.dtype,model: FluxModel,img_width: int,img_height: int,denoising_steps: int,clip_encodings: torch.Tensor,t5_encodings: torch.Tensor,enable_classifier_free_guidance: bool = False,empty_t5_encodings: torch.Tensor | None = None,empty_clip_encodings: torch.Tensor | None = None,classifier_free_guidance_scale: float | None = None,))→{get_schedule,utils::unpack_latents,utils::pack_latents,utils::create_position_encoding_for_latents,utils::generate_noise_latent},generate_image((device: torch.device,dtype: torch.dtype,job_config: JobConfig,model: FluxModel,prompt: str | list[str],autoencoder: AutoEncoder,t5_tokenizer: BaseTokenizer,clip_tokenizer: BaseTokenizer,t5_encoder: FluxEmbedder,clip_encoder: FluxEmbedder,))→{denoise,utils::preprocess_data,utils::preprocess_data},get_schedule((num_steps: int,image_seq_len: int,base_shift: float = 0.5,max_shift: float = 1.15,shift: bool = True,))→{time_shift,get_lin_function},time_shift((mu: float,sigma: float,t: Tensor)),get_lin_function((x1: float = 256,y1: float = 0.5,x2: float = 4096,y2: float = 1.15)),save_image((name: str,output_dir: str,x: torch.Tensor,add_sampling_metadata: bool,prompt: str,))] state_dict_adapter.py→[__init__((self,model_args: FluxModelArgs,hf_assets_path: str | None))[CTOR,DUNDER]→{math::rope},_swap_scale_shift((self,weight)),to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] tokenizer.py→[__init__((self,model_path: str = "t5-small",max_length: int = 77,**hf_kwargs))[CTOR,DUNDER],_pad_and_chunk_tokens((self,tokens: List[int],max_length: int,pad_token: int)),get_vocab_size((self)),encode((self,text: str | list[str])),decode((self,t: List[int])),__init__((self,model_path: str = "t5-small",max_length: int = 77,**hf_kwargs))[CTOR,DUNDER],get_vocab_size((self)),encode((self,s: str | list[str],)),decode((self,t: List[int])),build_flux_tokenizer((job_config: JobConfig))[HOT]] train.py→
[forward_backward_step((self,input_dict: dict[str,torch.Tensor],labels: torch.Tensor))→{utils::unpack_latents,utils::pack_latents,utils::create_position_encoding_for_latents,utils::preprocess_data},__init__((self,job_config: JobConfig))[CTOR,DUNDER]→{parallelize::parallelize_encoders,autoencoder::load_ae}] utils.py→[preprocess_data((# arguments from the recipe device: torch.device,dtype: torch.dtype,*,# arguments from the config autoencoder: Optional[AutoEncoder],clip_encoder: FluxEmbedder,t5_encoder: FluxEmbedder,batch: dict[str,Tensor],)),generate_noise_latent((bsz: int,height: int,width: int,device: str | torch.device,dtype: torch.dtype,seed: int | None = None,)),create_position_encoding_for_latents((bsz: int,latent_height: int,latent_width: int,position_dim: int = 3)),pack_latents((x: Tensor)),unpack_latents((x: Tensor,latent_height: int,latent_width: int))] validate.py→[__init__((self,job_config: JobConfig,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,parallel_dims: ParallelDims,loss_fn: LossFunction,validation_context: Generator[None,None,None],maybe_enable_amp: Generator[None,None,None],metrics_processor: MetricsProcessor | None = None,pp_schedule: _PipelineSchedule | None = None,pp_has_first_stage: bool | None = None,pp_has_last_stage: bool | None = None,))[CTOR,DUNDER]→{tokenizer::build_flux_tokenizer,flux_dataset::build_flux_validation_dataloader},flux_init((self,device: torch.device,_dtype: torch.dtype,autoencoder: AutoEncoder,t5_encoder: FluxEmbedder,clip_encoder: FluxEmbedder,)),build_flux_validator((job_config: JobConfig,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,parallel_dims: ParallelDims,loss_fn: LossFunction,validation_context: Generator[None,None,None],maybe_enable_amp: Generator[None,None,None],metrics_processor: MetricsProcessor | None = None,pp_schedule: _PipelineSchedule | None = None,pp_has_first_stage: bool | None = None,pp_has_last_stage: bool | None = None,))[HOT]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 47 edges
Contains: 74 edges

### CROSS_CLUSTER_FLOW
TESTS→UTILITY_LAYER: 6

