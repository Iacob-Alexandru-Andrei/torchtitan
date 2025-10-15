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
NODES:176 EDGES:80

## DIRECTORY_TREE
ROOT: torchtitan/experiments/multimodal/
├─ tests/ → TST[2]
└─ tokenizer/ → U[1]

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:12 CALL_DEPTH:1

test_multimodal_model.py→[test_llama_mm_vision_encoder((self))[TEST]] test_utils.py→[fixed_init_tensor((shape: torch.Size,min_val: Union[float,int] = 0.0,max_val: Union[float,int] = 1.0,nonlinear: bool = False,dtype: torch.dtype = torch.float,))] 
### UTILITY_LAYER
NODES:164 CALL_DEPTH:3

__init__.py→[] check_padding_mm.py→[] mm_collator.py→[padded_collate((batch: List[Dict[str,List[int]]],padding_idx: int = 0,ignore_idx: int = -100,))] mm_dataset.py→[_process_obelics_sample((sample: dict[str,Any],image_token: str = "<|image|>"))→{utils::load_image},__init__((self,dataset_name: str,dataset_path: Optional[str],tokenizer: BaseTokenizer,image_token: str = "<|image|>",tile_size: int = 448,max_num_tiles: int = 4,seq_len: int = 2048,dp_rank: int = 0,dp_world_size: int = 1,infinite: bool = False,))[CTOR,DUNDER]→{_validate_mm_dataset},_load_obelics_dataset((dataset_path: str)),_validate_mm_dataset((dataset_name: str,dataset_path: str = None)),__iter__((self))[DUNDER],_get_data_iter((self)),load_state_dict((self,state_dict)),state_dict((self)),build_mm_dataloader((dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: JobConfig,infinite: bool = True,))[HOT]] model.py→[forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,))→{repeat_kv,repeat_kv,apply_rotary_emb},forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,))→{repeat_kv,repeat_kv,apply_rotary_emb},forward((self,x: torch.Tensor,encoder_input: torch.Tensor,mask: Optional[torch.Tensor] = None,))→{repeat_kv,repeat_kv},apply_rotary_emb((xq: torch.Tensor,xk: torch.Tensor,freqs_cis: torch.Tensor,))→{reshape_for_broadcast},_precompute_freqs_cis((self,model_args))[HOT]→
{precompute_freqs_cis},__init__((self,*args: Any,**kwargs: Any))[CTOR,DUNDER],forward((self,x: torch.Tensor)),precompute_freqs_cis((dim: int,end: int,theta: float = 10000.0))[HOT],reshape_for_broadcast((freqs_cis: torch.Tensor,x: torch.Tensor)),repeat_kv((x: torch.Tensor,num_rep: int)),__init__((self,model_args: ModelArgs))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,dim: int,hidden_dim: int,multiple_of: int,ffn_dim_multiplier: Optional[float],activation: nn.Module = nn.SiLUvoid,))[CTOR,DUNDER],forward((self,x)),init_weights((self,init_std: float)),__init__((self))[CTOR,DUNDER],forward((self,x: torch.Tensor)),__init__((self,max_num_tiles: int,emb_dim: int,))[CTOR,DUNDER],forward((self,x: torch.Tensor,aspect_ratio: torch.Tensor)),__init__((self,emb_dim: int,tile_size: int,patch_size: int))[CTOR,DUNDER],forward((self,x: torch.Tensor,*args: Tuple[Any])),__init__((self,max_num_tiles: int,emb_dim: int,tile_size: int,patch_size: int))[CTOR,DUNDER],forward((self,x: torch.Tensor,aspect_ratio: torch.Tensor)),__init__((self,in_channels: int,out_channels: int,kernel_size: int,stride: int,bias: bool = False,))[CTOR,DUNDER],forward((self,x: torch.Tensor)),__init__((self,model_args: ModelArgs,attn_scale: Optional[nn.Module] = None,mlp_scale: Optional[nn.Module] = None,))[CTOR,DUNDER],forward((self,x: torch.Tensor,mask: Optional[torch.Tensor] = None,)),__init__((self,emb_dim: int))[CTOR,DUNDER],forward((self,x: torch.Tensor)),__init__((self,model_args: ModelArgs,))[CTOR,DUNDER],forward((self,images: torch.Tensor,aspect_ratio: Optional[torch.Tensor] = None)),__init__((self,model_args: ModelArgs,))[CTOR,DUNDER],forward((self,x: torch.Tensor,hidden_states: Optional[List[torch.Tensor]] = None,)),__init__((self,model_args: ModelArgs))[CTOR,DUNDER],forward((self,images: torch.Tensor,aspect_ratio: Optional[torch.Tensor] = None)),__init__((self,dim: int,hidden_dim: int,multiple_of: int,ffn_dim_multiplier: Optional[float],))[CTOR,DUNDER],forward((self,x)),init_weights((self,init_std: float)),__init__((self,model_args: ModelArgs))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,model_args: ModelArgs))[CTOR,DUNDER],init_weights((self,init_std: float)),__init__((self,model_args: ModelArgs,))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,**kwargs: Dict,)),__init__((self,model_args: ModelArgs,))[CTOR,DUNDER],_skip_mask((self,mask: Optional[torch.Tensor])),forward((self,x: torch.Tensor,*,encoder_input: Optional[torch.Tensor] = None,encoder_mask: Optional[torch.Tensor] = None,**kwargs: Dict,)),__init__((self,layer: nn.Module,fusion_layer: nn.Module,fusion_first: bool = True))[CTOR,DUNDER],forward((self,x: torch.Tensor,**kwargs: Dict)),__init__((self,vocab_size: int,fusion_vocab_size: int,embed_dim: int))[CTOR,DUNDER],forward((self,input: torch.Tensor)),__init__((self,model_args: ModelArgs))[CTOR,DUNDER],forward((self,tokens: torch.Tensor,*,encoder_input: Optional[torch.Tensor] = None,encoder_mask: Optional[torch.Tensor] = None,))] tiktoken.py→[__init__((self,model_path: str))[CTOR,DUNDER],encode((self,s: str,*,bos: bool,eos: bool,allowed_special: Optional[Union[Literal["all"],AbstractSet[str]]] = None,disallowed_special: Optional[Union[Literal["all"],Collection[str]]] = None,)),decode((self,t: Sequence[int])),encode_multimodal((self,sample: Mapping[str,Any])),build_tiktoken_tokenizer((job_config: JobConfig))[HOT]] transform.py→[__call__((self,image: torch.Tensor))[DUNDER]→{utils::tile_crop,utils::resize_with_pad,utils::get_canvas_best_fit},__init__((self,*,image_mean: Optional[List[float]] = None,image_std: Optional[List[float]] = None,possible_resolutions: Optional[List[Tuple[int,int]]] = None,tile_size: int = 224,max_num_tiles: Optional[int] = 4,dtype: torch.dtype = torch.bfloat16,resample: str = "bilinear",resize_to_max_canvas: bool = False,))[CTOR,DUNDER]→{utils::find_supported_resolutions}] utils.py→
[resize_with_pad((image: torch.Tensor,target_size: Tuple[int,int],resample: torchvision.transforms.InterpolationMode,max_size: Optional[int] = None,))→{_pad_image_top_left,_get_max_res_without_distortion},find_supported_resolutions((max_num_tiles: int,tile_size: int))→{_get_factors},tile_crop((image: torch.Tensor,tile_size: int)),_pad_image_top_left((image: torch.Tensor,target_size: Tuple[int,int],)),_get_max_res_without_distortion((image_size: Tuple[int,int],target_size: Tuple[int,int],)),_get_factors((n: int)),get_canvas_best_fit((image: torch.Tensor,possible_resolutions: torch.Tensor,resize_to_max_canvas: bool)),load_image((image_loc: Union[Path,str]))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 19 edges
Contains: 61 edges

### CROSS_CLUSTER_FLOW

