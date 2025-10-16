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
NODES:46 EDGES:19

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/models/mpt_mup/model/

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:46 CALL_DEPTH:1

mup_args.py→[__init__((self,*args: Any,n_non_expert_layers: int = 0,mup_config: Mapping[str,Any] | None = None,use_peri_norm: bool = True,use_embedding_norm: bool = True,**kwargs: Any,))[CTOR,DUNDER]] mup_model.py→[_partial_llama_getattribute((self: PartialLlamaConfig,key: str)),_partial_llama_getitem((self: PartialLlamaConfig,key: str)),__init__((self,vocab_size: int,d_model: int,*,scale: float = 1.0,use_embedding_norm: bool = False,norm_type: str = "low_precision_layernorm",norm_eps: float = 1e-5,device: str | None = None,**kwargs: Any,))[CTOR,DUNDER],forward((self,input: torch.Tensor,unembed: bool = False)),__init__((self,*,depth_multiplier: float = 1.0,depth_alpha_enabled: bool = False,depth_alpha_exp: float = 1.0,use_peri_norm: bool = False,norm_type: str = "low_precision_layernorm",norm_eps: float = 1e-5,**kwargs: Any,))[CTOR,DUNDER],forward((# noqa: PLR0913 self,x: torch.Tensor,past_key_value: tuple[torch.Tensor,torch.Tensor] | None = None,attn_bias: torch.Tensor | None = None,rotary_emb_w_meta_info: dict | None = None,attention_mask: torch.ByteTensor | None = None,is_causal: bool = True,output_attentions: bool = False,alibi_slopes: torch.Tensor | None = None,flash_attn_padding_info: dict[str,torch.Tensor] | None = None,prev_layer_key_value: tuple[torch.Tensor,torch.Tensor] | None = None,key_value_states: torch.Tensor | None = None,x_prev: torch.Tensor | None = None,pos_id_within_seq: torch.Tensor | None = None,)),__init__((self,config: MPTMuPConfig))[CTOR,DUNDER],extract_block_args((self,block_args: dict[str,Any])),forward((self,*args: Any,**kwargs: Any)),__init__((self,config: MPTMuPConfig))[CTOR,DUNDER],reset_parameters((self)),param_init_fn((self,module: nn.Module)),get_optimizer_param_groups((self,optimizer_config: Mapping[str,Any],)),forward((self,input_ids: torch.LongTensor | None = None,attention_mask: torch.ByteTensor | None = None,sequence_id: torch.LongTensor | None = None,labels: torch.LongTensor | None = None,**kwargs: Any,)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None)),forward((self,tokens: torch.Tensor,input_batch: MutableMapping[str,torch.Tensor] | None = None,)),_call_get_param_groups((self,*,lr: float,eps: float,weight_decay: float,)),build_mup_optimizer_overrides((self,*,lr: float,eps: float,weight_decay: float,))[HOT],get_optimizer_param_groups((self,optimizer_config: Mapping[str,Any],))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 19 edges

### CROSS_CLUSTER_FLOW

