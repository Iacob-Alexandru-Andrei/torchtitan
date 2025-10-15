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
NODES:113 EDGES:64

## DIRECTORY_TREE
ROOT: torchtitan/experiments/flux/model/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:113 CALL_DEPTH:3

args.py→[] autoencoder.py→[forward((self,x))→{swish,swish},__init__((self,params: AutoEncoderParams))[CTOR,DUNDER]→{decode,encode},forward((self,x: Tensor))→{swish},forward((self,z: Tensor))→{swish},swish((x: Tensor)),__init__((self,in_channels: int))[CTOR,DUNDER],attention((self,h_: Tensor)),forward((self,x: Tensor)),__init__((self,in_channels: int,out_channels: int))[CTOR,DUNDER],__init__((self,in_channels: int))[CTOR,DUNDER],forward((self,x: Tensor)),__init__((self,in_channels: int))[CTOR,DUNDER],forward((self,x: Tensor)),__init__((self,resolution: int,in_channels: int,ch: int,ch_mult: list[int],num_res_blocks: int,z_channels: int,))[CTOR,DUNDER],__init__((self,ch: int,out_ch: int,ch_mult: list[int],num_res_blocks: int,in_channels: int,resolution: int,z_channels: int,))[CTOR,DUNDER],__init__((self,sample: bool = True,chunk_dim: int = 1))[CTOR,DUNDER],forward((self,z: Tensor)),encode((self,x: Tensor)),decode((self,z: Tensor)),forward((self,x: Tensor)),load_ae((ckpt_path: str,autoencoder_params: AutoEncoderParams,device: str | torch.device = "cuda",dtype=torch.bfloat16,random_init=False,))] hf_embedder.py→[__init__((self,version: str,random_init=False,**hf_kwargs))[CTOR,DUNDER],forward((self,batch_tokens: Tensor))] layers.py→[forward((self,ids: Tensor))→{math::rope},forward((self,x: Tensor,pe: Tensor))→{math::attention},forward((self,img: Tensor,txt: Tensor,vec: Tensor,pe: Tensor))→{math::attention},forward((self,x: Tensor,vec: Tensor,pe: Tensor))→{math::attention},__init__((self,dim: int,theta: int,axes_dim: list[int]))[CTOR,DUNDER],timestep_embedding((t: Tensor,dim,max_period=10000,time_factor: float = 1000.0)),__init__((self,in_dim: int,hidden_dim: int))[CTOR,DUNDER],init_weights((self,init_std: float = 0.02)),forward((self,x: Tensor)),__init__((self,dim: int))[CTOR,DUNDER],init_weights((self)),forward((self,q: Tensor,k: Tensor,v: Tensor)),__init__((self,dim: int,num_heads: int = 8,qkv_bias: bool = False))[CTOR,DUNDER],init_weights((self)),__init__((self,dim: int,double: bool))[CTOR,DUNDER],init_weights((self)),forward((self,vec: Tensor)),__init__((self,hidden_size: int,num_heads: int,mlp_ratio: float,qkv_bias: bool = False))[CTOR,DUNDER],init_weights((self)),__init__((self,hidden_size: int,num_heads: int,mlp_ratio: float = 4.0,qk_scale: float | None = None,))[CTOR,DUNDER],init_weights((self)),__init__((self,hidden_size: int,patch_size: int,out_channels: int))[CTOR,DUNDER],init_weights((self)),forward((self,x: Tensor,vec: Tensor))] math.py→[attention((q: Tensor,k: Tensor,v: Tensor,pe: Tensor))→{apply_rope},rope((pos: Tensor,dim: int,theta: int)),apply_rope((xq: Tensor,xk: Tensor,freqs_cis: Tensor))] model.py→[forward((self,img: Tensor,img_ids: Tensor,txt: Tensor,txt_ids: Tensor,timesteps: Tensor,y: Tensor,))→{layers::timestep_embedding},__init__((self,model_args: FluxModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device=None))] state_dict_adapter.py→[__init__((self,model_args: FluxModelArgs,hf_assets_path: str | None))[CTOR,DUNDER]→{math::rope},_swap_scale_shift((self,weight)),to_hf((self,state_dict: dict[str,Any])),from_hf((self,hf_state_dict: dict[str,Any]))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 13 edges
Contains: 51 edges

### CROSS_CLUSTER_FLOW

