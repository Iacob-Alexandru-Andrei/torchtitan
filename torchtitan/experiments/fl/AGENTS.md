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
NODES:184 EDGES:34

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/
├─ configs/ → U[2]
├─ dataloader/ → U[2]
├─ models/ → E[12]
│  ├─ llama3_mup/ → E[9]
│  │  ├─ infra/ → E[2]
│  │  ├─ model/ → E[4]
│  │  ├─ tests/ → E[1]
│  │  └─ train_configs/ → E[1]
│  ├─ mosaic_llama3/ → E[1]
│  └─ mosaic_llama3_mup/ → E[1]
└─ optimizers/ → U[4]

## ARCHITECTURAL_CLUSTERS

### DATA_MODELS
NODES:88 CALL_DEPTH:1

__init__.py→[_get_llama3_mup_spec(void),get_train_spec(void),get_train_spec(void),build_mup_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT],get_train_spec(void)] mup_args.py→[] mup_model.py→[_precompute_freqs_cis((self))[HOT]→{_precompute_freqs_cis},init_weights((self,init_std: float)),init_weights((self,init_std: float)),__init__((self,layer_id: int,model_args: TransformerModelArgs # noqa: ARG002))[CTOR,DUNDER],forward((self,x: torch.Tensor,freqs_cis: torch.Tensor,)),init_weights((self)),__init__((self,model_args: TransformerModelArgs))[CTOR,DUNDER],init_weights((self,buffer_device: torch.device | None = None)),get_optimizer_param_groups((self,optimizer_config: dict[str,Any])),forward((self,tokens: torch.Tensor,input_batch: torch.Tensor | None = None,# noqa: ARG002))] parallelize.py→[parallelize_llama_mup((model: nn.Module,parallel_dims: ParallelDims,job_config: JobConfig,))] state_dict_adapter.py→[__init__((self,model_args: TransformerModelArgs,hf_assets_path: str | None,))[CTOR,DUNDER]] test_mup_model.py→[setUp((self)),test_model_initialization((self))[TEST],test_forward_pass((self))[TEST],test_weight_initialization((self))[TEST]] 
### UTILITY_LAYER
NODES:96 CALL_DEPTH:2

__init__.py→[] adopt.py→[_default_clip_lambda((step: Number | Tensor)),__init__((self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.9,0.9999),eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),_single_tensor_adopt((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1: float,beta2: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,)),_multi_tensor_adopt((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,lr: float | Tensor,clip_lambda: Callable[[Number | Tensor | Any],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,)),_fused_adopt((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,lr: float | Tensor,clip_lambda: Callable[[int],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))] config.py→[] dataloader.py→[__init__((self,*args: Any,**kwargs: Any))[CTOR,DUNDER],__getitem__((self,idx: int))[DUNDER],state_dict((self,num_samples: int | None = None,from_beginning: bool = True)),load_state_dict((self,obj: dict[str,Any])),__init__((self,dataset: StatefulStreamingTextDataset,dp_rank: int,dp_world_size: int,batch_size: int,collate_fn: Callable | None = None,num_workers: int = 0,prefetch_factor: int | None = 2,pin_memory: bool = True,persistent_workers: bool = True,drop_last: bool = True,))[CTOR,DUNDER],state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),titan_collate_fn((batch: list[Any])),build_mosaic_dataloader((*,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: MosaicJobConfig,))[HOT]] optimizer_builder.py→[build_mosaic_optimizers((model_parts: list[torch.nn.Module],optimizer_config: MosaicOptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,param_groups: list[dict[str,Any]] | None = None,))[HOT]] optimizers.py→[] qhadamw.py→
[__init__((self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.9,0.95),v1: float = 0.7,eps: float = 1e-8,weight_decay: float = 1e-5,*,amsgrad: bool = False,decouple: bool = True,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),_single_tensor_qhadamw((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,amsgrad: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,)),_multi_tensor_qhadamw((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,amsgrad: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,)),_fused_qhadamw((*args,**kwargs,))] qhadopt.py→[_default_clip_lambda((step: Number | Tensor)),__init__((self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.999,0.9999),v1: float = 0.9,eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),_single_tensor_qhadopt((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,)),_multi_tensor_qhadopt((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,clip_lambda: Callable[[Number | Tensor | Any],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,)),_fused_qhadopt((params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,clip_lambda: Callable[[int],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))] tokenizer.py→[build_mosaic_tokenizer((job_config: MosaicJobConfig,))[HOT]] train.py→[main(void)[ENTRY]→{__init__::get_train_spec,__init__::get_train_spec}] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 31 edges
Call: 3 edges

### CROSS_CLUSTER_FLOW
UTILITY_LAYER→DATA_MODELS: 2

