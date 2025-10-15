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
NODES:316 EDGES:127

## DIRECTORY_TREE
ROOT: torchtitan/components/
├─ ft/ → U[7]
│  ├─ config/ → U[2]
│  └─ diloco/ → U[3]
└─ quantization/ → U[4]

## ARCHITECTURAL_CLUSTERS

### UI_COMPONENTS
NODES:316 CALL_DEPTH:3

__init__.py→[__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER]] checkpoint.py→[__init__((self,model: nn.Module | list[nn.Module]))[CTOR,DUNDER],_get_state_dict((self)),state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),purge_thread((purge_queue: queue.Queue)),__init__((self,dataloader: BaseDataLoader | None,model_parts: list[nn.Module],optimizers: OptimizersContainer,lr_schedulers: LRSchedulersContainer,states: dict[str,Any],checkpoint_config: CheckpointConfig,sd_adapter: BaseStateDictAdapter | None,base_folder: str = "",ft_manager: FTManager | None = None,))[CTOR,DUNDER],__del__((self))[DUNDER],close((self)),dcp_load((self,state_dict: dict[str,Any],checkpoint_id: str,from_hf: bool,)),maybe_wait_for_staging((self)),_find_load_step((self,folder: str = "")),_ft_folder((self)),_create_checkpoint_id((self,step: int,folder: str = "")),_ft_save((self,step: int)),_ft_load((self)),_flattened_model_states_sd((self,state_dict: dict[str,Any] | None = None)),_states_to_load((self,model_only: bool)),_save_last_step((self,curr_step: int)),_should_save((self,curr_step: int,last_step: bool = False)),_async_wait((self)),_purge_stale_checkpoints((self))] dataloader.py→[__init__((self,dataset: IterableDataset,dp_rank: int,dp_world_size: int,batch_size: int,collate_fn: Callable | None = None,))[CTOR,DUNDER],state_dict((self)),load_state_dict((self,state_dict: dict[str,Any]))] float8.py→[__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER],_init_filter_fn((self,float8_config: Float8Linear)),convert((self,model: nn.Module)),post_optimizer_hook((self,model: nn.Module | list[nn.Module])),__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER],convert((self,model: nn.Module)),post_optimizer_hook((self,model: nn.Module | list[nn.Module]))] job_config.py→[] loss.py→[rescale_accumulated_loss((unwrapped_loss_fn,accumulation_steps))→{rescale_accumulated_loss},cross_entropy_loss((pred: torch.Tensor,labels: torch.Tensor)),build_cross_entropy_loss((job_config: JobConfig))[HOT],__init__((self,unwrapped_loss_fn,accumulation_steps))[CTOR,DUNDER],__call__((self,*args,**kwargs))[DUNDER]] lr_scheduler.py→[__iter__((self))[DUNDER]→{step},__init__((self,optimizers: OptimizersContainer,lr_lambda: Callable))[CTOR,DUNDER],__len__((self))[DUNDER],step((self)),state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),build_lr_schedulers((optimizers: OptimizersContainer,lr_scheduler_config: LRSchedulerConfig,training_steps: int,))[HOT]] manager.py→[__init__((self,ft_config: FTConfig,))[CTOR,DUNDER],get_dp_info((self,dp_degree: int,dp_rank: int)),maybe_set_all_reduce_hook((self,model_parts: list[torch.nn.Module])),maybe_semi_sync_training((ft_config: FTConfig,ft_manager: FTManager,model: torch.nn.Module,n_layers: int,optimizer: torch.optim.Optimizer,fragment_fn: Optional[Callable[...,list[nn.Module]]] = None,))] metrics.py→[_build_metric_logger((job_config: JobConfig,parallel_dims: ParallelDims,tag: str | None = None))[HOT]→{_get_metrics_rank,lr_scheduler::step},__init__((self,job_config: JobConfig,parallel_dims: ParallelDims,tag: str | None = None,))[CTOR,DUNDER]→{build_device_memory_monitor,_build_metric_logger},ensure_pp_loss_visible((parallel_dims: ParallelDims,job_config: JobConfig,color: Color))→
{lr_scheduler::step},__init__((self,device: str = f"{device_type}:0"))[CTOR,DUNDER],_to_gib((self,memory_in_bytes)),_to_pct((self,memory)),get_peak_stats((self)),reset_peak_stats((self)),build_device_memory_monitor(void)[HOT],log((self,metrics: dict[str,Any],step: int)),close((self)),__init__((self,log_dir: str,tag: str | None = None))[CTOR,DUNDER],log((self,metrics: dict[str,Any],step: int)),close((self)),__init__((self,log_dir: str,job_config: JobConfig,tag: str | None = None))[CTOR,DUNDER],log((self,metrics: dict[str,Any],step: int)),close((self)),__init__((self))[CTOR,DUNDER],add_logger((self,logger_instance: BaseLogger)),log((self,metrics: dict[str,Any],step: int)),close((self)),_get_metrics_rank((parallel_dims: ParallelDims,job_config: JobConfig,)),should_log((self,step: int)),log((self,step: int,global_avg_loss: float,global_max_loss: float,grad_norm: float,extra_metrics: dict[str,Any] | None = None,)),log_validation((self,loss: float,step: int)),close((self)),build_metrics_processor((job_config: JobConfig,parallel_dims: ParallelDims,model_args: "BaseModelArgs | None" = None,tag: str | None = None,))[HOT]] mx.py→[__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER],convert((self,model: nn.Module)),post_optimizer_hook((self,model: nn.Module | list[nn.Module])),__init__((self,job_config: JobConfig,parallel_dims: ParallelDims))[CTOR,DUNDER],convert((self,model: nn.Module)),post_optimizer_hook((self,model: nn.Module | list[nn.Module]))] optimizer.py→[__iter__((self))[DUNDER]→{lr_scheduler::step},__iter__((self))[DUNDER]→{lr_scheduler::step},build_optimizers_with_moe_load_balancing((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,))[HOT]→{build_optimizers},__init__((self,optimizers: list[Optimizer]))[CTOR,DUNDER],refresh((self)),_get_owner((self,param: nn.Parameter)),__getitem__((self,key: nn.Parameter))[DUNDER],__setitem__((self,key: nn.Parameter,value: dict[str,Any]))[DUNDER],setdefault((self,key: nn.Parameter,default: dict[str,Any] | None = None,)),__delitem__((self,key: nn.Parameter))[DUNDER],__contains__((self,key: object))[DUNDER],__len__((self))[DUNDER],clear((self)),__init__((self,model_parts: list[nn.Module],optimizer_cls: type[T],optimizer_kwargs: dict[str,Any],param_groups: list[dict[str,Any]] | None = None,))[CTOR,DUNDER],__len__((self))[DUNDER],zero_grad((self,*args,**kwargs)),state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),_validate_length((self,expected_length: int)),_post_init((self,all_params: list[nn.Parameter],optimizer_kwargs: dict[str,Any],param_groups: list[dict[str,Any]] | None = None,)),_refresh_views((self)),__init__((self,model_parts: list[nn.Module],optimizer_cls: type[T],optimizer_kwargs: dict[str,Any],))[CTOR,DUNDER],step((self)),zero_grad((self)),__init__((self,model_parts: list[nn.Module],optimizer_cls: type[T],optimizer_kwargs: dict[str,Any],ft_manager: "ft.Manager",use_ft_optimizer: bool = True,param_groups: list[dict[str,Any]] | None = None,))[CTOR,DUNDER],init_cache_state_dict((self)),state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),zero_grad((self,*args,**kwargs)),build_optimizers((model_parts: list[nn.Module],optimizer_config: OptimizerConfig,parallel_dims: ParallelDims,ft_manager: FTManager | None = None,param_groups: list[dict[str,Any]] | None = None,))[HOT]] protocol.py→[] tokenizer.py→
[__init__((self))[CTOR,DUNDER],__init__((self,tokenizer_path: str,))[CTOR,DUNDER],_load_config((self,config_path: str)),_load_tokenizer_from_path((self,tokenizer_path: str)),_get_token_from_config((self,config: dict[str,Any],key: str)),_process_special_token((self,token_str: str,token_config: dict,token_id: Optional[int] = None)),_infer_special_tokens((self)),_infer_should_add_bos_eos((self)),encode((self,*args,**kwargs)),get_vocab_size((self)),get_vocab((self)),token_to_id((self,token: str)),id_to_token((self,token_id: int)),build_hf_tokenizer((job_config: JobConfig,))[HOT]] utils.py→[fragment_llm((model: nn.Module,ft_config: FTConfig,n_layers: int,))→{module_split},module_filter_fn((mod: nn.Module,fqn: str,filter_fqns: list[str])),module_split((model: nn.Module,module_fqns_per_model_fragment: list[list[str]],))] validate.py→[build_validator((job_config: JobConfig,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,parallel_dims: ParallelDims,loss_fn: LossFunction,validation_context: Generator[None,None,None],maybe_enable_amp: Generator[None,None,None],metrics_processor: MetricsProcessor | None = None,pp_schedule: _PipelineSchedule | None = None,pp_has_first_stage: bool | None = None,pp_has_last_stage: bool | None = None,))[HOT]→{validate},__init__((self,job_config: JobConfig))[CTOR,DUNDER],validate((self,model_parts: list[nn.Module])),should_validate((self,step: int)),__init__((self,job_config: JobConfig,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,parallel_dims: ParallelDims,loss_fn: LossFunction,validation_context: Generator[None,None,None],maybe_enable_amp: Generator[None,None,None],metrics_processor: MetricsProcessor,pp_schedule: _PipelineSchedule | None = None,pp_has_first_stage: bool | None = None,pp_has_last_stage: bool | None = None,))[CTOR,DUNDER]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 115 edges
Call: 12 edges

### CROSS_CLUSTER_FLOW

