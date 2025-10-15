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
NODES:108 EDGES:60

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/optimizers/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:108 CALL_DEPTH:2

__init__.py→[] _decoupled_decay.py→[_compute_decay_factor((lr: float | Tensor,initial_lr: float | Tensor | None))[HOT]] adopt.py→[_multi_tensor_adopt((# noqa: C901,PLR0912,PLR0913,PLR0915 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,lr: float | Tensor,clip_lambda: Callable[[Number | Tensor | Any],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))→{_decoupled_decay::_compute_decay_factor,_decoupled_decay::_compute_decay_factor},report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))→{_decoupled_decay::_compute_decay_factor},_single_tensor_adopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1: float,beta2: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001))→{_decoupled_decay::_compute_decay_factor},_default_clip_lambda((step: Number | Tensor)),__init__((# noqa: C901,PLR0913 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.9,0.9999),eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((# noqa: PLR0913 self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),_fused_adopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,lr: float | Tensor,clip_lambda: Callable[[int],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))] aggmo_adamw.py→[report_per_parameter_metrics((# noqa: D102 self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))→{_decoupled_decay::_compute_decay_factor,aggmo_adopt::_sum_weights,aggmo_adopt::_build_moment_specs},_validate_vs_tuple((self,vs: Sequence[float]))→{aggmo_adopt::_sum_weights,aggmo_adopt::_build_moment_specs},_single_tensor_aggmo_qhadamw((# noqa: C901,PLR0913,PLR0912 params: list[Tensor],grads: list[Tensor],moment_buffers: list[list[Tensor]],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor] | None,state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | Tensor | None,decouple: bool,amsgrad: bool,beta1s: Sequence[float],beta2: float,vs: Sequence[float],lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001 grad_coeff: float,))→{_decoupled_decay::_compute_decay_factor,aggmo_adopt::_build_moment_specs},_validate_betas_tuple((self,betas: Sequence[float],vs: Sequence[float]))→{aggmo_adopt::_build_moment_specs},_setup_metric_functions((self,vs: Sequence[float]))→{aggmo_adopt::_build_moment_specs},_prepare_param_state((# noqa: C901 self,group: dict[str,Any],param: Tensor,moment_specs: Sequence[tuple[float,str]],))→
{aggmo_adopt::_is_moment_key},__init__((# noqa: PLR0913 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,...] = (0.9,0.95),vs: tuple[float,...] = (0.7,),eps: float = 1e-8,weight_decay: float = 1e-5,*,amsgrad: bool = False,decouple: bool = True,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],_validate_param_groups((self))] aggmo_adopt.py→[report_per_parameter_metrics((# noqa: D102 self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))→{_sum_weights,_build_moment_specs,_decoupled_decay::_compute_decay_factor},_validate_vs_tuple((self,vs: Sequence[float]))→{_sum_weights,_build_moment_specs},_single_tensor_aggmo_qhadopt((# noqa: C901,PLR0913 params: list[Tensor],grads: list[Tensor],moment_buffers: list[list[Tensor]],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | Tensor | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1s: Sequence[float],beta2: float,vs: Sequence[float],lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001 grad_coeff: float,))→{_build_moment_specs,_decoupled_decay::_compute_decay_factor},_validate_betas_tuple((self,betas: Sequence[float],vs: Sequence[float]))→{_build_moment_specs},_setup_metric_functions((self,vs: Sequence[float]))→{_build_moment_specs},_prepare_param_state((self,group: dict[str,Any],param: Tensor,moment_specs: Sequence[tuple[float,str]],))→{_is_moment_key},_build_moment_specs((vs: Sequence[float]))[HOT],_is_moment_key((key: str)),_sum_weights((moment_specs: Iterable[tuple[float,str]])),__init__((# noqa: PLR0913 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,...] = (0.999,0.9999),vs: tuple[float,...] = (0.9,),eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],_validate_param_groups((self))] decoupled_adamw.py→[report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))→{_decoupled_decay::_compute_decay_factor},__init__((# noqa: PLR0913 self,params: Iterable[torch.Tensor] | Iterable[dict],lr: float = 1e-3,betas: tuple[float,float] = (0.9,0.95),eps: float = 1e-8,weight_decay: float = 1e-5,*,amsgrad: bool = False,decouple: bool = True,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER]] qhadamw.py→[report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))→{_decoupled_decay::_compute_decay_factor},_single_tensor_qhadamw((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,amsgrad: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001))→{_decoupled_decay::_compute_decay_factor},_multi_tensor_qhadamw((# noqa: C901,PLR0913,PLR0912,PLR0915 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,amsgrad: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))→
{_decoupled_decay::_compute_decay_factor},__init__((# noqa: PLR0913,PLR0912,C901 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.9,0.95),vs: tuple[float,...] = (0.7,),eps: float = 1e-8,weight_decay: float = 1e-5,*,amsgrad: bool = False,decouple: bool = True,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((# noqa: PLR0913 self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),_fused_qhadamw((*args: Any,**kwargs: Any,))] qhadopt.py→[_multi_tensor_qhadopt((# noqa: C901,PLR0912,PLR0913,PLR0915 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,clip_lambda: Callable[[Number | Tensor | Any],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))→{_decoupled_decay::_compute_decay_factor,_decoupled_decay::_compute_decay_factor},report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))→{_decoupled_decay::_compute_decay_factor},_single_tensor_qhadopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001))→{_decoupled_decay::_compute_decay_factor},_default_clip_lambda((step: Number | Tensor)),__init__((# noqa: C901,PLR0913,PLR0912 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.999,0.9999),vs: tuple[float,...] = (0.9,),eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((# noqa: PLR0913 self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),_fused_qhadopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,clip_lambda: Callable[[int],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 32 edges
Contains: 28 edges

### CROSS_CLUSTER_FLOW

