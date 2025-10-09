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
NODES:60 EDGES:14

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/optimizers/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:60 CALL_DEPTH:1

__init__.py→[] adopt.py→[_default_clip_lambda((step: Number | Tensor)),__init__((# noqa: C901,PLR0913 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.9,0.9999),eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((# noqa: PLR0913 self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],)),_single_tensor_adopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1: float,beta2: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001)),_multi_tensor_adopt((# noqa: C901,PLR0912,PLR0913,PLR0915 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,lr: float | Tensor,clip_lambda: Callable[[Number | Tensor | Any],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,)),_fused_adopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,lr: float | Tensor,clip_lambda: Callable[[int],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))] decoupled_adamw.py→[__init__((# noqa: PLR0913 self,params: Iterable[torch.Tensor] | Iterable[dict],lr: float = 1e-3,betas: tuple[float,float] = (0.9,0.95),eps: float = 1e-8,weight_decay: float = 1e-5,*,amsgrad: bool = False,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],))] qhadamw.py→
[__init__((# noqa: PLR0913,PLR0912,C901 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.9,0.95),vs: tuple[float,...] = (0.7,),eps: float = 1e-8,weight_decay: float = 1e-5,*,amsgrad: bool = False,decouple: bool = True,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((# noqa: PLR0913 self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],)),_single_tensor_qhadamw((# noqa: C901,PLR0913,PLR0912 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,amsgrad: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001)),_multi_tensor_qhadamw((# noqa: C901,PLR0913,PLR0912,PLR0915 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],max_exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,amsgrad: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,)),_fused_qhadamw((*args: Any,**kwargs: Any,))] qhadopt.py→
[_default_clip_lambda((step: Number | Tensor)),__init__((# noqa: C901,PLR0913,PLR0912 self,params: ParamsT,lr: float | Tensor = 1e-3,betas: tuple[float,float] = (0.999,0.9999),vs: tuple[float,...] = (0.9,),eps: float = 1e-6,clip_lambda: (Callable[[Number | Tensor | Any],float] | None) = _default_clip_lambda,weight_decay: float = 0.0,*,decouple: bool = False,foreach: bool | None = None,maximize: bool = False,capturable: bool = False,differentiable: bool = False,fused: bool | None = None,))[CTOR,DUNDER],__setstate__((self,state: dict))[DUNDER],_init_group((# noqa: PLR0913 self,group: dict,params_with_grad: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],)),report_per_parameter_metrics((self,param: torch.Tensor,name: str,optimizer_metrics: dict[str,torch.Tensor],)),_single_tensor_qhadopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,decouple: bool,clip_lambda: Callable[[Number | Tensor | Any],float] | None,beta1: float,beta2: float,v1: float,lr: float | Tensor,weight_decay: float,eps: float,maximize: bool,capturable: bool,differentiable: bool,has_complex: bool,# noqa: ARG001)),_multi_tensor_qhadopt((# noqa: C901,PLR0912,PLR0913,PLR0915 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,clip_lambda: Callable[[Number | Tensor | Any],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,)),_fused_qhadopt((# noqa: PLR0913 params: list[Tensor],grads: list[Tensor],exp_avgs: list[Tensor],exp_avg_sqs: list[Tensor],state_steps: list[Tensor],grad_scale: Tensor | None,found_inf: Tensor | None,*,initial_lr: float | None,has_complex: bool,beta1: float,beta2: float,v1: float,lr: float | Tensor,clip_lambda: Callable[[int],float] | None,weight_decay: float,decouple: bool,eps: float,maximize: bool,capturable: bool,differentiable: bool,))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 14 edges

### CROSS_CLUSTER_FLOW

