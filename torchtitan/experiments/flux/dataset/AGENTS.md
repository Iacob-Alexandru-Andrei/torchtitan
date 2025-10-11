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
NODES:48 EDGES:21

## DIRECTORY_TREE
ROOT: torchtitan/experiments/flux/dataset/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:48 CALL_DEPTH:2

flux_dataset.py→[_cc12m_wds_data_processor((sample: dict[str,Any],t5_tokenizer: FluxTokenizer,clip_tokenizer: FluxTokenizer,output_size: int = 256,))→{_process_cc12m_image},_coco_data_processor((sample: dict[str,Any],t5_tokenizer: FluxTokenizer,clip_tokenizer: FluxTokenizer,output_size: int = 256,))→{_process_cc12m_image},__init__((self,dataset_name: str,dataset_path: Optional[str],t5_tokenizer: BaseTokenizer,clip_tokenizer: BaseTokenizer,job_config: Optional[JobConfig] = None,dp_rank: int = 0,dp_world_size: int = 1,infinite: bool = False,))[CTOR,DUNDER]→{_validate_dataset},build_flux_dataloader((dp_world_size: int,dp_rank: int,job_config: JobConfig,# This parameter is not used,keep it for compatibility tokenizer: FluxTokenizer | None,infinite: bool = True,))[HOT]→{tokenizer::build_flux_tokenizer},build_flux_validation_dataloader((dp_world_size: int,dp_rank: int,job_config: JobConfig,# This parameter is not used,keep it for compatibility tokenizer: BaseTokenizer | None,generate_timestamps: bool = True,infinite: bool = False,))[HOT]→{tokenizer::build_flux_tokenizer},_process_cc12m_image((img: PIL.Image.Image,output_size: int = 256,)),_validate_dataset((dataset_name: str,dataset_path: Optional[str] = None)),_get_data_iter((self)),__iter__((self))[DUNDER],load_state_dict((self,state_dict)),state_dict((self)),__init__((self,dataset_name: str,dataset_path: Optional[str],t5_tokenizer: BaseTokenizer,clip_tokenizer: BaseTokenizer,job_config: Optional[JobConfig] = None,dp_rank: int = 0,dp_world_size: int = 1,generate_timesteps: bool = True,infinite: bool = False,))[CTOR,DUNDER],__iter__((self))[DUNDER]] tokenizer.py→[__init__((self,model_path: str = "t5-small",max_length: int = 77,**hf_kwargs))[CTOR,DUNDER],_pad_and_chunk_tokens((self,tokens: List[int],max_length: int,pad_token: int)),get_vocab_size((self)),encode((self,text: str | list[str])),decode((self,t: List[int])),__init__((self,model_path: str = "t5-small",max_length: int = 77,**hf_kwargs))[CTOR,DUNDER],get_vocab_size((self)),encode((self,s: str | list[str],)),decode((self,t: List[int])),build_flux_tokenizer((job_config: JobConfig))[HOT]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 5 edges
Contains: 16 edges

### CROSS_CLUSTER_FLOW

