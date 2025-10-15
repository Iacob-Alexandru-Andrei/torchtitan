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
NODES:52 EDGES:31

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/dataloader/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:52 CALL_DEPTH:6

dataloader.py→[_extract_streams((dataset_cfg: dict[str,Any]))→{_join_local_path,_join_remote_path,_collect_group_stream_entries,_normalize_sampling_groups,_collect_group_stream_entries,_normalize_sampling_groups},_build_mosaic_dataloader((*,job_config: MosaicJobConfig,tokenizer: BaseTokenizer,dp_world_size: int,dp_rank: int,split: str,default_drop_last: bool,))[HOT]→{_create_streaming_dataset,_setup_unigram_metric,_prepare_dataset_kwargs,_select_stream_subset,_extract_streams,_extract_streams},_join_remote_path((root: str | None,path: str | None))→{_is_uri,_is_uri},_collect_group_stream_entries((group: Mapping[str,Any]))→{_flatten_stream_configs},_load_stream_unigram_counts((stream: Stream,*,root_remote: str | None,dataset_split: str | None,default_split: str,config: UnigramMetricConfig,))→{_maybe_download_unigram_file},_normalize_mosaic_dataloader_config((job_config: MosaicJobConfig,*,split: str,default_drop_last: bool,))→{_select_dataset_config},_setup_unigram_metric((assignment: StreamAssignment,*,job_config: MosaicJobConfig,split: str,tokenizer: BaseTokenizer,))→{_build_unigram_metric_for_group},_build_unigram_metric_for_group((streams: list[Stream] | None,default_split: str,tokenizer: BaseTokenizer,config: UnigramMetricConfig,group_key: str,dataset_root_remote: str | None,dataset_split_remote: str | None,))[HOT]→{_load_stream_unigram_counts},build_mosaic_dataloader((*,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: MosaicJobConfig,))[HOT]→{_build_mosaic_dataloader},build_mosaic_validation_dataloader((*,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: MosaicJobConfig,infinite: bool = False,# noqa: ARG001 - kept for compatibility))[HOT]→{_build_mosaic_dataloader},_is_uri((path: str | None)),_join_local_path((root: str | None,path: str | None)),_flatten_stream_configs((streams_cfg: Any)),_normalize_sampling_groups((config: Any)),_select_dataset_config((dataset_cfg: Mapping[str,Any] | None,split: str)),_maybe_download_unigram_file((remote_uri: str | None,root_remote: str | None,split: str,destination: Path,config: UnigramMetricConfig,)),_select_stream_subset((extraction: StreamExtractionResult,*,dp_rank: int,dp_world_size: int,)),_prepare_dataset_kwargs((dataset_cfg: dict[str,Any],*,dataset_split_remote: str | None,)),_create_streaming_dataset((*,assignment: StreamAssignment,tokenizer: BaseTokenizer,dataset_config: DatasetFactoryConfig,batch_size: int,split: str,)),__init__((self,*args: Any,**kwargs: Any))[CTOR,DUNDER],__getitem__((self,idx: int))[DUNDER],state_dict((self,num_samples: int | None = None,from_beginning: bool = True)),load_state_dict((self,obj: dict[str,Any])),__init__((self,dataset: StatefulStreamingTextDataset,dp_rank: int,dp_world_size: int,batch_size: int,collate_fn: Callable | None = None,num_workers: int = 0,prefetch_factor: int | None = 2,pin_memory: bool = True,persistent_workers: bool = True,drop_last: bool = True,))[CTOR,DUNDER],state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),titan_collate_fn((batch: list[Any]))] tokenizer.py→[build_mosaic_tokenizer((job_config: MosaicJobConfig,))[HOT]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 7 edges
Call: 24 edges

### CROSS_CLUSTER_FLOW

