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
NODES:86 EDGES:42

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/dataloader/

## ARCHITECTURAL_CLUSTERS

### UTILITY_LAYER
NODES:86 CALL_DEPTH:8

__init__.py→[] dataloader.py→[_build_mosaic_dataloader((request: DataloaderBuildRequest,*,register_unigram_metric: Callable[["PureUnigramCrossEntropy"],None] | None,))[HOT]→{_apply_split_overrides,unigram::setup_unigram_metric,dataset_factory::build_dataset_for_rank,streams::_extract_streams,dataset_factory::_normalize_mosaic_dataloader_config},build_mosaic_dataloader((*,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: MosaicJobConfig,register_unigram_metric: Callable[["PureUnigramCrossEntropy"],None] | None = None,))[HOT]→{_build_mosaic_dataloader},build_mosaic_validation_dataloader((*,dp_world_size: int,dp_rank: int,tokenizer: BaseTokenizer,job_config: MosaicJobConfig,infinite: bool = False,# noqa: ARG001 - kept for compatibility register_unigram_metric: Callable[["PureUnigramCrossEntropy"],None] | None = None,))[HOT]→{_build_mosaic_dataloader},_apply_split_overrides((normalized: NormalizedMosaicConfig,*,job_config: MosaicJobConfig,split: str))] dataset_factory.py→[build_dataset_for_rank((normalized: NormalizedMosaicConfig,extraction: StreamExtractionResult,*,dp_rank: int,dp_world_size: int,tokenizer: BaseTokenizer,split: str,))[HOT]→{create_streaming_dataset,_prepare_dataset_kwargs,streams::_select_stream_subset},_normalize_mosaic_dataloader_config((job_config: MosaicJobConfig,*,split: str,default_drop_last: bool,))→{_select_dataset_config},_select_dataset_config((dataset_cfg: Mapping[str,Any] | None,split: str)),_prepare_dataset_kwargs((dataset_cfg: dict[str,Any],*,dataset_split_remote: str | None,)),create_streaming_dataset((*,assignment: StreamAssignment,tokenizer: BaseTokenizer,dataset_config: DatasetFactoryConfig,batch_size: int,split: str,))] parallel.py→[__init__((self,*args: Any,**kwargs: Any))[CTOR,DUNDER],__getitem__((self,idx: int))[DUNDER],state_dict((self,num_samples: int | None = None,*,from_beginning: bool = True)),load_state_dict((self,obj: dict[str,Any])),__init__((self,dataset: StatefulStreamingTextDataset,request: ParallelDataLoaderRequest))[CTOR,DUNDER],state_dict((self)),load_state_dict((self,state_dict: dict[str,Any])),titan_collate_fn((batch: list[Any]))] streams.py→[_extract_streams((dataset_cfg: dict[str,Any]))→{_compute_group_indices,_materialize_streams,_resolve_group_stream_names,_aggregate_sampling_groups,_should_concat_sampling_groups,_normalize_sampling_mode},_aggregate_sampling_groups((flattened: dict[str,dict[str,Any]],sampling_groups_cfg: Any,mode_raw: Any,))→{_resolve_group_candidate,_collect_group_stream_entries,_normalize_sampling_groups},_join_remote_path((root: str | None,path: str | None))→{_is_uri,_is_uri},_resolve_group_stream_names((flattened: dict[str,dict[str,Any]],sampling_groups_cfg: Any))[HOT]→{_collect_group_stream_entries,_normalize_sampling_groups},_materialize_streams((flattened: dict[str,dict[str,Any]],*,root_remote: str | None,root_local: str | None,))→{_join_local_path,_join_remote_path},_collect_group_stream_entries((group: Mapping[str,Any]))→{_flatten_stream_configs},_is_uri((path: str | None)),_join_local_path((root: str | None,path: str | None)),_flatten_stream_configs((streams_cfg: Any)),_normalize_sampling_groups((config: Any)),_normalize_sampling_mode((mode_raw: Any)),_should_concat_sampling_groups((mode: str,sampling_groups_cfg: Any)),_resolve_group_candidate((entry: Any,*,flattened: dict[str,dict[str,Any]],group: Mapping[str,Any],entry_index: int,))[HOT],_compute_group_indices((group_stream_names: list[list[str]] | None,stream_names: list[str]))[HOT],_select_stream_subset((extraction: StreamExtractionResult,*,dp_rank: int,dp_world_size: int,))] tokenizer.py→[build_mosaic_tokenizer((job_config: MosaicJobConfig,))[HOT]] unigram.py→[_maybe_download_unigram_file((remote_uri: str | None,root_remote: str | None,split: str,destination: Path,config: UnigramMetricConfig,))→
{_create_remote_unigram_client,_resolve_unigram_remote_path},_load_stream_unigram_counts((stream: Stream,*,root_remote: str | None,dataset_split: str | None,default_split: str,config: UnigramMetricConfig,))→{_materialize_split_cache,_resolve_unigram_cache_path},_resolve_unigram_cache_path((stream: Stream,*,root_remote: str | None,dataset_split: str | None,default_split: str,config: UnigramMetricConfig,))[HOT]→{_maybe_download_unigram_file},_build_unigram_metric_for_group((context: UnigramMetricContext,))[HOT]→{_load_stream_unigram_counts},setup_unigram_metric((assignment: StreamAssignment,*,job_config: MosaicJobConfig,split: str,tokenizer: BaseTokenizer,collate_fn: Callable,))→{_build_unigram_metric_for_group},_resolve_unigram_remote_path((remote_uri: str,*,root_remote: str | None,split: str,))[HOT],_create_remote_unigram_client((bucket: str,config: UnigramMetricConfig)),_materialize_split_cache((cache_path: Path,split_path: Path))] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Contains: 7 edges
Call: 35 edges

### CROSS_CLUSTER_FLOW

