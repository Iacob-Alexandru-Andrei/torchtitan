# Development Standards
- Setup the environment by running the commands in `torchtitan/experiments/fl/scripts/setup/setup_env.sh`.
- Versions are brittle; when invoking uv run commands, always include `--no-sync` (for example, `uv run --no-sync ...`). Use the setup script for the initial `uv sync`.
- This fork centralizes new features and experiments under `torchtitan/experiments/fl`; keep changes elsewhere minimal and compatibility-focused.
- Keep compatibility with upstream torchtitan simple by reusing existing imports and functionality whenever possible.
- When modifying a torchtitan component, model, or layer, create a subclass with the required changes. If compatibility must be broken, look for a higher-level base class before rebuilding from scratch.
- If two versions of a component must coexist, provide an explicit selector (for example, model-name strings or config booleans) and keep legacy behavior intact.
- Follow the linting and typing style used in `torchtitan/experiments/fl`: add docstrings with parameters, types, and returns, and type every symbol.
- Always run linting and the relevant tests (or pre-commit) before sending changes.
- Keep experimental behaviors behind feature flags or config options; defaults should remain backward compatible.
- Prefer small, well-documented diffs; note new configs or model names and keep logging clear for debugging.
- Avoid hardcoding environment-specific paths or secrets; keep inputs configurable.
- Add focused unit tests for compatibility layers and behavioral changes, especially around experiment routing and model selection.
- Profile before altering performance-sensitive paths; avoid regressions in hot kernels or distributed code.

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
NODES:94 EDGES:41

## DIRECTORY_TREE
ROOT: torchtitan/experiments/fl/tests/

## ARCHITECTURAL_CLUSTERS

### TESTS
NODES:94 CALL_DEPTH:3

test_llama3_mup_scion.py→[test_scion_assigns_hidden_and_output_scales(void)[TEST]→{_find_group,_find_group,_build_param_groups},test_untied_embeddings_use_distinct_param_groups(void)[TEST]→{_find_group,_find_group,_build_param_groups_from_args},test_disco_assigns_sqrt_norms_when_untied(void)[TEST]→{_find_group,_find_group,_build_param_groups_from_args},test_use_disco_sets_sign_norm_for_embeddings(void)[TEST]→{_find_group,_build_param_groups},test_use_disco_sets_spectral_norm_for_hidden_weights(void)[TEST]→{_find_group,_build_param_groups},test_plain_scion_sets_spectral_norm_for_hidden_weights(void)[TEST]→{_find_group,_build_param_groups},test_plain_scion_sets_sign_norm_for_embeddings(void)[TEST]→{_find_group,_build_param_groups},test_scion_scale_overrides_respected_via_optimizer_config(void)[TEST]→{_find_group,_find_group},_build_param_groups((use_scion: bool,*,use_disco: bool = False))[HOT]→{_build_param_groups_from_args},test_attention_value_norm_bucketed_with_hidden_ln(void)[TEST]→{_bucketize_and_get},test_attention_output_norm_bucketed_with_hidden_ln(void)[TEST]→{_bucketize_and_get},test_mlp_mid_norm_bucketed_with_hidden_ln(void)[TEST]→{_bucketize_and_get},test_disco_embedding_init_matches_expected_norm(void)[TEST]→{_build_small_model},test_disco_output_init_matches_expected_norm(void)[TEST]→{_build_small_model},test_disco_hidden_inits_are_unit_norm(void)[TEST]→{_build_small_model},_build_model((use_scion: bool,*,use_disco: bool = False))[HOT],_build_small_model((use_scion: bool,*,use_disco: bool = False,**overrides: object))[HOT],_build_param_groups_from_args((model_args: TransformerModelArgs))[HOT],_find_group((param_groups: list[dict],parameter)),test_scion_preserves_width_multiplier_value(void)[TEST],test_scion_lr_scaling_ignores_width_multiplier(void)[TEST],test_layernorm_impl_overrides_flag(void)[TEST],test_qk_layernorm_impl_independent(void)[TEST],test_qk_layernorm_inherits_general_when_unset(void)[TEST],test_attention_value_norm_flag_creates_layer(void)[TEST],test_attention_output_norm_flag_creates_layer(void)[TEST],test_mlp_mid_norm_flag_creates_layer(void)[TEST],_bucketize_and_get((model: Transformer,param_name: str)),test_default_disco_init_types(void)[TEST],test_default_non_scion_init_types(void)[TEST],test_standard_scion_without_disco_uses_normal_init_types(void)[TEST],test_unembed_bucket_created_when_weights_untied(void)[TEST],test_custom_hidden_init_type_applied(void)[TEST],test_scion_skips_mup_input_output_alpha_scaling(void)[TEST],test_trunc_normal_init_respects_cutoff(void)[TEST]] test_optimizer_builder.py→[test_default_builder_uses_core_optimizer(void)[HOT,TEST]→{_dims},test_default_builder_rejects_mosaic_only_optimizer(void)[HOT,TEST]→{_dims},test_qhscion_builder_exposes_betas_and_vs(void)[HOT,TEST]→{_dims},test_qhscion_builder_prefers_scion_v_override(void)[HOT,TEST]→{_dims},test_scion_builder_accepts_custom_zeropower_coefficients(void)[HOT,TEST]→{_dims},__init__((self))[CTOR,DUNDER],_dims(void)] test_scion_optimizer.py→[test_qhscion_param_group_exposes_vs_tuple(void)[TEST]→{_param},test_qhscion_recovers_vs_from_legacy_v(void)[TEST]→{_param},_param(void),test_scion_scale_only_group_applies_radius(void)[TEST],test_scion_scale_applies_with_explicit_norm(void)[TEST],test_scion_sign_norm_respects_normalized_flag(void)[TEST]] test_unigram_metrics.py→
[__init__((self,*_args: object,**_kwargs: object))[CTOR,DUNDER],add_state((self,name: str,default: torch.Tensor,dist_reduce_fx: str | None = None,)),register_buffer((self,name: str,tensor: torch.Tensor)),__init__((self,*_args: object,**_kwargs: object))[CTOR,DUNDER],__init__((self,*_args: object,**_kwargs: object))[CTOR,DUNDER],get_peak_stats((self)),reset_peak_stats((self)),test_unigram_manager_aggregation_and_reset(void)[TEST],test_unigram_manager_teardown_removes_metric(void)[TEST],test_fl_metrics_processor_registers_expected_callbacks(void)[TEST],test_unigram_payload_reports_local_and_global_metrics(void)[TEST],test_unigram_local_metric_logged_before_global(void)[TEST]] 

## DEPENDENCY_PATTERNS

### EDGE_PATTERNS
Call: 33 edges
Contains: 8 edges

### CROSS_CLUSTER_FLOW

