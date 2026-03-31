# Retrieval Ablation Engineering Notes

This note captures non-mainline engineering agreements so experiments stay reproducible and extensible.

## 1) LLM generator/evaluator placement
- Keep one local copy of `llm_generator.py` and `llm_evaluator.py` under each group folder.
- Current layout supports direct local imports in each pipeline script:
  - Group A: ready
  - Group B: initialized
  - Group C: initialized

## 2) Week filtering policy
- Current experiment scope is Week 1 only.
- Filtering is path-based and can be disabled without changing retrieval/indexing algorithms.
- In Group A baseline indexing:
  - `TARGET_WEEK_FILTER=【第1周】Python基本语法元素` keeps Week 1 scope.
  - `TARGET_WEEK_FILTER=none` (or `all`, `*`, empty) disables filter and processes all weeks.

## 3) Output contract
- Keep `generated_answer` in output for manual verification.

## 4) Naming conventions (for A/B/C isolation)
- Group A
  - Node label: `BaselineChunk`
  - Index: `baseline_chunk_vector_index`
  - Strategy: `flat_vector_baseline`
- Group B (planned)
  - Node label: `RaptorTreeNodeB`
  - Index: `group_b_tree_vector_index`
  - Strategy: `tree_retrieval`
- Group C (planned)
  - Node label: `GraphClusterNodeC`
  - Index: `group_c_cluster_vector_index`
  - Strategy: `graph_cluster_retrieval`

## 5) Runtime configuration policy
- Keep hardcoded defaults for reproducibility.
- Allow environment variable overrides for portability/security.
- Shared loader: `shared_retrieval_utils.load_runtime_config`.

Recommended environment variables:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `OLLAMA_API_URL`
- `QA_DATASET_PATH`
- `DATASET_ROOT`
- `TARGET_WEEK_FILTER`
- `A_REWRITE_MODEL`
- `A_REWRITE_N`
- `A_TOP_K_PER_TERM`
- `A_SIMILARITY_THRESHOLD`

## 6) Shared query rewrite utility
- Use `shared_retrieval_utils.QueryRewriter` for all groups.
- Group-level defaults are set in `GROUP_REWRITE_DEFAULTS` and can be overridden at runtime.
