from __future__ import annotations

import logging
import os

from sentence_transformers import SentenceTransformer

from chunking_utils import TokenCounter
from group_b_leaf_indexing import build_config_from_env, parse_optional_positive_int
from group_b_recursive_tree_builder import GroupBRecursiveTreeBuilder, GroupBTreeGrowthConfig
from neo4j_ops import GroupBLeafRepository
from shared_retrieval_utils import GROUP_RESOURCE_NAMES, load_runtime_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupB_Tree_Relay_Resume")


def _parse_start_layer(raw_value: str | None) -> int:
    if raw_value is None:
        return 1
    parsed = int(raw_value)
    return max(1, parsed)


def main() -> None:
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
    )
    base_cfg = build_config_from_env()

    start_layer = _parse_start_layer(os.getenv("GROUP_B_RESUME_START_LAYER", "1"))
    resume_max_growth_steps = parse_optional_positive_int(
        os.getenv("GROUP_B_RESUME_MAX_GROWTH_STEPS")
    )
    effective_max_growth_steps = (
        resume_max_growth_steps
        if resume_max_growth_steps is not None
        else base_cfg.tree_max_layers
    )

    embedding_model_name = os.getenv("GROUP_B_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    token_counter = TokenCounter(model_name=embedding_model_name)

    logger.info(
        "Loading embedding model %s for Group B relay resume...",
        embedding_model_name,
    )
    embedding_model = SentenceTransformer(embedding_model_name)

    group_resource = GROUP_RESOURCE_NAMES["B"]
    repo = GroupBLeafRepository(
        uri=runtime_cfg.neo4j_uri,
        user=runtime_cfg.neo4j_user,
        password=runtime_cfg.neo4j_password,
        node_label=group_resource["node_label"],
        index_name=group_resource["index_name"],
        embedding_dim=512,
    )

    try:
        seed_nodes = repo.fetch_nodes_by_layer(start_layer)
        logger.info("Relay start check: layer=%d, node_count=%d", start_layer, len(seed_nodes))
        if not seed_nodes:
            logger.warning(
                "No nodes found in start layer %d. Nothing to resume.",
                start_layer,
            )
            return

        tree_cfg = GroupBTreeGrowthConfig(
            reduction_dimension=base_cfg.reduction_dimension,
            umap_metric=base_cfg.umap_metric,
            gmm_prob_threshold=base_cfg.gmm_prob_threshold,
            gmm_max_clusters=base_cfg.gmm_max_clusters,
            cluster_max_tokens=base_cfg.cluster_max_tokens,
            tree_max_layers=effective_max_growth_steps,
            random_seed=base_cfg.random_seed,
            summarizer_model=base_cfg.summarizer_model,
            summarizer_api_base=runtime_cfg.llm_api_base,
            summarizer_api_key=runtime_cfg.ark_api_key,
            summary_max_tokens=base_cfg.summarizer_max_tokens,
            relationship_type=base_cfg.relationship_type,
        )

        logger.info(
            "Relay config: start_layer=%d, max_growth_steps=%s, dim=%d, metric=%s, tau=%.3f, max_k=%d, cluster_max_tokens=%d",
            start_layer,
            tree_cfg.tree_max_layers,
            tree_cfg.reduction_dimension,
            tree_cfg.umap_metric,
            tree_cfg.gmm_prob_threshold,
            tree_cfg.gmm_max_clusters,
            tree_cfg.cluster_max_tokens,
        )

        tree_builder = GroupBRecursiveTreeBuilder(
            repository=repo,
            embedding_model=embedding_model,
            token_counter=token_counter,
            config=tree_cfg,
        )
        tree_stats = tree_builder.grow_tree(start_layer=start_layer)
        logger.info(
            "Relay growth completed: layers_grown=%d, summary_nodes=%d, stopped_by_condition=%d, stopped_by_max_layers=%d",
            tree_stats.get("layers_grown", 0),
            tree_stats.get("summary_nodes_created", 0),
            tree_stats.get("stopped_by_condition", 0),
            tree_stats.get("stopped_by_max_layers", 0),
        )
    finally:
        repo.close()


if __name__ == "__main__":
    main()
