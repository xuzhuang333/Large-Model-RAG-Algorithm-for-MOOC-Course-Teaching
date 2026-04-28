#这个文件是Group B的Leaf Indexing模块，负责从指定的物理目录中提取文本文件，进行基于句子的智能切分，并将切分后的文本块与其对应的元数据一起存储到Neo4j数据库中。这个模块还支持对SRT字幕文件进行特殊处理，去除时间戳和序号等非文本内容，以获得更干净的文本输入。切分过程中会尽量保持语义完整性，同时也提供了极端情况下的强制切分机制，以确保每个文本块都在模型可接受的Token限制范围内。
from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from chunking_utils import TokenCounter, build_sentence_complete_chunks
from group_b_recursive_tree_builder import GroupBRecursiveTreeBuilder, GroupBTreeGrowthConfig
from neo4j_ops import GroupBLeafRepository
from shared_retrieval_utils import (
    DEFAULT_DEEPSEEK_ENDPOINT,
    GROUP_RESOURCE_NAMES,
    load_runtime_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupB_Leaf_Indexing")

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".srt"}
SRT_TS_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$")


@dataclass
class GroupBLeafIndexingConfig:
    dataset_root: str
    token_limit_per_chunk: int = 100
    extreme_force_split_token_cap: int = 400
    embedding_batch_size: int = 64
    clear_before_insert: bool = True
    build_recursive_tree: bool = True
    reduction_dimension: int = 10
    umap_metric: str = "cosine"
    gmm_prob_threshold: float = 0.1
    gmm_max_clusters: int = 50
    cluster_max_tokens: int = 3500
    tree_max_layers: int | None = None
    random_seed: int = 224
    summarizer_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    summarizer_max_tokens: int = 220
    relationship_type: str = "GROUP_B_PARENT_OF"
    supported_extensions: tuple[str, ...] = (".txt", ".md", ".srt")


def parse_supported_extensions(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return tuple(sorted(SUPPORTED_TEXT_EXTENSIONS))

    normalized = []
    for part in raw_value.split(","):
        ext = part.strip().lower()
        if not ext:
            continue
        normalized.append(ext if ext.startswith(".") else f".{ext}")

    unique_exts = sorted(set(normalized))
    return tuple(unique_exts) if unique_exts else tuple(sorted(SUPPORTED_TEXT_EXTENSIONS))


def parse_optional_positive_int(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"", "none", "null", "all", "unlimited", "inf", "infinite"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError("Expected positive integer for max growth steps.")
    return parsed


def clean_srt_text(raw_text: str) -> str:
    cleaned_lines = []
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if SRT_TS_PATTERN.match(stripped):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def read_text_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "r", encoding="utf-8") as file:
        raw = file.read()
    return clean_srt_text(raw) if ext == ".srt" else raw


def build_leaf_node_id(source_file: str, chunk_order: int, text: str) -> str:
    seed = f"{source_file}::{chunk_order}::{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(seed).hexdigest()[:20]
    return f"B0_{digest}"


def collect_leaf_nodes(
    config: GroupBLeafIndexingConfig,
    token_counter: TokenCounter,
    audit_stats: dict[str, int] | None = None,
) -> list[dict]:
    all_rows: list[dict] = []
    processed_files = 0

    for dirpath, _, filenames in os.walk(config.dataset_root):

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in config.supported_extensions:
                continue

            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, config.dataset_root)

            try:
                text = read_text_file(abs_path)
            except Exception as exc:
                logger.warning("Skip file read failure: %s, error=%s", abs_path, exc)
                continue

            if not text.strip():
                continue

            chunks = build_sentence_complete_chunks(
                text,
                token_limit=config.token_limit_per_chunk,
                token_counter=token_counter,
                hard_token_cap=config.extreme_force_split_token_cap,
                audit_stats=audit_stats,
            )
            if not chunks:
                continue

            processed_files += 1
            for idx, chunk in enumerate(chunks, start=1):
                chunk_text = chunk["text"].strip()
                if not chunk_text:
                    continue

                all_rows.append(
                    {
                        "node_id": build_leaf_node_id(rel_path, idx, chunk_text),
                        "text": chunk_text,
                        "source_file": rel_path,
                        "source_type": ext.lstrip("."),
                        "chunk_order": idx,
                        "token_count": int(chunk["token_count"]),
                        "sentence_count": int(chunk["sentence_count"]),
                    }
                )

    logger.info("Collected Group B leaf chunks: files=%d, chunks=%d", processed_files, len(all_rows))
    return all_rows


def embed_and_insert_leaf_nodes(
    rows: list[dict],
    embedding_model: SentenceTransformer,
    repository: GroupBLeafRepository,
    batch_size: int,
) -> int:
    if not rows:
        return 0

    total_inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [item["text"] for item in batch]

        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
        )

        payload = []
        for row, emb in zip(batch, embeddings):
            payload.append(
                {
                    "node_id": row["node_id"],
                    "text": row["text"],
                    "source_file": row["source_file"],
                    "source_type": row["source_type"],
                    "chunk_order": row["chunk_order"],
                    "token_count": row["token_count"],
                    "sentence_count": row["sentence_count"],
                    "embedding": [float(x) for x in emb],
                }
            )

        inserted = repository.insert_leaf_nodes(payload)
        total_inserted += inserted
        logger.info("Inserted batch %d-%d, inserted=%d", start + 1, start + len(batch), inserted)

    return total_inserted


def build_config_from_env() -> GroupBLeafIndexingConfig:
    dataset_root = os.getenv(
        "DATASET_ROOT",
        r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学",
    )

    token_limit_per_chunk = int(os.getenv("GROUP_B_SENTENCE_CHUNK_TOKEN_LIMIT", "100"))
    extreme_force_split_token_cap = int(
        os.getenv("GROUP_B_EXTREME_FORCE_SPLIT_TOKEN_CAP", "400")
    )
    if extreme_force_split_token_cap < token_limit_per_chunk:
        extreme_force_split_token_cap = token_limit_per_chunk
    if extreme_force_split_token_cap > 512:
        extreme_force_split_token_cap = 512

    embedding_batch_size = int(os.getenv("GROUP_B_EMBEDDING_BATCH_SIZE", "64"))

    legacy_clear_before_insert = os.getenv("GROUP_B_CLEAR_BEFORE_INSERT")
    if legacy_clear_before_insert and legacy_clear_before_insert.strip().lower() in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        logger.warning(
            "GROUP_B_CLEAR_BEFORE_INSERT=%s ignored; clear-before-insert is forced to true for overwrite reruns.",
            legacy_clear_before_insert,
        )

    build_recursive_tree = os.getenv("GROUP_B_BUILD_RECURSIVE_TREE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    reduction_dimension = max(2, int(os.getenv("GROUP_B_REDUCTION_DIM", "10")))

    umap_metric = os.getenv("GROUP_B_UMAP_METRIC", "cosine").strip().lower() or "cosine"
    gmm_prob_threshold = float(os.getenv("GROUP_B_GMM_PROB_THRESHOLD", "0.1"))
    gmm_prob_threshold = min(1.0, max(0.0, gmm_prob_threshold))

    gmm_max_clusters = max(2, int(os.getenv("GROUP_B_GMM_MAX_CLUSTERS", "50")))
    cluster_max_tokens = max(
        token_limit_per_chunk,
        int(os.getenv("GROUP_B_CLUSTER_MAX_TOKENS", "3500")),
    )
    max_growth_steps_raw = os.getenv("GROUP_B_MAX_GROWTH_STEPS")
    tree_max_layers = parse_optional_positive_int(max_growth_steps_raw)
    deprecated_tree_max_layers = os.getenv("GROUP_B_TREE_MAX_LAYERS")
    if deprecated_tree_max_layers and max_growth_steps_raw is None:
        logger.warning(
            "Ignore deprecated env GROUP_B_TREE_MAX_LAYERS=%s to avoid accidental early stop. "
            "Use GROUP_B_MAX_GROWTH_STEPS when you need an explicit cap.",
            deprecated_tree_max_layers,
        )
    random_seed = int(os.getenv("GROUP_B_RANDOM_SEED", "224"))
    summarizer_model = os.getenv("GROUP_B_SUMMARIZER_MODEL", DEFAULT_DEEPSEEK_ENDPOINT)
    summarizer_max_tokens = max(64, int(os.getenv("GROUP_B_SUMMARY_MAX_TOKENS", "220")))
    relationship_type = os.getenv("GROUP_B_REL_TYPE", "GROUP_B_PARENT_OF")

    supported_extensions = parse_supported_extensions(os.getenv("GROUP_B_TEXT_EXTENSIONS"))

    return GroupBLeafIndexingConfig(
        dataset_root=dataset_root,
        token_limit_per_chunk=token_limit_per_chunk,
        extreme_force_split_token_cap=extreme_force_split_token_cap,
        embedding_batch_size=embedding_batch_size,
        clear_before_insert=True,
        build_recursive_tree=build_recursive_tree,
        reduction_dimension=reduction_dimension,
        umap_metric=umap_metric,
        gmm_prob_threshold=gmm_prob_threshold,
        gmm_max_clusters=gmm_max_clusters,
        cluster_max_tokens=cluster_max_tokens,
        tree_max_layers=tree_max_layers,
        random_seed=random_seed,
        summarizer_model=summarizer_model,
        summarizer_max_tokens=summarizer_max_tokens,
        relationship_type=relationship_type,
        supported_extensions=supported_extensions,
    )


def main() -> None:
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="YOUR VALUE",
    )
    indexing_cfg = build_config_from_env()
    embedding_model_name = os.getenv("GROUP_B_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    token_counter = TokenCounter(model_name=embedding_model_name)

    logger.info("Loading embedding model %s for Group B leaf indexing...", embedding_model_name)
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
    audit_stats: dict[str, int] = {
        "extreme_forced_split_sentences": 0,
        "extreme_forced_split_subchunks": 0,
    }

    logger.info(
        "Group B tree config: build=%s dim=%d metric=%s tau=%.3f max_k=%d cluster_max_tokens=%d max_growth_steps=%s seed=%d",
        indexing_cfg.build_recursive_tree,
        indexing_cfg.reduction_dimension,
        indexing_cfg.umap_metric,
        indexing_cfg.gmm_prob_threshold,
        indexing_cfg.gmm_max_clusters,
        indexing_cfg.cluster_max_tokens,
        indexing_cfg.tree_max_layers,
        indexing_cfg.random_seed,
    )

    try:
        if indexing_cfg.clear_before_insert:
            logger.info("Clearing Group B nodes and vector index before insertion...")
            repo.clear_group_b_nodes_and_index()

        repo.create_vector_index()

        rows = collect_leaf_nodes(
            indexing_cfg,
            token_counter=token_counter,
            audit_stats=audit_stats,
        )
        inserted = embed_and_insert_leaf_nodes(
            rows=rows,
            embedding_model=embedding_model,
            repository=repo,
            batch_size=indexing_cfg.embedding_batch_size,
        )
        logger.info(
            "Extreme force-split audit: triggered_sentences=%d, generated_subchunks=%d",
            audit_stats.get("extreme_forced_split_sentences", 0),
            audit_stats.get("extreme_forced_split_subchunks", 0),
        )

        if indexing_cfg.build_recursive_tree:
            tree_cfg = GroupBTreeGrowthConfig(
                reduction_dimension=indexing_cfg.reduction_dimension,
                umap_metric=indexing_cfg.umap_metric,
                gmm_prob_threshold=indexing_cfg.gmm_prob_threshold,
                gmm_max_clusters=indexing_cfg.gmm_max_clusters,
                cluster_max_tokens=indexing_cfg.cluster_max_tokens,
                tree_max_layers=indexing_cfg.tree_max_layers,
                random_seed=indexing_cfg.random_seed,
                summarizer_model=indexing_cfg.summarizer_model,
                summarizer_api_base=runtime_cfg.llm_api_base,
                summarizer_api_key=runtime_cfg.ark_api_key,
                summary_max_tokens=indexing_cfg.summarizer_max_tokens,
                relationship_type=indexing_cfg.relationship_type,
            )
            tree_builder = GroupBRecursiveTreeBuilder(
                repository=repo,
                embedding_model=embedding_model,
                token_counter=token_counter,
                config=tree_cfg,
            )
            tree_stats = tree_builder.grow_tree(start_layer=0)
            logger.info(
                "Recursive tree growth completed: layers_grown=%d, summary_nodes=%d, stopped_by_condition=%d, stopped_by_max_layers=%d",
                tree_stats.get("layers_grown", 0),
                tree_stats.get("summary_nodes_created", 0),
                tree_stats.get("stopped_by_condition", 0),
                tree_stats.get("stopped_by_max_layers", 0),
            )

        logger.info("Group B leaf ingestion completed: inserted=%d", inserted)
    finally:
        repo.close()


if __name__ == "__main__":
    main()