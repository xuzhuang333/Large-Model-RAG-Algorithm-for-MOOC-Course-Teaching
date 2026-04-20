#该文件是Group C课程大纲节点的嵌入回填脚本，负责从Neo4j图数据库中提取课程大纲节点的文本信息（如节点名称），使用SentenceTransformer模型生成文本嵌入，并将嵌入结果回写到对应的节点属性中。脚本支持批量处理、过滤特定周次的节点，以及在回写前进行干运行验证。完成后还会创建向量索引以优化后续的相似度检索。
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
GROUP_B_DIR = os.path.join(PROJECT_DIR, "Group B")

for path in (PROJECT_DIR, GROUP_B_DIR):
    if path not in sys.path:
        sys.path.append(path)

from neo4j_ops import GroupCStaticRepository
from shared_retrieval_utils import load_runtime_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupC_Syllabus_Embedding_Backfill")


@dataclass
class BackfillConfig:
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_batch_size: int = 64
    vector_index_name: str = "group_c_syllabus_vector_index"
    embedding_dim: int = 512
    dry_run: bool = False


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config_from_env() -> BackfillConfig:
    return BackfillConfig(
        embedding_model_name=os.getenv("GROUP_C_SYLLABUS_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_batch_size=max(1, int(os.getenv("GROUP_C_SYLLABUS_EMBED_BATCH_SIZE", "64"))),
        vector_index_name=os.getenv("GROUP_C_SYLLABUS_VECTOR_INDEX_NAME", "group_c_syllabus_vector_index"),
        embedding_dim=max(1, int(os.getenv("GROUP_C_SYLLABUS_EMBED_DIM", "512"))),
        dry_run=_parse_bool(os.getenv("GROUP_C_SYLLABUS_DRY_RUN"), default=False),
    )


def _create_syllabus_vector_index(repo: GroupCStaticRepository, index_name: str, embedding_dim: int) -> None:
    query = """
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:GroupC_SyllabusNode)
    ON (n.embedding)
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {embedding_dim},
      `vector.similarity_function`: 'cosine'
    }}}}
    """
    with repo.driver.session() as session:
        session.run(query.format(index_name=index_name, embedding_dim=embedding_dim))


def _fetch_syllabus_nodes(
    repo: GroupCStaticRepository,
) -> list[dict[str, Any]]:
    query = """
    MATCH (s:GroupC_SyllabusNode)
    RETURN
      s.node_id AS node_id,
      s.name AS name,
      s.depth AS depth,
      s.week_tag AS week_tag,
      s.abs_path AS abs_path
    ORDER BY s.depth DESC, s.abs_path ASC
    """

    with repo.driver.session() as session:
        records = list(session.run(query))

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "node_id": record["node_id"],
            "name": record["name"],
            "abs_path": record["abs_path"] or "",
        }
        rows.append(row)

    return rows


def _build_name_only_text(row: dict[str, Any]) -> tuple[str, str]:
    name = (row.get("name") or "").strip()
    if not name:
        name = "UNNAMED_SYLLABUS_NODE"
    return name, "name_only"


def _upsert_syllabus_embeddings(repo: GroupCStaticRepository, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    query = """
    UNWIND $rows AS row
    MATCH (s:GroupC_SyllabusNode {node_id: row.node_id})
    SET s.embedding_source = row.embedding_source,
        s.embedding_updated_at = datetime()
    WITH s, row
    CALL db.create.setNodeVectorProperty(s, 'embedding', row.embedding)
    RETURN count(s) AS count
    """

    with repo.driver.session() as session:
        record = session.run(query, rows=rows).single()
        return int(record["count"]) if record else 0


def run_backfill() -> None:
    cfg = build_config_from_env()

    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
    )

    repo = GroupCStaticRepository(
        uri=runtime_cfg.neo4j_uri,
        user=runtime_cfg.neo4j_user,
        password=runtime_cfg.neo4j_password,
    )

    try:
        source_rows = _fetch_syllabus_nodes(
            repo=repo,
        )
        if not source_rows:
            logger.warning("No GroupC_SyllabusNode rows found for embedding backfill.")
            return

        contexts: list[str] = []
        payload_rows: list[dict[str, Any]] = []
        source_stats: dict[str, int] = {"name_only": 0}

        for row in source_rows:
            context, source = _build_name_only_text(row)
            contexts.append(context)
            payload_rows.append(
                {
                    "node_id": row["node_id"],
                    "embedding_source": source,
                }
            )
            source_stats[source] = source_stats.get(source, 0) + 1

        logger.info(
            "Backfill source prepared: nodes=%d, name_only=%d",
            len(payload_rows),
            source_stats.get("name_only", 0),
        )

        if cfg.dry_run:
            logger.info("GROUP_C_SYLLABUS_DRY_RUN=true, skip embedding write.")
            return

        embedding_model = SentenceTransformer(cfg.embedding_model_name)
        vectors = embedding_model.encode(
            contexts,
            normalize_embeddings=True,
            batch_size=cfg.embedding_batch_size,
        )

        write_rows: list[dict[str, Any]] = []
        for item, vec in zip(payload_rows, vectors):
            write_rows.append(
                {
                    "node_id": item["node_id"],
                    "embedding_source": item["embedding_source"],
                    "embedding": [float(x) for x in vec],
                }
            )

        updated = _upsert_syllabus_embeddings(repo, write_rows)
        _create_syllabus_vector_index(repo, cfg.vector_index_name, cfg.embedding_dim)

        logger.info(
            "GroupC_SyllabusNode embedding backfill done: updated=%d, index=%s",
            updated,
            cfg.vector_index_name,
        )
    finally:
        repo.close()


if __name__ == "__main__":
    run_backfill()
