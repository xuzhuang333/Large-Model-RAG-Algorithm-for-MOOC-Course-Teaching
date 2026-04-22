from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_B_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(GROUP_B_DIR)
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    GROUP_RESOURCE_NAMES,
    ark_chat_completion,
    load_runtime_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupB_CollapsedTree_Pipeline")


@dataclass
class GroupBCollapsedTreeConfig:
    qa_dataset_path: str
    retrieval_output_file: str
    answer_output_file: str
    top_k: int = 20
    include_subsumed_layer0: bool = False
    context_max_chars: int = 7000
    relationship_type: str = "GROUP_B_PARENT_OF"
    llm_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    llm_api_base: str = DEFAULT_ARK_API_BASE
    llm_temperature: float = 0.3
    llm_max_tokens: int = 520
    only_qid: str = ""


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sanitize_rel_type(raw_rel: str | None) -> str:
    candidate = (raw_rel or "").strip().upper()
    if not candidate:
        return "GROUP_B_PARENT_OF"
    if all(ch.isalnum() or ch == "_" for ch in candidate) and (candidate[0].isalpha() or candidate[0] == "_"):
        return candidate
    return "GROUP_B_PARENT_OF"


def build_config_from_env() -> GroupBCollapsedTreeConfig:
    qa_dataset_path = os.getenv(
        "QA_DATASET_PATH",
        os.path.join(WORKSPACE_DIR, "QAdata", "qa_dataset.json"),
    )
    answer_dir = os.path.join(GROUP_B_DIR, "Answer")
    retrieval_output_file = os.getenv(
        "B_RETRIEVAL_OUTPUT_FILE",
        os.path.join(answer_dir, "group_b_collapsed_retrieval.json"),
    )
    answer_output_file = os.getenv(
        "B_ANSWER_OUTPUT_FILE",
        os.path.join(answer_dir, "group_b_collapsed_answers.json"),
    )

    return GroupBCollapsedTreeConfig(
        qa_dataset_path=qa_dataset_path,
        retrieval_output_file=retrieval_output_file,
        answer_output_file=answer_output_file,
        top_k=max(1, int(os.getenv("B_TOP_K", "20"))),
        include_subsumed_layer0=_parse_bool(os.getenv("B_INCLUDE_SUBSUMED_LAYER0"), default=False),
        context_max_chars=max(1000, int(os.getenv("B_CONTEXT_MAX_CHARS", "7000"))),
        relationship_type=_sanitize_rel_type(os.getenv("GROUP_B_REL_TYPE", "GROUP_B_PARENT_OF")),
        llm_model=os.getenv("B_GENERATOR_MODEL", DEFAULT_DEEPSEEK_ENDPOINT),
        llm_api_base=os.getenv("ARK_API_BASE", DEFAULT_ARK_API_BASE),
        llm_temperature=float(os.getenv("B_LLM_TEMPERATURE", "0.3")),
        llm_max_tokens=max(64, int(os.getenv("B_LLM_MAX_TOKENS", "520"))),
        only_qid=os.getenv("ONLY_QID", "").strip(),
    )


class GroupBCollapsedTreeRetriever:
    def __init__(self, uri: str, user: str, password: str, relationship_type: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.relationship_type = relationship_type
        self.node_label = GROUP_RESOURCE_NAMES["B"]["node_label"]
        self.index_name = GROUP_RESOURCE_NAMES["B"]["index_name"]

        logger.info("Loading BGE model for Group B collapsed retrieval...")
        self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def close(self) -> None:
        self.driver.close()

    def _vectorize_query(self, user_query: str) -> list[float]:
        vector = self.embedding_model.encode([user_query], normalize_embeddings=True)[0]
        return [float(x) for x in vector]

    def retrieve_global_topk(self, user_query: str, top_k: int) -> list[dict]:
        query_vec = self._vectorize_query(user_query)
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $query_vec)
        YIELD node, score
        WHERE node:{self.node_label}
        OPTIONAL MATCH (node)-[:{self.relationship_type}]->(child:{self.node_label})
        WITH node, score, collect(DISTINCT child.node_id) AS child_node_ids
        OPTIONAL MATCH (parent:{self.node_label})-[:{self.relationship_type}]->(node)
        RETURN
            node.node_id AS node_id,
            coalesce(node.text, '') AS text,
            coalesce(node.layer, 0) AS layer,
            score AS score,
            child_node_ids AS child_node_ids,
            collect(DISTINCT parent.node_id) AS parent_node_ids,
            coalesce(node.source_file, '') AS source_file,
            coalesce(node.source_type, '') AS source_type
        ORDER BY score DESC
        LIMIT $k
        """
        with self.driver.session() as session:
            result = session.run(
                cypher,
                index_name=self.index_name,
                k=top_k,
                query_vec=query_vec,
            )
            rows: list[dict] = []
            for record in result:
                rows.append(
                    {
                        "node_id": record["node_id"],
                        "text": record["text"],
                        "layer": int(record["layer"]),
                        "score": float(record["score"]),
                        "child_node_ids": [cid for cid in (record["child_node_ids"] or []) if cid],
                        "parent_node_ids": [pid for pid in (record["parent_node_ids"] or []) if pid],
                        "source_file": record["source_file"],
                        "source_type": record["source_type"],
                    }
                )
            return rows


def resolve_lineage_and_bucket(raw_nodes: list[dict]) -> tuple[dict[int, list[dict]], dict[str, int]]:
    layered_buckets: dict[int, list[dict]] = {}
    retrieved_ids = {node["node_id"] for node in raw_nodes}
    subsumed_count = 0

    for node in raw_nodes:
        layer = int(node["layer"])
        parent_ids = node.get("parent_node_ids") or []
        is_subsumed = layer == 0 and any(pid in retrieved_ids for pid in parent_ids)
        if is_subsumed:
            subsumed_count += 1

        normalized_node = {
            "node_id": node["node_id"],
            "text": node["text"],
            "layer": layer,
            "score": float(node["score"]),
            "child_node_ids": node.get("child_node_ids", []),
            "parent_node_ids": parent_ids,
            "source_file": node.get("source_file", ""),
            "source_type": node.get("source_type", ""),
            "is_subsumed": is_subsumed,
        }
        layered_buckets.setdefault(layer, []).append(normalized_node)

    for layer in layered_buckets:
        layered_buckets[layer].sort(key=lambda x: x["score"], reverse=True)

    metrics = {
        "raw_retrieved_count": len(raw_nodes),
        "unique_layers": len(layered_buckets),
        "subsumed_layer0_count": subsumed_count,
    }
    return layered_buckets, metrics


def assemble_macro_to_micro_context(
    layered_buckets: dict[int, list[dict]],
    include_subsumed_layer0: bool,
    context_max_chars: int,
) -> tuple[str, list[dict], dict[str, int]]:
    ordered_layers = sorted(layered_buckets.keys(), reverse=True)
    parts: list[str] = []
    selected_nodes: list[dict] = []

    current_chars = 0
    skipped_subsumed = 0

    for layer in ordered_layers:
        if layer >= 2:
            section_header = f"\n【宏观主题概述 - 第 {layer} 层】\n"
        elif layer == 1:
            section_header = "\n【微观关联摘要】\n"
        else:
            section_header = "\n【底层切片细节】\n"

        if current_chars + len(section_header) > context_max_chars:
            break
        parts.append(section_header)
        current_chars += len(section_header)

        for node in layered_buckets[layer]:
            if layer == 0 and node["is_subsumed"] and not include_subsumed_layer0:
                skipped_subsumed += 1
                continue

            node_text = str(node.get("text", "") or "").strip()
            if not node_text:
                continue

            # Keep only instructional content in generation context.
            line = f"- {node_text}\n"
            if current_chars + len(line) > context_max_chars:
                break

            parts.append(line)
            current_chars += len(line)
            selected_nodes.append(node)

    assembled_text = "".join(parts).strip()
    metrics = {
        "assembled_chars": current_chars,
        "selected_nodes": len(selected_nodes),
        "skipped_subsumed_layer0": skipped_subsumed,
        "ordered_layer_count": len(ordered_layers),
    }
    return assembled_text, selected_nodes, metrics


def generate_final_answer(
    user_query: str,
    assembled_context: str,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    prompt = f"""
你是一个严谨的学术助教。请仔细阅读以下按宏观到微观结构组织的参考上下文。你的任务是仅根据这些上下文回答用户的问题。
在回答时，请优先使用宏观主题定基调，并用底层切片细节作为具体论据支撑。

【结构化参考上下文】
{assembled_context}

【用户问题】
{user_query}

请直接给出回答：
"""

    return ark_chat_completion(
        model=model,
        prompt=prompt,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=180,
    )


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_group_b_collapsed_tree_pipeline() -> None:
    cfg = build_config_from_env()
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
        default_llm_api_base=cfg.llm_api_base,
    )

    if not runtime_cfg.ark_api_key:
        raise RuntimeError("ARK_API_KEY not found. Please set ARK_API_KEY before running Group B pipeline.")
    if not os.path.exists(cfg.qa_dataset_path):
        raise FileNotFoundError(f"QA dataset not found: {cfg.qa_dataset_path}")

    ensure_parent_dir(cfg.retrieval_output_file)
    ensure_parent_dir(cfg.answer_output_file)

    with open(cfg.qa_dataset_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    retriever = GroupBCollapsedTreeRetriever(
        uri=runtime_cfg.neo4j_uri,
        user=runtime_cfg.neo4j_user,
        password=runtime_cfg.neo4j_password,
        relationship_type=cfg.relationship_type,
    )

    retrieval_logs: list[dict] = []
    answer_logs: list[dict] = []
    matched_count = 0

    try:
        for item in qa_data:
            qid = str(item.get("id", "")).strip()
            qtype = item.get("type", "")
            user_query = item.get("question", "")

            if cfg.only_qid and qid != cfg.only_qid:
                continue

            matched_count += 1
            logger.info("Processing question: %s (%s)", qid, qtype)

            retrieval_started = time.time()
            raw_nodes = retriever.retrieve_global_topk(user_query=user_query, top_k=cfg.top_k)
            layered_buckets, lineage_metrics = resolve_lineage_and_bucket(raw_nodes)
            assembled_context, selected_nodes, assemble_metrics = assemble_macro_to_micro_context(
                layered_buckets=layered_buckets,
                include_subsumed_layer0=cfg.include_subsumed_layer0,
                context_max_chars=cfg.context_max_chars,
            )
            retrieval_elapsed_ms = int((time.time() - retrieval_started) * 1000)

            generation_started = time.time()
            final_answer = generate_final_answer(
                user_query=user_query,
                assembled_context=assembled_context,
                model=cfg.llm_model,
                api_base=runtime_cfg.llm_api_base,
                api_key=runtime_cfg.ark_api_key,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
            )
            generation_elapsed_ms = int((time.time() - generation_started) * 1000)

            ordered_layers = sorted(layered_buckets.keys(), reverse=True)
            retrieved_context = {
                "ordered_layers": ordered_layers,
                "layered_buckets": {str(layer): nodes for layer, nodes in layered_buckets.items()},
                "final_structured_context": assembled_context,
            }

            retrieval_logs.append(
                {
                    "id": qid,
                    "type": qtype,
                    "question": user_query,
                    "top_k": cfg.top_k,
                    "raw_retrieved_nodes": raw_nodes,
                    "retrieved_context": retrieved_context,
                    "lineage_metrics": lineage_metrics,
                    "assembly_metrics": assemble_metrics,
                    "retrieval_elapsed_ms": retrieval_elapsed_ms,
                }
            )

            answer_logs.append(
                {
                    "id": qid,
                    "type": qtype,
                    "question": user_query,
                    "final_answer": final_answer,
                    "retrieved_context": retrieved_context,
                    "selected_node_ids": [node["node_id"] for node in selected_nodes],
                    "retrieval_elapsed_ms": retrieval_elapsed_ms,
                    "generation_elapsed_ms": generation_elapsed_ms,
                    "llm_model": cfg.llm_model,
                    "llm_temperature": cfg.llm_temperature,
                    "llm_max_tokens": cfg.llm_max_tokens,
                }
            )

            logger.info(
                "Done %s: raw_nodes=%d selected_nodes=%d layers=%s retrieval_ms=%d generation_ms=%d",
                qid,
                len(raw_nodes),
                len(selected_nodes),
                ordered_layers,
                retrieval_elapsed_ms,
                generation_elapsed_ms,
            )

        if cfg.only_qid and matched_count == 0:
            logger.warning("ONLY_QID=%s not found in dataset.", cfg.only_qid)

        with open(cfg.retrieval_output_file, "w", encoding="utf-8") as f:
            json.dump(retrieval_logs, f, ensure_ascii=False, indent=2)

        with open(cfg.answer_output_file, "w", encoding="utf-8") as f:
            json.dump(answer_logs, f, ensure_ascii=False, indent=2)

        logger.info("Saved retrieval logs to: %s", cfg.retrieval_output_file)
        logger.info("Saved answer logs to: %s", cfg.answer_output_file)
    finally:
        retriever.close()


if __name__ == "__main__":
    run_group_b_collapsed_tree_pipeline()
