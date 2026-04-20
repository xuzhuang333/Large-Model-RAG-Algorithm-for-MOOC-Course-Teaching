from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from itertools import combinations
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
GROUP_B_DIR = os.path.join(PROJECT_DIR, "Group B")

for path in (PROJECT_DIR, GROUP_B_DIR):
    if path not in sys.path:
        sys.path.append(path)

from shared_retrieval_utils import (  # noqa: E402
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    ark_chat_completion,
    get_adaptive_generation_params,
    load_runtime_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupC_Dual_Retriever")


@dataclass
class GroupCDualRetrieverConfig:
    top_k_micro: int = 15
    top_k_macro: int = 5
    top_n_final: int = 10
    gravity_alpha: float = 0.3
    hard_neg_tolerance: float = 0.05
    hard_neg_top_m: int = 5
    diversity_max_per_syllabus: int = 2
    enable_legacy_prereq_edge: bool = True
    hard_neg_lca_depth_threshold: int = 1
    rewrite_max_keywords: int = 8
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    syllabus_vector_index: str = "group_c_syllabus_vector_index"
    text_vector_index: str = "group_c_text_vector_index"
    code_vector_index: str = "group_c_code_vector_index"
    llm_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    llm_api_base: str = DEFAULT_ARK_API_BASE
    state_weight_rho: float = 0.25
    user_active_context_n: int = 5
    route_timeout_sec: int = 90
    lambda_base: float = 0.1
    ltm_resistance_mu: float = 0.5
    stm_beta: float = 0.6
    ltm_alpha_min: float = 0.05
    struggle_gamma: float = 0.3
    backprop_stm_gate: float = 0.8
    topic_shift_penalty_factor: float = 1.5
    state_review_bonus: float = 0.5
    state_struggle_bonus: float = 0.3
    state_mastery_penalty: float = -0.2


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config_from_env() -> GroupCDualRetrieverConfig:
    return GroupCDualRetrieverConfig(
        top_k_micro=max(1, int(os.getenv("GROUP_C_TOP_K_MICRO", "15"))),
        top_k_macro=max(1, int(os.getenv("GROUP_C_TOP_K_MACRO", "5"))),
        top_n_final=max(1, int(os.getenv("GROUP_C_TOP_N_FINAL", "10"))),
        gravity_alpha=float(os.getenv("GROUP_C_GRAVITY_ALPHA", "0.3")),
        hard_neg_tolerance=max(0.0, float(os.getenv("GROUP_C_HARD_NEG_TOLERANCE", "0.05"))),
        hard_neg_top_m=max(2, int(os.getenv("GROUP_C_HARD_NEG_TOP_M", "5"))),
        diversity_max_per_syllabus=max(1, int(os.getenv("GROUP_C_DIVERSITY_MAX_PER_SYLLABUS", "2"))),
        enable_legacy_prereq_edge=_parse_bool(os.getenv("GROUP_C_ENABLE_LEGACY_PREREQ_EDGE"), default=True),
        hard_neg_lca_depth_threshold=max(0, int(os.getenv("GROUP_C_HARD_NEG_LCA_DEPTH_THRESHOLD", "1"))),
        rewrite_max_keywords=max(1, int(os.getenv("GROUP_C_REWRITE_MAX_KEYWORDS", "8"))),
        embedding_model_name=os.getenv("GROUP_C_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        syllabus_vector_index=os.getenv("GROUP_C_SYLLABUS_VECTOR_INDEX_NAME", "group_c_syllabus_vector_index"),
        text_vector_index=os.getenv("GROUP_C_TEXT_VECTOR_INDEX_NAME", "group_c_text_vector_index"),
        code_vector_index=os.getenv("GROUP_C_CODE_VECTOR_INDEX_NAME", "group_c_code_vector_index"),
        llm_model=os.getenv("GROUP_C_SUMMARY_MODEL", DEFAULT_DEEPSEEK_ENDPOINT),
        llm_api_base=os.getenv("ARK_API_BASE", DEFAULT_ARK_API_BASE),
        state_weight_rho=float(os.getenv("GROUP_C_STATE_WEIGHT_RHO", "0.25")),
        user_active_context_n=max(1, int(os.getenv("GROUP_C_USER_ACTIVE_CONTEXT_N", "5"))),
        route_timeout_sec=max(30, int(os.getenv("GROUP_C_ROUTE_TIMEOUT_SEC", "90"))),
        lambda_base=max(0.0, float(os.getenv("GROUP_C_LAMBDA_BASE", "0.1"))),
        ltm_resistance_mu=max(0.0, min(1.0, float(os.getenv("GROUP_C_LTM_RESISTANCE_MU", "0.5")))),
        stm_beta=max(0.0, min(1.0, float(os.getenv("GROUP_C_STM_BETA", "0.6")))),
        ltm_alpha_min=max(0.0, min(1.0, float(os.getenv("GROUP_C_LTM_ALPHA_MIN", "0.05")))),
        struggle_gamma=max(0.0, min(1.0, float(os.getenv("GROUP_C_STRUGGLE_GAMMA", "0.3")))),
        backprop_stm_gate=max(0.0, min(1.0, float(os.getenv("GROUP_C_BACKPROP_STM_GATE", "0.8")))),
        topic_shift_penalty_factor=max(1.0, float(os.getenv("GROUP_C_TOPIC_SHIFT_PENALTY_FACTOR", "1.5"))),
        state_review_bonus=float(os.getenv("GROUP_C_STATE_REVIEW_BONUS", "0.5")),
        state_struggle_bonus=float(os.getenv("GROUP_C_STATE_STRUGGLE_BONUS", "0.3")),
        state_mastery_penalty=float(os.getenv("GROUP_C_STATE_MASTERY_PENALTY", "-0.2")),
    )


@dataclass
class RetrievalResult:
    original_query: str
    expanded_global_keywords: list[str]
    expanded_local_keywords: list[str]
    candidates: list[dict[str, Any]]
    is_contrastive_triggered: bool
    contrastive_pairs: list[dict[str, Any]]
    pedagogical_context: list[str]
    state_update_context: dict[str, Any] = field(default_factory=dict)
    debug_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_json_parse(raw_text: str) -> dict[str, Any]:
    payload = (raw_text or "").strip()
    if not payload:
        return {}

    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(payload[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}


def _normalize_scores(rows: list[dict[str, Any]], id_key: str, score_key: str) -> dict[str, float]:
    if not rows:
        return {}

    ids: list[str] = []
    scores: list[float] = []
    for row in rows:
        item_id = str(row.get(id_key, "")).strip()
        if not item_id:
            continue
        ids.append(item_id)
        scores.append(float(row.get(score_key, 0.0)))

    if not scores:
        return {}

    arr = np.asarray(scores, dtype=np.float64)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if abs(max_val - min_val) < 1e-12:
        return {item_id: 1.0 for item_id in ids}

    norm = (arr - min_val) / (max_val - min_val)
    return {item_id: float(val) for item_id, val in zip(ids, norm.tolist())}


def evaluate_retrieval_metrics(
    candidates: list[dict[str, Any]],
    golden_parent_syllabus_ids: list[str],
    k: int,
) -> dict[str, float]:
    """Compute Recall@K, MRR and Hit@1 using parent syllabus ids from ranked candidates."""
    top_k = max(1, int(k))
    golden_set = {str(item).strip() for item in golden_parent_syllabus_ids if str(item).strip()}
    if not golden_set:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "hit_at_1": 0.0,
        }

    ranked_parent_ids: list[str] = []
    seen: set[str] = set()
    for row in candidates:
        parent_id = str(row.get("parent_syllabus_id", "")).strip()
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        ranked_parent_ids.append(parent_id)

    cutoff = ranked_parent_ids[:top_k]
    hit_count = sum(1 for item in cutoff if item in golden_set)
    recall_at_k = float(hit_count / len(golden_set))

    reciprocal_rank = 0.0
    for rank, item in enumerate(ranked_parent_ids, start=1):
        if item in golden_set:
            reciprocal_rank = 1.0 / rank
            break

    hit_at_1 = 1.0 if ranked_parent_ids and ranked_parent_ids[0] in golden_set else 0.0
    return {
        "recall_at_k": recall_at_k,
        "mrr": float(reciprocal_rank),
        "hit_at_1": float(hit_at_1),
    }


def _parse_optional_int_from_any(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    try:
        text = str(raw).strip()
        if not text:
            return None
        return int(text)
    except Exception:
        return None


def _parse_optional_float_from_any(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        text = str(raw).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _load_group_c_input_samples(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_samples: list[Any]
    if isinstance(payload, list):
        raw_samples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        raw_samples = payload["samples"]
    else:
        raise ValueError("Group C input json must be a list or an object with key 'samples'.")

    samples: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, dict):
            continue

        item_id = str(raw.get("id", f"GC_Q{idx:03d}")).strip() or f"GC_Q{idx:03d}"
        q_type = str(raw.get("type", "")).strip()
        question = str(raw.get("question", "")).strip()
        user_id = str(raw.get("user_id", "")).strip() or None
        current_turn = _parse_optional_int_from_any(raw.get("current_turn"))
        qa_score = _parse_optional_float_from_any(raw.get("qa_score"))
        current_struggle = _parse_optional_float_from_any(raw.get("current_struggle"))

        golden_raw = raw.get("golden_parent_syllabus_ids", [])
        if not isinstance(golden_raw, list):
            golden_raw = []
        golden_parent_syllabus_ids = [
            str(item).strip() for item in golden_raw if str(item).strip()
        ]

        samples.append(
            {
                "id": item_id,
                "type": q_type,
                "question": question,
                "user_id": user_id,
                "current_turn": current_turn,
                "qa_score": qa_score,
                "current_struggle": current_struggle,
                "golden_parent_syllabus_ids": golden_parent_syllabus_ids,
                "notes": raw.get("notes", ""),
            }
        )
    return samples


class GroupCDualRetriever:
    def __init__(self, cfg: GroupCDualRetrieverConfig | None = None):
        self.cfg = cfg or build_config_from_env()

        runtime_cfg = load_runtime_config(
            default_uri="bolt://localhost:7687",
            default_user="neo4j",
            default_password="12345678",
            default_llm_api_base=self.cfg.llm_api_base,
        )

        self.driver = GraphDatabase.driver(
            runtime_cfg.neo4j_uri,
            auth=(runtime_cfg.neo4j_user, runtime_cfg.neo4j_password),
        )
        self.ark_api_key = runtime_cfg.ark_api_key
        self.embedding_model = SentenceTransformer(self.cfg.embedding_model_name)

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _parse_optional_int(raw: str | None) -> int | None:
        if raw is None:
            return None
        value = raw.strip()
        if not value:
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _parse_optional_float(raw: str | None) -> float | None:
        if raw is None:
            return None
        value = raw.strip()
        if not value:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _expand_query_keywords(self, query: str) -> tuple[list[str], list[str]]:
        if not self.ark_api_key:
            logger.warning("ARK_API_KEY missing, fallback to raw query without expansion.")
            return [], []

        prompt = f"""
你是一位经验丰富的计算机课程教师。
请分析学生问题，并提取两组关键词，严格输出 JSON：
1. global_keywords: 面向章节主题、底层原理、宏观知识域。
2. local_keywords: 面向函数名、代码关键字、语法术语、API 名称。

学生问题: {query}

输出要求：
1. 只输出 JSON，不要输出解释文本。
2. JSON 结构必须是：
{{
  "global_keywords": ["..."],
  "local_keywords": ["..."]
}}
3. 每组关键词最多 {self.cfg.rewrite_max_keywords} 个。
"""

        try:
            temperature, adaptive_max_tokens = get_adaptive_generation_params(prompt, task="rewrite")
            max_tokens = max(160, adaptive_max_tokens)
            response = ark_chat_completion(
                model=self.cfg.llm_model,
                prompt=prompt,
                api_base=self.cfg.llm_api_base,
                api_key=self.ark_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=90,
            )
            parsed = _safe_json_parse(response)
            global_keywords = parsed.get("global_keywords", [])
            local_keywords = parsed.get("local_keywords", [])

            if not isinstance(global_keywords, list):
                global_keywords = []
            if not isinstance(local_keywords, list):
                local_keywords = []

            gk = [str(item).strip() for item in global_keywords if str(item).strip()]
            lk = [str(item).strip() for item in local_keywords if str(item).strip()]
            return gk[: self.cfg.rewrite_max_keywords], lk[: self.cfg.rewrite_max_keywords]
        except Exception as exc:
            logger.warning("Query expansion failed, fallback to raw query: %s", exc)
            return [], []

    def _classify_query_route(
        self,
        user_query: str,
        recent_nodes: list[dict[str, Any]],
        target_syllabus_ids: list[str],
    ) -> str:
        if not recent_nodes:
            return "Topic-Shift"

        recent_syllabus_ids = {
            str(item.get("linked_syllabus_id", "")).strip()
            for item in recent_nodes
            if str(item.get("linked_syllabus_id", "")).strip()
        }
        if recent_syllabus_ids & set(target_syllabus_ids):
            return "Drill-down"

        if not self.ark_api_key:
            return "Lateral"

        compact_recent = [
            {
                "qa_node_id": item.get("qa_node_id"),
                "linked_syllabus_id": item.get("linked_syllabus_id"),
                "last_interact_turn": item.get("last_interact_turn"),
                "stm_score": item.get("stm_score"),
                "ltm_score": item.get("ltm_score"),
                "struggle_index": item.get("struggle_index"),
            }
            for item in recent_nodes
        ]

        prompt = f"""
你是教学认知轨迹路由器。请根据用户当前问题、最近活跃节点和命中的目标大纲节点，判断路由类型。

可选类别仅允许：
1. Drill-down
2. Lateral
3. Topic-Shift

用户问题:
{user_query}

最近活跃节点(JSON):
{json.dumps(compact_recent, ensure_ascii=False)}

本轮命中的目标大纲节点(JSON):
{json.dumps(target_syllabus_ids, ensure_ascii=False)}

输出要求：
1. 只输出 JSON，不要输出任何解释。
2. JSON 格式严格为：
{{"route": "Drill-down|Lateral|Topic-Shift"}}
"""

        try:
            temperature, adaptive_max_tokens = get_adaptive_generation_params(prompt, task="evaluate")
            response = ark_chat_completion(
                model=self.cfg.llm_model,
                prompt=prompt,
                api_base=self.cfg.llm_api_base,
                api_key=self.ark_api_key,
                temperature=temperature,
                max_tokens=max(128, adaptive_max_tokens),
                timeout_sec=self.cfg.route_timeout_sec,
            )
            parsed = _safe_json_parse(response)
            route = str(parsed.get("route", "")).strip()
            if route in {"Drill-down", "Lateral", "Topic-Shift"}:
                return route
        except Exception as exc:
            logger.warning("Route classification failed, fallback to heuristic: %s", exc)

        return "Lateral"

    @staticmethod
    def _compose_query(query: str, keywords: list[str]) -> str:
        if not keywords:
            return query
        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            norm = kw.strip()
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(norm)
        if not unique:
            return query
        return f"{query} {' '.join(unique)}"

    def _embed_query(self, query: str) -> list[float]:
        vec = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

    def _retrieve_macro_track(self, macro_query: str) -> list[dict[str, Any]]:
        query_vec = self._embed_query(macro_query)
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_vec)
        YIELD node, score
        WITH node, score
        WHERE node:GroupC_SyllabusNode
        RETURN node.node_id AS syllabus_id,
               coalesce(node.name, '') AS syllabus_name,
               score AS macro_score
        ORDER BY macro_score DESC
        """

        with self.driver.session() as session:
            records = list(
                session.run(
                    cypher,
                    index_name=self.cfg.syllabus_vector_index,
                    k=self.cfg.top_k_macro,
                    query_vec=query_vec,
                )
            )

        rows: list[dict[str, Any]] = []
        for record in records:
            syllabus_id = str(record["syllabus_id"] or "").strip()
            if not syllabus_id:
                continue
            rows.append(
                {
                    "syllabus_id": syllabus_id,
                    "syllabus_name": record["syllabus_name"],
                    "macro_score": float(record["macro_score"]),
                }
            )
        return rows

    def _retrieve_micro_track(self, micro_query: str) -> list[dict[str, Any]]:
        query_vec = self._embed_query(micro_query)

        text_cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_vec)
        YIELD node, score
        WITH node, score
        WHERE node:GroupC_TextSnippet
        MATCH (s:GroupC_SyllabusNode)-[:HAS_TEXT]->(node)
        WHERE node.snippet_id IS NOT NULL
          AND node.text IS NOT NULL
          AND trim(node.text) <> ''
        RETURN node.snippet_id AS snippet_id,
               s.node_id AS parent_syllabus_id,
               node.text AS content,
               'text' AS content_type,
               score AS micro_score
        ORDER BY micro_score DESC
        """

        code_cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $query_vec)
        YIELD node, score
        WITH node, score
        WHERE node:GroupC_CodeSnippet
        MATCH (s:GroupC_SyllabusNode)-[:HAS_CODE]->(node)
        WHERE node.snippet_id IS NOT NULL
          AND node.code IS NOT NULL
          AND trim(node.code) <> ''
        RETURN node.snippet_id AS snippet_id,
               s.node_id AS parent_syllabus_id,
               node.code AS content,
               'code' AS content_type,
               score AS micro_score
        ORDER BY micro_score DESC
        """

        with self.driver.session() as session:
            text_records = list(
                session.run(
                    text_cypher,
                    index_name=self.cfg.text_vector_index,
                    k=self.cfg.top_k_micro,
                    query_vec=query_vec,
                )
            )
            code_records = list(
                session.run(
                    code_cypher,
                    index_name=self.cfg.code_vector_index,
                    k=self.cfg.top_k_micro,
                    query_vec=query_vec,
                )
            )

        dedup: dict[str, dict[str, Any]] = {}
        for record in text_records + code_records:
            snippet_id = str(record["snippet_id"] or "").strip()
            parent_syllabus_id = str(record["parent_syllabus_id"] or "").strip()
            if not snippet_id or not parent_syllabus_id:
                continue
            payload = {
                "snippet_id": snippet_id,
                "parent_syllabus_id": parent_syllabus_id,
                "content": record["content"],
                "type": record["content_type"],
                "micro_score": float(record["micro_score"]),
            }
            prev = dedup.get(snippet_id)
            if prev is None or payload["micro_score"] > prev["micro_score"]:
                dedup[snippet_id] = payload

        return sorted(dedup.values(), key=lambda x: x["micro_score"], reverse=True)

    def _fuse_candidates(
        self,
        macro_rows: list[dict[str, Any]],
        micro_rows: list[dict[str, Any]],
        user_state_by_syllabus: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        user_state_by_syllabus = user_state_by_syllabus or {}
        macro_raw = {row["syllabus_id"]: float(row["macro_score"]) for row in macro_rows}
        macro_norm = _normalize_scores(macro_rows, id_key="syllabus_id", score_key="macro_score")
        micro_norm = _normalize_scores(micro_rows, id_key="snippet_id", score_key="micro_score")

        fused: list[dict[str, Any]] = []
        for row in micro_rows:
            snippet_id = row["snippet_id"]
            parent_syllabus_id = row["parent_syllabus_id"]
            micro_score_norm = micro_norm.get(snippet_id, 0.0)
            macro_score_norm = macro_norm.get(parent_syllabus_id, 0.0)

            state_weight, state_flags = self._compute_state_weight(
                user_state_by_syllabus.get(parent_syllabus_id)
            )
            final_score = (
                micro_score_norm
                + (self.cfg.gravity_alpha * macro_score_norm)
                + (self.cfg.state_weight_rho * state_weight)
            )

            fused.append(
                {
                    "snippet_id": snippet_id,
                    "parent_syllabus_id": parent_syllabus_id,
                    "content": row["content"],
                    "type": row["type"],
                    "micro_score": float(row["micro_score"]),
                    "macro_score": float(macro_raw.get(parent_syllabus_id, 0.0)),
                    "state_weight": float(state_weight),
                    "state_flags": state_flags,
                    "final_score": float(final_score),
                }
            )

        fused.sort(key=lambda x: x["final_score"], reverse=True)

        # Diversity limit by syllabus branch before final top-n truncation.
        selected: list[dict[str, Any]] = []
        per_syllabus: dict[str, int] = {}
        for row in fused:
            parent_id = row["parent_syllabus_id"]
            count = per_syllabus.get(parent_id, 0)
            if count >= self.cfg.diversity_max_per_syllabus:
                continue
            per_syllabus[parent_id] = count + 1
            selected.append(row)
            if len(selected) >= self.cfg.top_n_final:
                break

        return selected

    def _fetch_user_state_by_syllabus(
        self,
        user_id: str,
        syllabus_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        ids = [str(item).strip() for item in syllabus_ids if str(item).strip()]
        if not user_id or not ids:
            return {}

        query = """
        MATCH (n:GroupC_UserQANode {user_id: $user_id})
        WHERE n.linked_syllabus_id IN $syllabus_ids
        WITH n.linked_syllabus_id AS syllabus_id, n
        ORDER BY n.last_interact_turn DESC, n.create_turn DESC
        WITH syllabus_id, collect(n)[0] AS latest
        RETURN syllabus_id,
               latest.qa_node_id AS qa_node_id,
               coalesce(latest.stm_score, 0.0) AS stm_score,
               coalesce(latest.ltm_score, 0.0) AS ltm_score,
               coalesce(latest.struggle_index, 0.0) AS struggle_index,
               coalesce(latest.interact_count, 0) AS interact_count,
               coalesce(latest.last_interact_turn, 0) AS last_interact_turn
        """

        result: dict[str, dict[str, Any]] = {}
        with self.driver.session() as session:
            records = list(session.run(query, user_id=user_id, syllabus_ids=ids))
        for record in records:
            syllabus_id = str(record["syllabus_id"] or "").strip()
            if not syllabus_id:
                continue
            result[syllabus_id] = {
                "qa_node_id": record["qa_node_id"],
                "stm_score": float(record["stm_score"]),
                "ltm_score": float(record["ltm_score"]),
                "struggle_index": float(record["struggle_index"]),
                "interact_count": int(record["interact_count"]),
                "last_interact_turn": int(record["last_interact_turn"]),
            }
        return result

    def _compute_state_weight(self, state: dict[str, Any] | None) -> tuple[float, list[str]]:
        if not state:
            return 0.0, []

        stm_score = float(state.get("stm_score", 0.0))
        ltm_score = float(state.get("ltm_score", 0.0))
        struggle_index = float(state.get("struggle_index", 0.0))

        weight = 0.0
        flags: list[str] = []
        if stm_score < 0.3 and ltm_score > 0.4:
            weight += self.cfg.state_review_bonus
            flags.append("review_needed")
        if struggle_index > 0.7:
            weight += self.cfg.state_struggle_bonus
            flags.append("struggle_focus")
        if ltm_score > 0.9:
            weight += self.cfg.state_mastery_penalty
            flags.append("mastery_penalty")

        return float(weight), flags

    def _fetch_recent_user_nodes(self, user_id: str, n: int) -> list[dict[str, Any]]:
        if not user_id:
            return []
        query = """
        MATCH (n:GroupC_UserQANode {user_id: $user_id})
        OPTIONAL MATCH (p:GroupC_UserQANode)-[:DEEPENS_INTO|COMPARES_WITH|STARTS_NEW_TOPIC]->(n)
        RETURN n.qa_node_id AS qa_node_id,
               n.linked_syllabus_id AS linked_syllabus_id,
               coalesce(n.last_interact_turn, n.create_turn, 0) AS last_interact_turn,
               coalesce(n.stm_score, 0.0) AS stm_score,
               coalesce(n.ltm_score, 0.0) AS ltm_score,
               coalesce(n.struggle_index, 0.0) AS struggle_index,
               p.qa_node_id AS parent_qa_node_id
        ORDER BY last_interact_turn DESC, n.create_turn DESC
        LIMIT $n
        """
        with self.driver.session() as session:
            records = list(session.run(query, user_id=user_id, n=max(1, n)))
        rows: list[dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "qa_node_id": record["qa_node_id"],
                    "linked_syllabus_id": record["linked_syllabus_id"],
                    "last_interact_turn": int(record["last_interact_turn"]),
                    "stm_score": float(record["stm_score"]),
                    "ltm_score": float(record["ltm_score"]),
                    "struggle_index": float(record["struggle_index"]),
                    "parent_qa_node_id": record["parent_qa_node_id"],
                }
            )
        return rows

    def _ensure_user_root(self, user_id: str) -> None:
        query = """
        MERGE (u:GroupC_User {user_id: $user_id})
        ON CREATE SET u.created_at = datetime()
        SET u.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(query, user_id=user_id)

    def _create_user_qa_node(self, user_id: str, linked_syllabus_id: str, current_turn: int) -> str:
        qa_node_id = f"GC_UQA_{uuid4().hex[:24]}"
        query = """
        CREATE (n:GroupC_UserQANode {
            qa_node_id: $qa_node_id,
            user_id: $user_id,
            linked_syllabus_id: $linked_syllabus_id,
            create_turn: $current_turn,
            last_interact_turn: $current_turn,
            interact_count: 0,
            stm_score: 0.5,
            ltm_score: 0.5,
            struggle_index: 0.5,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN n.qa_node_id AS qa_node_id
        """
        with self.driver.session() as session:
            record = session.run(
                query,
                qa_node_id=qa_node_id,
                user_id=user_id,
                linked_syllabus_id=linked_syllabus_id,
                current_turn=current_turn,
            ).single()
        return str(record["qa_node_id"]) if record else qa_node_id

    def _attach_state_node_by_route(
        self,
        user_id: str,
        route: str,
        new_qa_node_id: str,
        recent_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "route": route,
            "new_qa_node_id": new_qa_node_id,
            "route_parent_qa_node_id": None,
            "abandoned_qa_node_id": None,
        }
        most_recent = recent_nodes[0] if recent_nodes else None

        with self.driver.session() as session:
            if route == "Drill-down" and most_recent and most_recent.get("qa_node_id"):
                parent_id = str(most_recent["qa_node_id"])
                query = """
                MATCH (p:GroupC_UserQANode {qa_node_id: $parent_id})
                MATCH (c:GroupC_UserQANode {qa_node_id: $child_id})
                MERGE (p)-[:DEEPENS_INTO]->(c)
                """
                session.run(query, parent_id=parent_id, child_id=new_qa_node_id)
                context["route_parent_qa_node_id"] = parent_id
                return context

            if route == "Lateral" and most_recent and most_recent.get("qa_node_id"):
                recent_id = str(most_recent["qa_node_id"])
                parent_query = """
                MATCH (p:GroupC_UserQANode)-[:DEEPENS_INTO|COMPARES_WITH|STARTS_NEW_TOPIC]->(c:GroupC_UserQANode {qa_node_id: $recent_id})
                RETURN p.qa_node_id AS parent_id
                LIMIT 1
                """
                rec = session.run(parent_query, recent_id=recent_id).single()
                anchor_id = str(rec["parent_id"]) if rec and rec.get("parent_id") else recent_id
                edge_query = """
                MATCH (a:GroupC_UserQANode {qa_node_id: $anchor_id})
                MATCH (c:GroupC_UserQANode {qa_node_id: $child_id})
                MERGE (a)-[:COMPARES_WITH]->(c)
                """
                session.run(edge_query, anchor_id=anchor_id, child_id=new_qa_node_id)
                context["route_parent_qa_node_id"] = anchor_id
                return context

            # Topic-Shift fallback path.
            self._ensure_user_root(user_id)
            shift_query = """
            MATCH (u:GroupC_User {user_id: $user_id})
            MATCH (c:GroupC_UserQANode {qa_node_id: $child_id})
            MERGE (u)-[:STARTS_NEW_TOPIC]->(c)
            """
            session.run(shift_query, user_id=user_id, child_id=new_qa_node_id)
            context["route"] = "Topic-Shift"
            context["abandoned_qa_node_id"] = most_recent.get("qa_node_id") if most_recent else None
            return context

    def _route_and_mutate_user_graph(
        self,
        user_id: str,
        current_turn: int,
        query: str,
        target_syllabus_ids: list[str],
    ) -> dict[str, Any]:
        if not user_id or current_turn is None or not target_syllabus_ids:
            return {}

        recent_nodes = self._fetch_recent_user_nodes(user_id, self.cfg.user_active_context_n)
        route = self._classify_query_route(query, recent_nodes, target_syllabus_ids)
        linked_syllabus_id = target_syllabus_ids[0]
        new_qa_node_id = self._create_user_qa_node(user_id, linked_syllabus_id, current_turn)
        context = self._attach_state_node_by_route(user_id, route, new_qa_node_id, recent_nodes)
        context["target_syllabus_ids"] = target_syllabus_ids
        return context

    def _fetch_user_nodes_for_consolidation(
        self,
        user_id: str,
        linked_syllabus_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        state_map = self._fetch_user_state_by_syllabus(user_id, linked_syllabus_ids)
        result: dict[str, dict[str, Any]] = {}
        for syllabus_id, payload in state_map.items():
            row = dict(payload)
            row["linked_syllabus_id"] = syllabus_id
            result[syllabus_id] = row
        return result

    def _calculate_memory_update(
        self,
        old_state: dict[str, Any],
        current_turn: int,
        qa_score: float,
        current_struggle: float,
    ) -> dict[str, Any]:
        stm_old = self._clamp01(float(old_state.get("stm_score", 0.0)))
        ltm_old = self._clamp01(float(old_state.get("ltm_score", 0.0)))
        struggle_old = self._clamp01(float(old_state.get("struggle_index", 0.0)))
        c_old = int(old_state.get("interact_count", 0))
        last_turn = int(old_state.get("last_interact_turn", current_turn))

        delta_t = max(0, int(current_turn) - int(last_turn))
        lambda_actual = self.cfg.lambda_base * (1.0 - self.cfg.ltm_resistance_mu * ltm_old)
        stm_decayed = stm_old * math.exp(-lambda_actual * delta_t)
        stm_new = self.cfg.stm_beta * qa_score + (1.0 - self.cfg.stm_beta) * stm_decayed
        stm_new = self._clamp01(stm_new)

        c_new = c_old + 1
        alpha_ltm = max(2.0 / (c_new + 1.0), self.cfg.ltm_alpha_min)
        effective_stm = stm_new * (1.0 - struggle_old)
        ltm_new = (1.0 - alpha_ltm) * ltm_old + alpha_ltm * effective_stm
        ltm_new = self._clamp01(ltm_new)

        struggle_new = self.cfg.struggle_gamma * current_struggle + (1.0 - self.cfg.struggle_gamma) * struggle_old
        struggle_new = self._clamp01(struggle_new)

        return {
            "interact_count": c_new,
            "stm_score": stm_new,
            "ltm_score": ltm_new,
            "struggle_index": struggle_new,
            "last_interact_turn": int(current_turn),
            "effective_stm": float(effective_stm),
            "alpha_ltm": float(alpha_ltm),
            "delta_t": int(delta_t),
        }

    def _apply_user_node_update(self, qa_node_id: str, payload: dict[str, Any]) -> None:
        query = """
        MATCH (n:GroupC_UserQANode {qa_node_id: $qa_node_id})
        SET n.interact_count = $interact_count,
            n.stm_score = $stm_score,
            n.ltm_score = $ltm_score,
            n.struggle_index = $struggle_index,
            n.last_interact_turn = $last_interact_turn,
            n.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(
                query,
                qa_node_id=qa_node_id,
                interact_count=int(payload["interact_count"]),
                stm_score=float(payload["stm_score"]),
                ltm_score=float(payload["ltm_score"]),
                struggle_index=float(payload["struggle_index"]),
                last_interact_turn=int(payload["last_interact_turn"]),
            )

    def _apply_backpropagation(
        self,
        route_context: dict[str, Any],
        child_effective_stm: float,
        current_turn: int,
    ) -> bool:
        if route_context.get("route") != "Drill-down":
            return False
        if child_effective_stm <= self.cfg.backprop_stm_gate:
            return False

        child_id = str(route_context.get("new_qa_node_id") or "").strip()
        if not child_id:
            return False

        parent_query = """
        MATCH (p:GroupC_UserQANode)-[:DEEPENS_INTO]->(c:GroupC_UserQANode {qa_node_id: $child_id})
        RETURN p.qa_node_id AS qa_node_id,
               coalesce(p.interact_count, 0) AS interact_count,
               coalesce(p.ltm_score, 0.0) AS ltm_score,
               coalesce(p.stm_score, 0.0) AS stm_score,
               coalesce(p.struggle_index, 0.0) AS struggle_index
        LIMIT 1
        """

        with self.driver.session() as session:
            parent = session.run(parent_query, child_id=child_id).single()
        if not parent:
            return False

        c_old = int(parent["interact_count"])
        c_new = c_old + 1
        alpha_ltm = max(2.0 / (c_new + 1.0), self.cfg.ltm_alpha_min)
        ltm_old = self._clamp01(float(parent["ltm_score"]))
        ltm_new = (1.0 - alpha_ltm) * ltm_old + alpha_ltm * self._clamp01(child_effective_stm)
        ltm_new = self._clamp01(ltm_new)

        update_query = """
        MATCH (p:GroupC_UserQANode {qa_node_id: $qa_node_id})
        SET p.interact_count = $interact_count,
            p.ltm_score = $ltm_score,
            p.last_interact_turn = $last_interact_turn,
            p.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(
                update_query,
                qa_node_id=parent["qa_node_id"],
                interact_count=c_new,
                ltm_score=float(ltm_new),
                last_interact_turn=int(current_turn),
            )
        return True

    def _apply_topic_shift_penalty(self, route_context: dict[str, Any], current_turn: int) -> bool:
        if route_context.get("route") != "Topic-Shift":
            return False

        abandoned_id = str(route_context.get("abandoned_qa_node_id") or "").strip()
        if not abandoned_id:
            return False

        fetch_query = """
        MATCH (n:GroupC_UserQANode {qa_node_id: $qa_node_id})
        RETURN coalesce(n.stm_score, 0.0) AS stm_score,
               coalesce(n.ltm_score, 0.0) AS ltm_score,
               coalesce(n.last_interact_turn, n.create_turn, 0) AS last_turn
        """
        with self.driver.session() as session:
            record = session.run(fetch_query, qa_node_id=abandoned_id).single()
        if not record:
            return False

        stm_old = self._clamp01(float(record["stm_score"]))
        ltm_old = self._clamp01(float(record["ltm_score"]))
        last_turn = int(record["last_turn"])
        delta_t = max(1, int(current_turn) - int(last_turn))
        lambda_actual = self.cfg.lambda_base * (1.0 - self.cfg.ltm_resistance_mu * ltm_old)
        penalty_dt = delta_t * self.cfg.topic_shift_penalty_factor
        stm_new = stm_old * math.exp(-lambda_actual * penalty_dt)
        stm_new = self._clamp01(stm_new)

        update_query = """
        MATCH (n:GroupC_UserQANode {qa_node_id: $qa_node_id})
        SET n.stm_score = $stm_score,
            n.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(update_query, qa_node_id=abandoned_id, stm_score=float(stm_new))
        return True

    def consolidate_after_qa(
        self,
        user_id: str,
        current_turn: int,
        qa_score: float,
        current_struggle: float,
        hit_syllabus_ids: list[str],
        route_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        route_context = route_context or {}
        summary: dict[str, Any] = {
            "updated_nodes": 0,
            "backprop_applied": False,
            "topic_shift_penalty_applied": False,
        }

        if not user_id or current_turn is None or not hit_syllabus_ids:
            summary["skipped"] = "missing_user_or_turn_or_hits"
            return summary

        qa_score = self._clamp01(float(qa_score))
        current_struggle = self._clamp01(float(current_struggle))

        state_nodes = self._fetch_user_nodes_for_consolidation(user_id, hit_syllabus_ids)
        if not state_nodes:
            summary["skipped"] = "no_state_nodes_for_hits"
            return summary

        effective_map: dict[str, float] = {}
        for syllabus_id, old_state in state_nodes.items():
            qa_node_id = str(old_state.get("qa_node_id") or "").strip()
            if not qa_node_id:
                continue
            new_state = self._calculate_memory_update(old_state, current_turn, qa_score, current_struggle)
            self._apply_user_node_update(qa_node_id, new_state)
            effective_map[qa_node_id] = float(new_state["effective_stm"])
            summary["updated_nodes"] += 1

        child_id = str(route_context.get("new_qa_node_id") or "").strip()
        child_effective = float(effective_map.get(child_id, max(effective_map.values(), default=0.0)))
        summary["backprop_applied"] = self._apply_backpropagation(route_context, child_effective, current_turn)
        summary["topic_shift_penalty_applied"] = self._apply_topic_shift_penalty(route_context, current_turn)
        return summary

    def _fetch_prereq_adjacency(self) -> dict[str, set[str]]:
        rel_types = ["GROUP_C_REQUIRES", "GROUP_C_NEEDS_REVIEW"]
        if self.cfg.enable_legacy_prereq_edge:
            rel_types.append("REQUIRES_PREREQUISITE")
        rel_pattern = "|".join(rel_types)

        query = f"""
        MATCH (a:GroupC_SyllabusNode)-[r:{rel_pattern}]->(b:GroupC_SyllabusNode)
        RETURN a.node_id AS source_id, b.node_id AS target_id
        """

        adjacency: dict[str, set[str]] = {}
        with self.driver.session() as session:
            for record in session.run(query):
                source_id = str(record["source_id"] or "").strip()
                target_id = str(record["target_id"] or "").strip()
                if not source_id or not target_id:
                    continue
                adjacency.setdefault(source_id, set()).add(target_id)
        return adjacency

    @staticmethod
    def _has_path(adjacency: dict[str, set[str]], source_id: str, target_id: str) -> bool:
        if source_id == target_id:
            return True
        stack = [source_id]
        visited = {source_id}
        while stack:
            node = stack.pop()
            for nxt in adjacency.get(node, set()):
                if nxt == target_id:
                    return True
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return False

    def _fetch_ancestor_depth_map(self, node_id: str) -> dict[str, int]:
        query = """
        MATCH (anc:GroupC_SyllabusNode)-[:HAS_SUBTOPIC*0..]->(n:GroupC_SyllabusNode {node_id: $node_id})
        RETURN anc.node_id AS ancestor_id,
               coalesce(anc.depth, 0) AS ancestor_depth
        """

        ancestors: dict[str, int] = {}
        with self.driver.session() as session:
            records = list(session.run(query, node_id=node_id))

        for record in records:
            anc_id = str(record["ancestor_id"] or "").strip()
            if not anc_id:
                continue
            depth = int(record["ancestor_depth"]) if record["ancestor_depth"] is not None else 0
            prev = ancestors.get(anc_id)
            if prev is None or depth > prev:
                ancestors[anc_id] = depth
        return ancestors

    @staticmethod
    def _find_lca_depth(a_anc: dict[str, int], b_anc: dict[str, int]) -> int:
        common = set(a_anc.keys()) & set(b_anc.keys())
        if not common:
            return -1
        return max(min(a_anc[item], b_anc[item]) for item in common)

    def _mine_hard_negative_pairs(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[bool, list[dict[str, Any]]]:
        if len(candidates) < 2:
            return False, []

        top_candidates = candidates[: self.cfg.hard_neg_top_m]
        adjacency = self._fetch_prereq_adjacency()

        parent_ids = {row["parent_syllabus_id"] for row in top_candidates}
        ancestor_cache = {pid: self._fetch_ancestor_depth_map(pid) for pid in parent_ids}

        pairs: list[dict[str, Any]] = []
        for left, right in combinations(top_candidates, 2):
            score_diff = abs(left["final_score"] - right["final_score"])
            if score_diff > self.cfg.hard_neg_tolerance:
                continue

            left_parent = left["parent_syllabus_id"]
            right_parent = right["parent_syllabus_id"]
            if left_parent == right_parent:
                continue

            has_dependency_path = self._has_path(adjacency, left_parent, right_parent) or self._has_path(
                adjacency,
                right_parent,
                left_parent,
            )
            lca_depth = self._find_lca_depth(ancestor_cache.get(left_parent, {}), ancestor_cache.get(right_parent, {}))
            lca_condition = lca_depth >= 0 and lca_depth < self.cfg.hard_neg_lca_depth_threshold

            if has_dependency_path and not lca_condition:
                continue

            topology_flags: list[str] = []
            if not has_dependency_path:
                topology_flags.append("no_dependency_path")
            if lca_condition:
                topology_flags.append("low_lca_depth")

            pairs.append(
                {
                    "candidate_i_snippet_id": left["snippet_id"],
                    "candidate_j_snippet_id": right["snippet_id"],
                    "candidate_i_parent_syllabus_id": left_parent,
                    "candidate_j_parent_syllabus_id": right_parent,
                    "candidate_i_final_score": left["final_score"],
                    "candidate_j_final_score": right["final_score"],
                    "score_diff": score_diff,
                    "lca_depth": lca_depth,
                    "topology_flags": topology_flags,
                }
            )

        return len(pairs) > 0, pairs

    def _fetch_pedagogical_context(self, selected_parent_ids: list[str]) -> list[str]:
        if not selected_parent_ids:
            return []

        rel_types = ["GROUP_C_REQUIRES", "GROUP_C_NEEDS_REVIEW"]
        if self.cfg.enable_legacy_prereq_edge:
            rel_types.append("REQUIRES_PREREQUISITE")
        rel_pattern = "|".join(rel_types)

        query = f"""
        MATCH (prereq:GroupC_SyllabusNode)-[r:{rel_pattern}]->(s:GroupC_SyllabusNode)
        WHERE s.node_id IN $selected_ids
        RETURN prereq.node_id AS prereq_node_id,
               prereq.name AS prereq_name,
               s.node_id AS target_node_id,
               s.name AS target_name,
               type(r) AS rel_type,
               coalesce(r.llm_score, r.confidence, 0.0) AS rel_score
        ORDER BY rel_score DESC, prereq_name ASC
        """

        with self.driver.session() as session:
            records = list(session.run(query, selected_ids=selected_parent_ids))

        contexts: list[str] = []
        seen: set[str] = set()
        for record in records:
            prereq_name = str(record["prereq_name"] or "").strip()
            target_name = str(record["target_name"] or "").strip()
            rel_type = str(record["rel_type"] or "").strip()
            rel_score = float(record["rel_score"])
            if not prereq_name or not target_name or not rel_type:
                continue

            text = f"[{rel_type}] {prereq_name} -> {target_name} (score={rel_score:.4f})"
            if text in seen:
                continue
            seen.add(text)
            contexts.append(text)

        return contexts

    def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        current_turn: int | None = None,
        qa_score: float | None = None,
        current_struggle: float | None = None,
    ) -> RetrievalResult:
        t0 = perf_counter()
        debug_metrics: dict[str, Any] = {}

        rewrite_start = perf_counter()
        global_keywords, local_keywords = self._expand_query_keywords(query)
        macro_query = self._compose_query(query, global_keywords)
        micro_query = self._compose_query(query, local_keywords)
        debug_metrics["rewrite_ms"] = round((perf_counter() - rewrite_start) * 1000, 2)

        macro_rows: list[dict[str, Any]] = []
        macro_start = perf_counter()
        try:
            macro_rows = self._retrieve_macro_track(macro_query)
        except Exception as exc:
            logger.warning("Macro retrieval failed, continue with micro only: %s", exc)
            debug_metrics["macro_error"] = str(exc)
        debug_metrics["macro_retrieval_ms"] = round((perf_counter() - macro_start) * 1000, 2)
        debug_metrics["macro_hit_count"] = len(macro_rows)

        micro_rows: list[dict[str, Any]] = []
        micro_start = perf_counter()
        try:
            micro_rows = self._retrieve_micro_track(micro_query)
        except Exception as exc:
            logger.error("Micro retrieval failed: %s", exc)
            debug_metrics["micro_error"] = str(exc)
            return RetrievalResult(
                original_query=query,
                expanded_global_keywords=global_keywords,
                expanded_local_keywords=local_keywords,
                candidates=[],
                is_contrastive_triggered=False,
                contrastive_pairs=[],
                pedagogical_context=[f"[ERROR] micro retrieval failed: {exc}"],
                debug_metrics=debug_metrics,
            )
        debug_metrics["micro_retrieval_ms"] = round((perf_counter() - micro_start) * 1000, 2)
        debug_metrics["micro_hit_count"] = len(micro_rows)

        user_state_by_syllabus: dict[str, dict[str, Any]] = {}
        if user_id:
            syllabus_ids = [
                str(row.get("parent_syllabus_id", "")).strip()
                for row in micro_rows
                if str(row.get("parent_syllabus_id", "")).strip()
            ]
            user_state_by_syllabus = self._fetch_user_state_by_syllabus(user_id, syllabus_ids)
        debug_metrics["state_hit_syllabus_count"] = len(user_state_by_syllabus)

        fuse_start = perf_counter()
        candidates = self._fuse_candidates(
            macro_rows=macro_rows,
            micro_rows=micro_rows,
            user_state_by_syllabus=user_state_by_syllabus,
        )
        debug_metrics["fuse_ms"] = round((perf_counter() - fuse_start) * 1000, 2)
        debug_metrics["final_candidate_count"] = len(candidates)

        selected_parent_ids: list[str] = []
        _seen_parent_ids: set[str] = set()
        for row in candidates:
            pid = str(row.get("parent_syllabus_id", "")).strip()
            if not pid or pid in _seen_parent_ids:
                continue
            _seen_parent_ids.add(pid)
            selected_parent_ids.append(pid)

        route_context: dict[str, Any] = {}
        route_start = perf_counter()
        if user_id and current_turn is not None and selected_parent_ids:
            try:
                route_context = self._route_and_mutate_user_graph(
                    user_id=user_id,
                    current_turn=current_turn,
                    query=query,
                    target_syllabus_ids=selected_parent_ids,
                )
            except Exception as exc:
                debug_metrics["state_route_error"] = str(exc)
                logger.warning("State routing failed, continue retrieval flow: %s", exc)
        debug_metrics["state_route_ms"] = round((perf_counter() - route_start) * 1000, 2)

        hard_neg_start = perf_counter()
        try:
            is_triggered, contrastive_pairs = self._mine_hard_negative_pairs(candidates)
        except Exception as exc:
            logger.warning("Hard-negative mining failed and will be skipped: %s", exc)
            is_triggered, contrastive_pairs = False, []
            debug_metrics["hard_negative_error"] = str(exc)
        debug_metrics["hard_negative_ms"] = round((perf_counter() - hard_neg_start) * 1000, 2)
        debug_metrics["hard_negative_pair_count"] = len(contrastive_pairs)

        context_start = perf_counter()
        pedagogical_context: list[str] = []
        try:
            pedagogical_context = self._fetch_pedagogical_context(selected_parent_ids)
        except Exception as exc:
            logger.warning("Pedagogical 1-hop fetch failed and will be skipped: %s", exc)
            debug_metrics["pedagogical_context_error"] = str(exc)
        debug_metrics["pedagogical_context_ms"] = round((perf_counter() - context_start) * 1000, 2)
        debug_metrics["pedagogical_context_count"] = len(pedagogical_context)

        if user_id and current_turn is not None and qa_score is not None and current_struggle is not None:
            consolidate_start = perf_counter()
            try:
                consolidation_summary = self.consolidate_after_qa(
                    user_id=user_id,
                    current_turn=current_turn,
                    qa_score=qa_score,
                    current_struggle=current_struggle,
                    hit_syllabus_ids=selected_parent_ids,
                    route_context=route_context,
                )
                debug_metrics["state_consolidation"] = consolidation_summary
            except Exception as exc:
                debug_metrics["state_consolidation_error"] = str(exc)
                logger.warning("State consolidation failed, continue retrieval output: %s", exc)
            debug_metrics["state_consolidation_ms"] = round((perf_counter() - consolidate_start) * 1000, 2)

        debug_metrics["total_ms"] = round((perf_counter() - t0) * 1000, 2)

        logger.info(
            "Group C dual retrieval done | rewrite=%sms macro=%sms micro=%sms fuse=%sms hard_neg=%sms context=%sms final=%s",
            debug_metrics.get("rewrite_ms", 0),
            debug_metrics.get("macro_retrieval_ms", 0),
            debug_metrics.get("micro_retrieval_ms", 0),
            debug_metrics.get("fuse_ms", 0),
            debug_metrics.get("hard_negative_ms", 0),
            debug_metrics.get("pedagogical_context_ms", 0),
            len(candidates),
        )

        return RetrievalResult(
            original_query=query,
            expanded_global_keywords=global_keywords,
            expanded_local_keywords=local_keywords,
            candidates=candidates,
            is_contrastive_triggered=is_triggered,
            contrastive_pairs=contrastive_pairs,
            pedagogical_context=pedagogical_context,
            state_update_context=route_context,
            debug_metrics=debug_metrics,
        )


if __name__ == "__main__":
    default_input_json = os.path.normpath(
        os.path.join(PROJECT_DIR, "..", "QAdata", "group_c_input_template.json")
    )
    input_json_path = os.getenv("GROUP_C_INPUT_JSON_PATH", default_input_json).strip()
    only_qid = os.getenv("GROUP_C_ONLY_QID", "").strip()
    output_json_path = os.getenv(
        "GROUP_C_OUTPUT_JSON_PATH",
        os.path.join(CURRENT_DIR, "group_c_retrieval_results.json"),
    ).strip()

    fallback_query = os.getenv("GROUP_C_QUERY", "Python 列表和元组的区别是什么？")
    fallback_user_id = os.getenv("GROUP_C_USER_ID", "").strip() or None
    fallback_turn = GroupCDualRetriever._parse_optional_int(os.getenv("GROUP_C_TURN"))
    fallback_qa_score = GroupCDualRetriever._parse_optional_float(os.getenv("GROUP_C_QA_SCORE"))
    fallback_current_struggle = GroupCDualRetriever._parse_optional_float(os.getenv("GROUP_C_CURRENT_STRUGGLE"))

    retriever = GroupCDualRetriever()
    try:
        if os.path.exists(input_json_path):
            samples = _load_group_c_input_samples(input_json_path)
            if only_qid:
                samples = [item for item in samples if item.get("id") == only_qid]

            if not samples:
                logger.warning(
                    "No valid sample found in %s (GROUP_C_ONLY_QID=%s).",
                    input_json_path,
                    only_qid,
                )
                print("[]")
            else:
                logger.info(
                    "Running Group C retrieval from input file: %s (samples=%s)",
                    input_json_path,
                    len(samples),
                )
                batch_results: list[dict[str, Any]] = []
                for item in samples:
                    q_id = str(item.get("id", "")).strip()
                    question = str(item.get("question", "")).strip()
                    if not question:
                        logger.warning("Skip %s because question is empty. Please fill it in input json.", q_id)
                        batch_results.append(
                            {
                                "id": q_id,
                                "type": item.get("type", ""),
                                "skipped": "empty_question",
                                "input": item,
                            }
                        )
                        continue

                    result = retriever.retrieve(
                        query=question,
                        user_id=item.get("user_id"),
                        current_turn=item.get("current_turn"),
                        qa_score=item.get("qa_score"),
                        current_struggle=item.get("current_struggle"),
                    )
                    result_dict = result.to_dict()

                    golden_ids = item.get("golden_parent_syllabus_ids") or []
                    metrics = evaluate_retrieval_metrics(
                        result_dict.get("candidates", []),
                        golden_parent_syllabus_ids=golden_ids,
                        k=retriever.cfg.top_n_final,
                    )

                    batch_results.append(
                        {
                            "id": q_id,
                            "type": item.get("type", ""),
                            "input": item,
                            "retrieval_result": result_dict,
                            "retrieval_metrics": metrics,
                        }
                    )

                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(batch_results, f, ensure_ascii=False, indent=2)
                logger.info("Group C retrieval output saved to: %s", output_json_path)
                print(json.dumps(batch_results, ensure_ascii=False, indent=2))
        else:
            logger.warning(
                "Input json not found: %s. Fallback to single-query mode (GROUP_C_QUERY).",
                input_json_path,
            )
            result = retriever.retrieve(
                query=fallback_query,
                user_id=fallback_user_id,
                current_turn=fallback_turn,
                qa_score=fallback_qa_score,
                current_struggle=fallback_current_struggle,
            )
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally:
        retriever.close()
