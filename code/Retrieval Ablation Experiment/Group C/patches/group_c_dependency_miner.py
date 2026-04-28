from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATCH_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_C_DIR = os.path.dirname(PATCH_DIR)
PROJECT_DIR = os.path.dirname(GROUP_C_DIR)
GROUP_B_DIR = os.path.join(PROJECT_DIR, "Group B")

for path in (GROUP_C_DIR, PROJECT_DIR, GROUP_B_DIR):
    if path not in sys.path:
        sys.path.append(path)

from neo4j_ops import GroupCStaticRepository
from shared_retrieval_utils import (  # noqa: E402
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    ark_chat_completion,
    get_adaptive_generation_params,
    load_runtime_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupC_Dependency_Miner")


@dataclass
class DependencyMinerConfig:
    lookback_window_size: int = 50
    global_anchor_layer: int = 2
    llm_score_threshold: int = 6
    cognitive_decay_threshold: int = 33
    llm_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    llm_api_base: str = DEFAULT_ARK_API_BASE
    llm_min_response_tokens: int = 1200
    llm_max_response_tokens_cap: int = 4000
    max_context_chars: int = 8000
    output_json_path: str = os.path.join(PATCH_DIR, "dry_run_dependencies.json")
    dry_run: bool = False


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config_from_env() -> DependencyMinerConfig:
    return DependencyMinerConfig(
        lookback_window_size=max(1, int(os.getenv("LOOKBACK_WINDOW_SIZE", "50"))),
        global_anchor_layer=max(0, int(os.getenv("GLOBAL_ANCHOR_LAYER", "2"))),
        llm_score_threshold=max(0, min(10, int(os.getenv("LLM_SCORE_THRESHOLD", "6")))),
        cognitive_decay_threshold=max(1, int(os.getenv("COGNITIVE_DECAY_THRESHOLD", "33"))),
        llm_model=os.getenv("GROUP_C_DEP_MODEL", DEFAULT_DEEPSEEK_ENDPOINT),
        llm_api_base=os.getenv("ARK_API_BASE", DEFAULT_ARK_API_BASE),
        llm_min_response_tokens=max(128, int(os.getenv("GROUP_C_DEP_MIN_RESPONSE_TOKENS", "1200"))),
        llm_max_response_tokens_cap=max(256, int(os.getenv("GROUP_C_DEP_RESPONSE_TOKENS_CAP", "4000"))),
        max_context_chars=max(1000, int(os.getenv("GROUP_C_DEP_MAX_CONTEXT_CHARS", "8000"))),
        output_json_path=os.getenv(
            "GROUP_C_DEP_OUTPUT_JSON",
            os.path.join(PATCH_DIR, "dry_run_dependencies.json"),
        ),
        dry_run=_parse_bool(os.getenv("GROUP_C_DEP_DRY_RUN"), default=False),
    )


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


def _fetch_ordered_nodes(repo: GroupCStaticRepository) -> list[dict[str, Any]]:
    query = """
    MATCH (s:GroupC_SyllabusNode)
    WHERE s.teaching_order IS NOT NULL
    RETURN s.node_id AS node_id,
           s.name AS name,
           s.depth AS depth,
           s.week_tag AS week_tag,
           s.abs_path AS abs_path,
           s.teaching_order AS teaching_order
    ORDER BY s.teaching_order ASC, s.depth ASC, s.name ASC
    """

    with repo.driver.session() as session:
        records = list(session.run(query))

    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "node_id": record["node_id"],
            "name": record["name"] or "",
            "depth": int(record["depth"]) if record["depth"] is not None else -1,
            "week_tag": record["week_tag"] or "",
            "abs_path": record["abs_path"] or "",
            "teaching_order": int(record["teaching_order"]),
        }
        rows.append(row)

    return rows


def _fetch_aggregated_contexts(
    repo: GroupCStaticRepository,
    nodes: list[dict[str, Any]],
    max_context_chars: int,
) -> dict[str, str]:
    node_ids = [node["node_id"] for node in nodes]
    if not node_ids:
        return {}

    text_query = """
    MATCH (s:GroupC_SyllabusNode)-[:HAS_TEXT]->(t:GroupC_TextSnippet)
    WHERE s.node_id IN $node_ids
      AND t.text IS NOT NULL
      AND trim(t.text) <> ''
    RETURN s.node_id AS node_id,
           t.text AS text,
           coalesce(t.summary_level, -1) AS summary_level,
           coalesce(t.chunk_order, 0) AS chunk_order
    ORDER BY node_id ASC, summary_level DESC, chunk_order ASC
    """

    code_query = """
    MATCH (s:GroupC_SyllabusNode)-[:HAS_CODE]->(c:GroupC_CodeSnippet)
    WHERE s.node_id IN $node_ids
      AND c.code IS NOT NULL
      AND trim(c.code) <> ''
    RETURN s.node_id AS node_id,
           c.code AS code,
           coalesce(c.chunk_order, 0) AS chunk_order
    ORDER BY node_id ASC, chunk_order ASC
    """

    text_map: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    code_map: dict[str, list[str]] = {node_id: [] for node_id in node_ids}

    with repo.driver.session() as session:
        text_records = list(session.run(text_query, node_ids=node_ids))
        code_records = list(session.run(code_query, node_ids=node_ids))

    for record in text_records:
        node_id = record["node_id"]
        text = (record["text"] or "").strip()
        if text:
            text_map[node_id].append(text)

    for record in code_records:
        node_id = record["node_id"]
        code = (record["code"] or "").strip()
        if code:
            code_map[node_id].append(code)

    contexts: dict[str, str] = {}
    for node in nodes:
        node_id = node["node_id"]
        parts: list[str] = [
            f"[Node] {node['name']}",
            f"[TeachingOrder] {node['teaching_order']}",
            f"[Depth] {node['depth']}",
        ]

        if text_map[node_id]:
            parts.append("[TextSnippets]")
            parts.append("\n\n".join(text_map[node_id]))

        if code_map[node_id]:
            parts.append("[CodeSnippets]")
            parts.append("\n\n".join(code_map[node_id]))

        context = "\n".join(parts).strip()
        if len(context) > max_context_chars:
            context = context[:max_context_chars]
        contexts[node_id] = context

    return contexts


def _build_candidate_list(
    ordered_nodes: list[dict[str, Any]],
    current_idx: int,
    lookback_window_size: int,
    global_anchor_layer: int,
) -> list[dict[str, Any]]:
    short_start = max(0, current_idx - lookback_window_size)
    short_term_nodes = ordered_nodes[short_start:current_idx]

    far_end = max(0, current_idx - lookback_window_size)
    far_nodes = ordered_nodes[:far_end]
    anchors = [node for node in far_nodes if node["depth"] <= global_anchor_layer]

    merged = anchors + short_term_nodes
    dedup: dict[str, dict[str, Any]] = {}
    for node in merged:
        dedup[node["node_id"]] = {
            "candidate_id": node["node_id"],
            "chunk_title": node["name"],
            "depth": node["depth"],
            "teaching_order": node["teaching_order"],
        }

    return list(dedup.values())


def _score_candidates_with_llm(
    cfg: DependencyMinerConfig,
    runtime_api_key: str | None,
    current_node: dict[str, Any],
    current_context: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_prompt = f"""
你是课程知识图谱中的前置依赖挖掘助手。
请基于“当前教学节点”和“候选前置节点列表”，对每个候选节点打分（0-10 分）：
- 分数越高，表示该候选节点越是理解当前节点所必需的前置知识。

[当前节点]
名称: {current_node['name']}
教学顺序: {current_node['teaching_order']}

[当前节点上下文]
{current_context}

[候选节点列表 Candidate_List(JSON)]
{json.dumps(candidates, ensure_ascii=False)}
"""

    try:
        temperature, adaptive_max_tokens = get_adaptive_generation_params(base_prompt, task="evaluate")
        response_token_budget = max(
            adaptive_max_tokens,
            cfg.llm_min_response_tokens,
            220 + 32 * len(candidates),
        )
        response_token_budget = min(response_token_budget, cfg.llm_max_response_tokens_cap)

        prompt = f"""
{base_prompt}

[输出约束]
1. 本次调用的最大输出 token 上限为: {response_token_budget}
2. 你必须严格输出 JSON，不要输出任何解释性文本。
3. `dependencies` 中每项只包含: `candidate_id`, `score`, `reason`。
4. `score` 必须是 0-10 的数字（可以是整数或小数）。
5. `reason` 必须是简短中文原因。

请按以下 JSON 结构返回：
{{
    "dependencies": [
        {{
            "candidate_id": "...",
            "score": 0,
            "reason": "简短中文原因"
        }}
    ]
}}
"""

        response = ark_chat_completion(
            model=cfg.llm_model,
            prompt=prompt,
            api_base=cfg.llm_api_base,
            api_key=runtime_api_key,
            temperature=temperature,
            max_tokens=response_token_budget,
            timeout_sec=180,
        )
        parsed = _safe_json_parse(response)
        dependencies = parsed.get("dependencies", [])
        if not isinstance(dependencies, list):
            logger.warning(
                "LLM scoring parse warning for node_id=%s: empty or invalid dependencies payload.",
                current_node["node_id"],
            )
            return []
        return dependencies
    except Exception as exc:
        logger.warning(
            "LLM scoring failed for node_id=%s due to: %s",
            current_node["node_id"],
            exc,
        )
        return []


def _mine_raw_edges(
    cfg: DependencyMinerConfig,
    runtime_api_key: str | None,
    ordered_nodes: list[dict[str, Any]],
    contexts: dict[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    raw_edges: dict[tuple[str, str], dict[str, Any]] = {}

    total = len(ordered_nodes)
    for idx in range(1, total):
        current = ordered_nodes[idx]
        current_context = contexts.get(current["node_id"], "")
        candidates = _build_candidate_list(
            ordered_nodes=ordered_nodes,
            current_idx=idx,
            lookback_window_size=cfg.lookback_window_size,
            global_anchor_layer=cfg.global_anchor_layer,
        )
        if not candidates:
            continue

        logger.info("Processing node %d/%d, candidates=%d", idx + 1, total, len(candidates))
        scored = _score_candidates_with_llm(
            cfg=cfg,
            runtime_api_key=runtime_api_key,
            current_node=current,
            current_context=current_context,
            candidates=candidates,
        )

        for item in scored:
            candidate_id = str(item.get("candidate_id", "")).strip()
            if not candidate_id:
                continue

            score_raw = item.get("score", 0)
            try:
                score = float(score_raw)
            except Exception:
                continue

            if score < cfg.llm_score_threshold:
                continue

            source_id = candidate_id
            target_id = current["node_id"]
            if source_id == target_id:
                continue

            reason = str(item.get("reason", "")).strip()
            edge_key = (source_id, target_id)
            prev = raw_edges.get(edge_key)
            if prev is None or score > prev["llm_score"]:
                raw_edges[edge_key] = {
                    "source_id": source_id,
                    "target_id": target_id,
                    "llm_score": score,
                    "reason": reason,
                }

    return raw_edges


def _has_transitive_path(adjacency: dict[str, set[str]], source: str, target: str) -> bool:
    stack = [source]
    visited = {source}

    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, set()):
            if node == source and nxt == target:
                continue
            if nxt == target:
                return True
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return False


def _prune_and_mutate_edges(
    raw_edges: dict[tuple[str, str], dict[str, Any]],
    order_map: dict[str, int],
    cognitive_decay_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Keep only forward edges for DAG pruning, based on teaching_order.
    forward_edges: dict[tuple[str, str], dict[str, Any]] = {}
    adjacency: dict[str, set[str]] = {}
    for (source_id, target_id), payload in raw_edges.items():
        if source_id not in order_map or target_id not in order_map:
            continue
        if order_map[source_id] >= order_map[target_id]:
            continue
        forward_edges[(source_id, target_id)] = payload
        adjacency.setdefault(source_id, set()).add(target_id)

    stats = {
        "raw_edges": len(raw_edges),
        "forward_edges": len(forward_edges),
        "dropped_short_transitive": 0,
        "mutated_to_needs_review": 0,
        "kept_requires": 0,
    }

    final_edges: list[dict[str, Any]] = []
    for (source_id, target_id), payload in forward_edges.items():

        transitive = _has_transitive_path(adjacency, source_id, target_id)
        if transitive:
            span = order_map[target_id] - order_map[source_id]
            if span <= cognitive_decay_threshold:
                stats["dropped_short_transitive"] += 1
                continue
            edge_type = "NEEDS_REVIEW"
            stats["mutated_to_needs_review"] += 1
        else:
            edge_type = "REQUIRES"
            stats["kept_requires"] += 1

        final_edges.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "llm_score": payload["llm_score"],
                "reason": payload["reason"],
            }
        )

    final_edges.sort(key=lambda x: (x["source_id"], x["target_id"], x["edge_type"]))
    return final_edges, stats


def _clear_existing_edges(repo: GroupCStaticRepository) -> int:
    query = """
    MATCH (:GroupC_SyllabusNode)-[r:GROUP_C_REQUIRES|GROUP_C_NEEDS_REVIEW]->(:GroupC_SyllabusNode)
    DELETE r
    RETURN count(r) AS count
    """
    with repo.driver.session() as session:
        record = session.run(query).single()
        return int(record["count"]) if record else 0


def _write_edges(repo: GroupCStaticRepository, edges: list[dict[str, Any]]) -> tuple[int, int]:
    requires_rows = [row for row in edges if row["edge_type"] == "REQUIRES"]
    review_rows = [row for row in edges if row["edge_type"] == "NEEDS_REVIEW"]

    requires_query = """
    UNWIND $rows AS row
    MATCH (a:GroupC_SyllabusNode {node_id: row.source_id})
    MATCH (b:GroupC_SyllabusNode {node_id: row.target_id})
    MERGE (a)-[r:GROUP_C_REQUIRES]->(b)
    ON CREATE SET r.created_at = datetime()
    SET r.llm_score = row.llm_score,
        r.reason = row.reason,
        r.updated_at = datetime()
    RETURN count(r) AS count
    """

    review_query = """
    UNWIND $rows AS row
    MATCH (a:GroupC_SyllabusNode {node_id: row.source_id})
    MATCH (b:GroupC_SyllabusNode {node_id: row.target_id})
    MERGE (a)-[r:GROUP_C_NEEDS_REVIEW]->(b)
    ON CREATE SET r.created_at = datetime()
    SET r.llm_score = row.llm_score,
        r.reason = row.reason,
        r.updated_at = datetime()
    RETURN count(r) AS count
    """

    requires_count = 0
    review_count = 0
    with repo.driver.session() as session:
        if requires_rows:
            record = session.run(requires_query, rows=requires_rows).single()
            requires_count = int(record["count"]) if record else 0
        if review_rows:
            record = session.run(review_query, rows=review_rows).single()
            review_count = int(record["count"]) if record else 0

    return requires_count, review_count


def _write_audit_json(
    output_path: str,
    cfg: DependencyMinerConfig,
    ordered_nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    cleared_count: int,
    requires_count: int,
    review_count: int,
    prune_stats: dict[str, int],
) -> None:
    payload = {
        "metadata": {
            "dry_run": cfg.dry_run,
            "lookback_window_size": cfg.lookback_window_size,
            "global_anchor_layer": cfg.global_anchor_layer,
            "llm_score_threshold": cfg.llm_score_threshold,
            "cognitive_decay_threshold": cfg.cognitive_decay_threshold,
            "llm_model": cfg.llm_model,
            "node_count": len(ordered_nodes),
            "cleared_existing_edges": cleared_count,
            "written_requires": requires_count,
            "written_needs_review": review_count,
            "prune_stats": prune_stats,
        },
        "cypher_templates": {
            "clear_old_edges": "MATCH (:GroupC_SyllabusNode)-[r:GROUP_C_REQUIRES|GROUP_C_NEEDS_REVIEW]->(:GroupC_SyllabusNode) DELETE r",
            "write_requires": "MERGE (a)-[:GROUP_C_REQUIRES]->(b)",
            "write_needs_review": "MERGE (a)-[:GROUP_C_NEEDS_REVIEW]->(b)",
        },
        "edges": edges,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dependency_miner() -> None:
    cfg = build_config_from_env()

    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="YOUR VALUE",
        default_llm_api_base=cfg.llm_api_base,
    )

    repo = GroupCStaticRepository(
        uri=runtime_cfg.neo4j_uri,
        user=runtime_cfg.neo4j_user,
        password=runtime_cfg.neo4j_password,
    )

    try:
        ordered_nodes = _fetch_ordered_nodes(repo)
        if not ordered_nodes:
            logger.warning("No ordered GroupC_SyllabusNode found. Ensure teaching_order has been written.")
            _write_audit_json(
                output_path=cfg.output_json_path,
                cfg=cfg,
                ordered_nodes=[],
                edges=[],
                cleared_count=0,
                requires_count=0,
                review_count=0,
                prune_stats={
                    "raw_edges": 0,
                    "forward_edges": 0,
                    "dropped_short_transitive": 0,
                    "mutated_to_needs_review": 0,
                    "kept_requires": 0,
                },
            )
            return

        logger.info("Loaded ordered syllabus nodes: %d", len(ordered_nodes))
        contexts = _fetch_aggregated_contexts(repo, ordered_nodes, cfg.max_context_chars)
        logger.info("Aggregated node contexts ready: %d", len(contexts))

        raw_edges = _mine_raw_edges(
            cfg=cfg,
            runtime_api_key=runtime_cfg.ark_api_key,
            ordered_nodes=ordered_nodes,
            contexts=contexts,
        )
        logger.info("Raw dependency edges mined: %d", len(raw_edges))

        order_map = {node["node_id"]: node["teaching_order"] for node in ordered_nodes}
        final_edges, prune_stats = _prune_and_mutate_edges(
            raw_edges=raw_edges,
            order_map=order_map,
            cognitive_decay_threshold=cfg.cognitive_decay_threshold,
        )
        logger.info(
            "Final edges after pruning: total=%d, requires=%d, needs_review=%d, dropped_short_transitive=%d",
            len(final_edges),
            prune_stats.get("kept_requires", 0),
            prune_stats.get("mutated_to_needs_review", 0),
            prune_stats.get("dropped_short_transitive", 0),
        )

        cleared_count = 0
        requires_count = 0
        review_count = 0
        if cfg.dry_run:
            logger.info("Dry-run mode enabled. Skip database writes.")
        else:
            cleared_count = _clear_existing_edges(repo)
            logger.info("Cleared old dependency edges: %d", cleared_count)
            requires_count, review_count = _write_edges(repo, final_edges)
            logger.info(
                "Dependency edges written: requires=%d, needs_review=%d",
                requires_count,
                review_count,
            )

        _write_audit_json(
            output_path=cfg.output_json_path,
            cfg=cfg,
            ordered_nodes=ordered_nodes,
            edges=final_edges,
            cleared_count=cleared_count,
            requires_count=requires_count,
            review_count=review_count,
            prune_stats=prune_stats,
        )
        logger.info("Audit JSON exported: %s", cfg.output_json_path)

        preview = final_edges[:10]
        logger.info("Edge preview (up to 10): %s", json.dumps(preview, ensure_ascii=False))
    finally:
        repo.close()


if __name__ == "__main__":
    run_dependency_miner()
