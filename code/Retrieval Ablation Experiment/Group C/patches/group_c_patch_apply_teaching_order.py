from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

PATCH_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_C_DIR = os.path.dirname(PATCH_DIR)
PROJECT_DIR = os.path.dirname(GROUP_C_DIR)
GROUP_B_DIR = os.path.join(PROJECT_DIR, "Group B")

for path in (GROUP_C_DIR, PROJECT_DIR, GROUP_B_DIR):
    if path not in sys.path:
        sys.path.append(path)

from neo4j_ops import GroupCStaticRepository
from shared_retrieval_utils import load_runtime_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupC_Teaching_Order_Patch")

WEEK_RE = re.compile(r"第\s*(\d+)\s*周")
CHAPTER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


@dataclass
class PatcherConfig:
    dry_run: bool = False


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config_from_env() -> PatcherConfig:
    return PatcherConfig(
        dry_run=_parse_bool(os.getenv("GROUP_C_TEACHING_ORDER_DRY_RUN"), default=False),
    )


def _extract_week_num(node: dict[str, Any]) -> int:
    week_tag = str(node.get("week_tag") or "")
    abs_path = str(node.get("abs_path") or "")

    match = WEEK_RE.search(week_tag) or WEEK_RE.search(abs_path)
    if match:
        return int(match.group(1))
    return 10**9


def _extract_chapter_tuple(name: str) -> tuple[int, ...]:
    match = CHAPTER_RE.match(name or "")
    if not match:
        return (10**9,)
    return tuple(int(part) for part in match.group(1).split("."))


def _node_sort_key(node: dict[str, Any]) -> tuple[Any, ...]:
    name = str(node.get("name") or "")
    depth = int(node.get("depth") or -1)
    abs_path = str(node.get("abs_path") or "")

    return (
        _extract_week_num(node),
        depth,
        _extract_chapter_tuple(name),
        name,
        abs_path,
    )


def _fetch_graph(repo: GroupCStaticRepository) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    node_query = """
    MATCH (s:GroupC_SyllabusNode)
    RETURN s.node_id AS node_id,
           s.name AS name,
           s.depth AS depth,
           s.week_tag AS week_tag,
           s.abs_path AS abs_path
    """

    with repo.driver.session() as session:
        node_records = list(session.run(node_query))

    nodes: dict[str, dict[str, Any]] = {}
    for record in node_records:
        row = {
            "node_id": record["node_id"],
            "name": record["name"] or "",
            "depth": int(record["depth"]) if record["depth"] is not None else -1,
            "week_tag": record["week_tag"] or "",
            "abs_path": record["abs_path"] or "",
        }
        nodes[row["node_id"]] = row

    if not nodes:
        return {}, {}, {}

    edge_query = """
    MATCH (p:GroupC_SyllabusNode)-[:HAS_SUBTOPIC]->(c:GroupC_SyllabusNode)
    WHERE p.node_id IN $node_ids AND c.node_id IN $node_ids
    RETURN p.node_id AS parent_id, c.node_id AS child_id
    """

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes.keys()}
    indegree: dict[str, int] = {node_id: 0 for node_id in nodes.keys()}

    with repo.driver.session() as session:
        edge_records = list(session.run(edge_query, node_ids=list(nodes.keys())))

    for record in edge_records:
        parent_id = record["parent_id"]
        child_id = record["child_id"]
        adjacency[parent_id].append(child_id)
        indegree[child_id] += 1

    return nodes, adjacency, indegree


def _build_dfs_teaching_order(
    nodes: dict[str, dict[str, Any]],
    adjacency: dict[str, list[str]],
    indegree: dict[str, int],
) -> list[str]:
    roots = [node_id for node_id in nodes.keys() if indegree.get(node_id, 0) == 0]
    roots.sort(key=lambda node_id: _node_sort_key(nodes[node_id]))

    visited: set[str] = set()
    ordered_node_ids: list[str] = []

    def dfs(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        ordered_node_ids.append(node_id)

        children = sorted(
            adjacency.get(node_id, []),
            key=lambda child_id: _node_sort_key(nodes[child_id]),
        )
        for child_id in children:
            dfs(child_id)

    for root_id in roots:
        dfs(root_id)

    if len(visited) != len(nodes):
        leftovers = [node_id for node_id in nodes.keys() if node_id not in visited]
        leftovers.sort(key=lambda node_id: _node_sort_key(nodes[node_id]))
        for node_id in leftovers:
            dfs(node_id)

    return ordered_node_ids


def _apply_teaching_order(repo: GroupCStaticRepository, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    query = """
    UNWIND $rows AS row
    MATCH (s:GroupC_SyllabusNode {node_id: row.node_id})
    SET s.teaching_order = row.teaching_order
    RETURN count(s) AS count
    """

    with repo.driver.session() as session:
        record = session.run(query, rows=rows).single()
        return int(record["count"]) if record else 0


def run_patch() -> None:
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
        nodes, adjacency, indegree = _fetch_graph(repo)
        if not nodes:
            logger.warning("No GroupC_SyllabusNode found for teaching order patch.")
            return

        ordered_node_ids = _build_dfs_teaching_order(nodes, adjacency, indegree)
        write_rows = [
            {"node_id": node_id, "teaching_order": order}
            for order, node_id in enumerate(ordered_node_ids, start=1)
        ]

        logger.info(
            "Teaching order prepared: nodes=%d, roots=%d, scope=%s",
            len(write_rows),
            sum(1 for node_id in nodes.keys() if indegree.get(node_id, 0) == 0),
            "ALL_WEEKS",
        )

        preview_count = min(10, len(ordered_node_ids))
        for i in range(preview_count):
            node_id = ordered_node_ids[i]
            logger.info(
                "Preview order=%d, name=%s, depth=%s",
                i + 1,
                nodes[node_id].get("name", ""),
                nodes[node_id].get("depth", -1),
            )

        if cfg.dry_run:
            logger.info("GROUP_C_TEACHING_ORDER_DRY_RUN=true, skip writing teaching_order.")
            return

        updated = _apply_teaching_order(repo, write_rows)
        logger.info("Teaching order patch done: updated_nodes=%d", updated)
    finally:
        repo.close()


if __name__ == "__main__":
    run_patch()
