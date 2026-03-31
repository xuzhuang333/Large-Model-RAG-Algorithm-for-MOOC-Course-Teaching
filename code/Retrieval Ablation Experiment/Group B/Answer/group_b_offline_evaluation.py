from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_B_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(GROUP_B_DIR)
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)
if GROUP_B_DIR not in sys.path:
    sys.path.append(GROUP_B_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from llm_evaluator import RAGEvaluator
from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DOUBAO_ENDPOINT,
    load_runtime_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupB_Offline_Evaluation")


@dataclass
class GroupBOfflineEvalConfig:
    annotate_dataset_path: str
    answers_input_path: str
    output_path: str
    evaluator_model: str = DEFAULT_DOUBAO_ENDPOINT
    llm_api_base: str = DEFAULT_ARK_API_BASE
    only_qid: str = ""


def build_config_from_env() -> GroupBOfflineEvalConfig:
    return GroupBOfflineEvalConfig(
        annotate_dataset_path=os.getenv(
            "QA_ANNOTATE_DATASET_PATH",
            os.path.join(WORKSPACE_DIR, "QAdata", "qa_dataset_to_annotate.json"),
        ),
        answers_input_path=os.getenv(
            "B_ANSWER_INPUT_FILE",
            os.path.join(CURRENT_DIR, "group_b_collapsed_answers.json"),
        ),
        output_path=os.getenv(
            "B_EVAL_OUTPUT_FILE",
            os.path.join(CURRENT_DIR, "eval_results_GroupB.json"),
        ),
        evaluator_model=os.getenv("B_EVALUATOR_MODEL", DEFAULT_DOUBAO_ENDPOINT),
        llm_api_base=os.getenv("ARK_API_BASE", DEFAULT_ARK_API_BASE),
        only_qid=os.getenv("ONLY_QID", "").strip(),
    )


def load_golden_sources_map(annotate_dataset_path: str) -> dict[str, list[str]]:
    if not os.path.exists(annotate_dataset_path):
        raise FileNotFoundError(f"Annotated dataset not found: {annotate_dataset_path}")

    with open(annotate_dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    golden_map: dict[str, list[str]] = {}
    for item in data:
        qid = str(item.get("id", "")).strip()
        if not qid:
            continue
        raw_sources = item.get("golden_sources", [])
        if isinstance(raw_sources, list):
            golden_map[qid] = [str(x) for x in raw_sources if str(x).strip()]
        else:
            golden_map[qid] = []
    return golden_map


def _build_selected_node_lookup(layered_buckets: dict) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for nodes in layered_buckets.values():
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id", "")).strip()
            if node_id and node_id not in lookup:
                lookup[node_id] = node
    return lookup


def extract_purified_retrieved_sources(answer_item: dict) -> tuple[list[str], dict[str, int]]:
    retrieved_context = answer_item.get("retrieved_context") or {}
    layered_buckets = retrieved_context.get("layered_buckets") or {}
    selected_node_ids = answer_item.get("selected_node_ids") or []

    node_lookup = _build_selected_node_lookup(layered_buckets)

    purified_sources: set[str] = set()
    missing_nodes = 0

    for node_id in selected_node_ids:
        node = node_lookup.get(str(node_id))
        if not node:
            missing_nodes += 1
            continue

        source_file = str(node.get("source_file", "")).strip()
        raw_layer = node.get("layer", -1)
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError):
            layer = -1

        # Keep physical sources from bottom-layer nodes that were actually selected.
        if layer >= 1 or source_file == "cluster_summary":
            continue
        if layer == 0 and source_file:
            purified_sources.add(source_file)

    return sorted(purified_sources), {
        "selected_nodes": len(selected_node_ids),
        "missing_selected_nodes": missing_nodes,
        "purified_sources": len(purified_sources),
    }


def run_group_b_offline_evaluation() -> None:
    cfg = build_config_from_env()
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
        default_llm_api_base=cfg.llm_api_base,
    )

    if not runtime_cfg.ark_api_key:
        raise RuntimeError("ARK_API_KEY not found. Please set ARK_API_KEY before offline evaluation.")
    if not os.path.exists(cfg.answers_input_path):
        raise FileNotFoundError(f"Group B answers file not found: {cfg.answers_input_path}")

    golden_sources_map = load_golden_sources_map(cfg.annotate_dataset_path)

    with open(cfg.answers_input_path, "r", encoding="utf-8") as f:
        answer_items = json.load(f)

    evaluator = RAGEvaluator(
        model_name=cfg.evaluator_model,
        api_url=runtime_cfg.llm_api_base,
        api_key=runtime_cfg.ark_api_key,
    )

    results: list[dict] = []
    matched_count = 0

    for item in answer_items:
        qid = str(item.get("id", "")).strip()
        if not qid:
            continue
        if cfg.only_qid and qid != cfg.only_qid:
            continue

        matched_count += 1

        qtype = item.get("type", "")
        query = item.get("question", "")
        context = (item.get("retrieved_context") or {}).get("final_structured_context", "")
        answer = item.get("final_answer", "")
        golden_sources = golden_sources_map.get(qid, [])
        purified_retrieved_sources, source_stats = extract_purified_retrieved_sources(item)

        scores = evaluator.evaluate(
            query,
            context,
            answer,
            retrieved_sources=purified_retrieved_sources,
            golden_sources=golden_sources,
        )

        results.append(
            {
                "id": qid,
                "type": qtype,
                "question": query,
                "evaluation_scores": scores,
                "generated_answer": answer,
                "retrieved_sources": purified_retrieved_sources,
                "golden_sources": golden_sources,
            }
        )

        logger.info(
            "Offline evaluated %s: retrieved_sources=%d, golden_sources=%d, missing_selected_nodes=%d",
            qid,
            len(purified_retrieved_sources),
            len(golden_sources),
            source_stats["missing_selected_nodes"],
        )

    if cfg.only_qid and matched_count == 0:
        logger.warning("ONLY_QID=%s not found in Group B answers file.", cfg.only_qid)

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info("Group B offline evaluation completed. Results saved to %s", cfg.output_path)


if __name__ == "__main__":
    run_group_b_offline_evaluation()
