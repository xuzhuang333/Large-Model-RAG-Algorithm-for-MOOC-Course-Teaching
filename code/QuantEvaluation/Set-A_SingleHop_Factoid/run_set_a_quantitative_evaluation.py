from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_ROOT = os.path.dirname(CURRENT_DIR)
CODE_ROOT = os.path.dirname(QUANT_ROOT)
ABLATION_DIR = os.path.join(CODE_ROOT, "Retrieval Ablation Experiment")

GROUP_A_DIR = os.path.join(ABLATION_DIR, "Group A")
GROUP_B_DIR = os.path.join(ABLATION_DIR, "Group B")
GROUP_B_RETRIEVER_DIR = os.path.join(GROUP_B_DIR, "Retriever")
GROUP_C_DIR = os.path.join(ABLATION_DIR, "Group C")

DEFAULT_SET_A_PATH = os.path.join(CURRENT_DIR, "set_a_auto_generated.json")
DEFAULT_SAMPLE_OUT = os.path.join(CURRENT_DIR, "set_a_quant_eval_samples.json")
DEFAULT_SUMMARY_OUT = os.path.join(CURRENT_DIR, "set_a_quant_eval_summary.json")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SetA_Quantitative_Eval")


def _ensure_sys_paths() -> None:
    for path in (ABLATION_DIR, GROUP_A_DIR, GROUP_B_DIR, GROUP_B_RETRIEVER_DIR, GROUP_C_DIR):
        if path not in sys.path:
            sys.path.append(path)


_ensure_sys_paths()


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so decorators like @dataclass can resolve module globals.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


shared_utils = _load_module(
    "seta_shared_utils",
    os.path.join(ABLATION_DIR, "shared_retrieval_utils.py"),
)
group_a_pipeline_mod = _load_module(
    "seta_group_a_pipeline",
    os.path.join(GROUP_A_DIR, "group_a_pipeline.py"),
)
group_b_pipeline_mod = _load_module(
    "seta_group_b_pipeline",
    os.path.join(GROUP_B_RETRIEVER_DIR, "group_b_collapsed_tree_pipeline.py"),
)
group_c_retriever_mod = _load_module(
    "seta_group_c_retriever",
    os.path.join(GROUP_C_DIR, "group_c_dual_retriever.py"),
)
llm_eval_mod = _load_module(
    "seta_llm_eval",
    os.path.join(ABLATION_DIR, "llm_evaluator.py"),
)
generator_mod = _load_module(
    "seta_group_a_generator",
    os.path.join(GROUP_A_DIR, "llm_generator.py"),
)

QueryRewriter = shared_utils.QueryRewriter
RewriteConfig = shared_utils.RewriteConfig
GROUP_REWRITE_DEFAULTS = shared_utils.GROUP_REWRITE_DEFAULTS
DEFAULT_ARK_API_BASE = shared_utils.DEFAULT_ARK_API_BASE
DEFAULT_DEEPSEEK_ENDPOINT = shared_utils.DEFAULT_DEEPSEEK_ENDPOINT
DEFAULT_DOUBAO_ENDPOINT = shared_utils.DEFAULT_DOUBAO_ENDPOINT
load_runtime_config = shared_utils.load_runtime_config

GroupARetriever = group_a_pipeline_mod.GroupARetriever
_build_context_chunks_with_budget = group_a_pipeline_mod._build_context_chunks_with_budget

GroupBCollapsedTreeRetriever = group_b_pipeline_mod.GroupBCollapsedTreeRetriever
resolve_lineage_and_bucket = group_b_pipeline_mod.resolve_lineage_and_bucket
assemble_macro_to_micro_context = group_b_pipeline_mod.assemble_macro_to_micro_context

GroupCDualRetriever = group_c_retriever_mod.GroupCDualRetriever
GroupCDualRetrieverConfig = group_c_retriever_mod.GroupCDualRetrieverConfig

RAGGenerator = generator_mod.RAGGenerator
RAGEvaluator = llm_eval_mod.RAGEvaluator


@dataclass
class RewriteResult:
    rewrite_mode: str
    expanded_terms: list[str]
    rewritten_query_main: str
    rewritten_query_aux: str
    rewrite_latency_ms: float
    rewrite_fallback: bool


@dataclass
class RetrievalResult:
    ranked_items: list[dict[str, Any]]
    retrieved_sources: list[str]
    metric_sources: list[str]
    context_text: str
    retrieval_latency_ms: float
    debug: dict[str, Any]


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"count": 1, "mean": values[0], "std": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def _extract_first_match_start_segment(path_text: str) -> str:
    norm = path_text.replace("/", "\\").strip("\\")
    if not norm:
        return ""

    segments = [seg.strip() for seg in norm.split("\\") if seg.strip()]
    for idx, seg in enumerate(segments):
        if seg.startswith("【第"):
            return "\\".join(segments[idx:]).lower()

    # fallback: keep last up to 6 segments to improve matching for absolute paths
    return "\\".join(segments[-6:]).lower()


def _normalize_source(source: str) -> str:
    return _extract_first_match_start_segment(str(source or ""))


def _ordered_unique_sources(sources: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for src in sources:
        norm = _normalize_source(src)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result


_CANONICAL_SOURCE_EXTS = {".md", ".txt", ".srt"}
_SOURCE_CODE_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+){1,2})(?!\d)")


def _strip_known_source_extension(source_norm: str) -> str:
    base, ext = os.path.splitext(source_norm)
    if ext.lower() in _CANONICAL_SOURCE_EXTS:
        return base
    return source_norm


def _extract_unit_key(source_norm: str) -> str:
    match = _SOURCE_CODE_PATTERN.search(source_norm)
    if not match:
        return ""
    parts = match.group(1).split(".")
    if len(parts) < 2:
        return ""
    return f"{parts[0]}.{parts[1]}"


def _is_summary_source(source_norm: str) -> bool:
    return "#summary" in source_norm or "单元小结" in source_norm


def _source_equiv_key(source_norm: str) -> str:
    return _strip_known_source_extension(source_norm)


def _ordered_unique_source_equiv(source_norm_list: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for source_norm in source_norm_list:
        equiv = _source_equiv_key(source_norm)
        if not equiv or equiv in seen:
            continue
        seen.add(equiv)
        result.append(equiv)
    return result


def _extract_ranked_sources_from_items(ranked_items: list[dict[str, Any]]) -> list[str]:
    ordered_sources: list[str] = []
    seen: set[str] = set()
    for item in ranked_items:
        source = str(item.get("source_file") or item.get("source") or "").strip()
        source_norm = _normalize_source(source)
        if not source or not source_norm or source_norm in seen:
            continue
        seen.add(source_norm)
        ordered_sources.append(source)
    return ordered_sources


def _is_source_match(retrieved_source_norm: str, golden_source_norm: str) -> bool:
    if not retrieved_source_norm or not golden_source_norm:
        return False

    if retrieved_source_norm == golden_source_norm:
        return True

    if _source_equiv_key(retrieved_source_norm) == _source_equiv_key(golden_source_norm):
        return True

    # Relax to same-unit match only when retrieval hit is a summary source
    # while the golden source is a non-summary material in that same unit.
    if _is_summary_source(retrieved_source_norm) and not _is_summary_source(golden_source_norm):
        golden_unit = _extract_unit_key(golden_source_norm)
        retrieved_unit = _extract_unit_key(retrieved_source_norm)
        if golden_unit and golden_unit == retrieved_unit:
            return True

    return False


def _find_first_unmatched_golden(
    retrieved_source_norm: str,
    golden_sources_norm: list[str],
    matched_golden: list[bool],
) -> int | None:
    for idx, golden_source_norm in enumerate(golden_sources_norm):
        if matched_golden[idx]:
            continue
        if _is_source_match(retrieved_source_norm, golden_source_norm):
            return idx
    return None


def _maximum_bipartite_source_matches(retrieved_norm: list[str], golden_norm: list[str]) -> int:
    if not retrieved_norm or not golden_norm:
        return 0

    adjacency: list[list[int]] = []
    for retrieved_source in retrieved_norm:
        candidate_golden: list[int] = []
        for golden_idx, golden_source in enumerate(golden_norm):
            if _is_source_match(retrieved_source, golden_source):
                candidate_golden.append(golden_idx)
        adjacency.append(candidate_golden)

    match_to_retrieved = [-1] * len(golden_norm)

    def _dfs(retrieved_idx: int, seen_golden: list[bool]) -> bool:
        for golden_idx in adjacency[retrieved_idx]:
            if seen_golden[golden_idx]:
                continue
            seen_golden[golden_idx] = True
            if match_to_retrieved[golden_idx] == -1:
                match_to_retrieved[golden_idx] = retrieved_idx
                return True
            if _dfs(match_to_retrieved[golden_idx], seen_golden):
                match_to_retrieved[golden_idx] = retrieved_idx
                return True
        return False

    matched = 0
    for retrieved_idx in range(len(retrieved_norm)):
        seen = [False] * len(golden_norm)
        if _dfs(retrieved_idx, seen):
            matched += 1
    return matched


def _tokenize_text(text: str) -> list[str]:
    normalized = str(text or "").lower()
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", normalized)


def _normalize_answer_text(text: str) -> str:
    tokens = _tokenize_text(text)
    return "".join(tokens)


def _compute_em(prediction: str, gold: str) -> float:
    pred_norm = _normalize_answer_text(prediction)
    gold_norm = _normalize_answer_text(gold)
    if not pred_norm or not gold_norm:
        return 0.0

    # Strict exact match.
    if pred_norm == gold_norm:
        return 1.0

    # Relaxed match for generative long-form answers:
    # if normalized gold appears contiguously in prediction, count as correct.
    if gold_norm in pred_norm:
        return 1.0

    # Also allow concise prediction as a substantial contiguous part of gold.
    min_len = max(4, int(0.5 * len(gold_norm)))
    if len(pred_norm) >= min_len and pred_norm in gold_norm:
        return 1.0

    return 0.0


def _compute_token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _tokenize_text(prediction)
    gold_tokens = _tokenize_text(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_gold_points_coverage(answer: str, gold_points: list[str]) -> float | None:
    points = [p for p in gold_points if str(p).strip()]
    if not points:
        return None

    answer_norm = _normalize_answer_text(answer)
    hit = 0
    for point in points:
        if _normalize_answer_text(point) in answer_norm:
            hit += 1
    return hit / len(points)


def _is_refusal_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False

    normalized = _normalize_answer_text(text)
    refusal_markers = [
        "根据现有资料无法回答",
        "无法回答",
        "信息不足",
        "资料不足",
        "未提供相关信息",
        "no relevant context found",
    ]

    for marker in refusal_markers:
        marker_norm = _normalize_answer_text(marker)
        if marker_norm and marker_norm in normalized:
            return True
    return False


def _compute_retrieval_metrics(
    ranked_sources_norm: list[str],
    golden_sources_norm: list[str],
    k: int,
) -> dict[str, float]:
    k = max(1, int(k))
    if not golden_sources_norm:
        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "hit_at_1": 0.0,
            "ndcg_at_k": 0.0,
        }

    top_k = ranked_sources_norm[:k]
    matched_golden = [False] * len(golden_sources_norm)

    dcg = 0.0
    for idx, src in enumerate(top_k, start=1):
        matched_idx = _find_first_unmatched_golden(src, golden_sources_norm, matched_golden)
        rel = 0.0
        if matched_idx is not None:
            matched_golden[matched_idx] = True
            rel = 1.0
        dcg += rel / math.log2(idx + 1)

    recall_at_k = sum(1 for flag in matched_golden if flag) / len(golden_sources_norm)

    mrr = 0.0
    for rank, src in enumerate(ranked_sources_norm, start=1):
        if any(_is_source_match(src, golden_src) for golden_src in golden_sources_norm):
            mrr = 1.0 / rank
            break

    hit_at_1 = 0.0
    if ranked_sources_norm and any(
        _is_source_match(ranked_sources_norm[0], golden_src) for golden_src in golden_sources_norm
    ):
        hit_at_1 = 1.0

    ideal_hits = min(k, len(golden_sources_norm))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_k": float(recall_at_k),
        "mrr": float(mrr),
        "hit_at_1": float(hit_at_1),
        "ndcg_at_k": float(ndcg),
    }


def _compute_source_prf(retrieved_norm: list[str], golden_norm: list[str]) -> dict[str, float]:
    matched = _maximum_bipartite_source_matches(retrieved_norm, golden_norm)
    precision = matched / len(retrieved_norm) if retrieved_norm else 0.0
    recall = matched / len(golden_norm) if golden_norm else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "source_precision": float(precision),
        "source_recall": float(recall),
        "source_f1": float(f1),
    }


def _compute_source_precision_at_k(
    ranked_sources_norm: list[str],
    golden_sources_norm: list[str],
    k: int,
) -> float:
    top_k = ranked_sources_norm[: max(1, int(k))]
    if not top_k:
        return 0.0

    matched = _maximum_bipartite_source_matches(top_k, golden_sources_norm)
    return float(matched / len(top_k))


class SetARewriteGateway:
    def __init__(
        self,
        llm_api_base: str,
        ark_api_key: str | None,
        ab_model: str,
        a_rewrite_n: int,
        b_rewrite_n: int,
    ):
        self.a_rewriter = QueryRewriter(
            RewriteConfig(
                group_name="A",
                rewrite_n=max(1, a_rewrite_n),
                llm_model=ab_model,
                api_base=llm_api_base,
                api_key=ark_api_key,
            )
        )
        self.b_rewriter = QueryRewriter(
            RewriteConfig(
                group_name="B",
                rewrite_n=max(1, b_rewrite_n),
                llm_model=ab_model,
                api_base=llm_api_base,
                api_key=ark_api_key,
            )
        )

    @staticmethod
    def _compose_main_query(raw_query: str, terms: list[str]) -> str:
        uniq: list[str] = []
        seen: set[str] = set()
        for term in terms:
            t = str(term).strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        if not uniq:
            return raw_query
        return f"{raw_query} {' '.join(uniq)}"

    def rewrite_for_group(self, group: str, raw_query: str) -> RewriteResult:
        started = time.time()
        g = group.upper().strip()

        if g == "C":
            return RewriteResult(
                rewrite_mode="C_delegate_internal_rewrite",
                expanded_terms=[],
                rewritten_query_main=raw_query,
                rewritten_query_aux="",
                rewrite_latency_ms=(time.time() - started) * 1000,
                rewrite_fallback=False,
            )

        if g == "A":
            terms = self.a_rewriter.rewrite(raw_query)
        elif g == "B":
            terms = self.b_rewriter.rewrite(raw_query)
        else:
            terms = [raw_query]

        fallback = len(terms) == 1 and terms[0] == raw_query
        main_query = self._compose_main_query(raw_query, terms)

        return RewriteResult(
            rewrite_mode=f"{g}_shared_query_rewrite",
            expanded_terms=terms,
            rewritten_query_main=main_query,
            rewritten_query_aux="",
            rewrite_latency_ms=(time.time() - started) * 1000,
            rewrite_fallback=fallback,
        )


class SetAEvaluatorRunner:
    def __init__(self) -> None:
        runtime_cfg = load_runtime_config(
            default_uri="bolt://localhost:7687",
            default_user="neo4j",
            default_password="12345678",
            default_llm_api_base=DEFAULT_ARK_API_BASE,
        )

        self.runtime_cfg = runtime_cfg
        self.input_path = os.getenv("SET_A_INPUT_PATH", DEFAULT_SET_A_PATH)
        self.sample_out = os.getenv("SET_A_SAMPLE_OUTPUT_PATH", DEFAULT_SAMPLE_OUT)
        self.summary_out = os.getenv("SET_A_SUMMARY_OUTPUT_PATH", DEFAULT_SUMMARY_OUT)

        self.metric_k = max(1, int(os.getenv("SET_A_METRIC_K", "5")))
        self.limit_n = max(0, int(os.getenv("SET_A_LIMIT", "0")))
        self.random_sample_n = max(0, int(os.getenv("SET_A_RANDOM_SAMPLE_N", "0")))
        self.random_seed = int(os.getenv("SET_A_RANDOM_SEED", "42"))
        self.only_qid = os.getenv("SET_A_ONLY_QID", "").strip()

        self.enable_judge = _parse_bool(os.getenv("SET_A_ENABLE_JUDGE"), default=True)

        self.context_budget_chars = max(500, int(os.getenv("SET_A_CONTEXT_BUDGET_CHARS", "2000")))

        self.a_top_k_per_term = max(1, int(os.getenv("SET_A_A_TOP_K_PER_TERM", "3")))
        self.a_similarity_threshold = float(os.getenv("SET_A_A_SIMILARITY_THRESHOLD", "0.5"))

        self.b_top_k = max(1, int(os.getenv("SET_A_B_TOP_K", "20")))
        self.b_include_subsumed_layer0 = _parse_bool(os.getenv("SET_A_B_INCLUDE_SUBSUMED_LAYER0"), default=False)
        self.b_rel_type = os.getenv("SET_A_B_REL_TYPE", "GROUP_B_PARENT_OF").strip() or "GROUP_B_PARENT_OF"

        self.ab_rewrite_model = os.getenv("SET_A_AB_REWRITE_MODEL", DEFAULT_DEEPSEEK_ENDPOINT)
        self.a_rewrite_n = max(
            1,
            int(os.getenv("SET_A_A_REWRITE_N", str(GROUP_REWRITE_DEFAULTS["A"]["rewrite_n"]))),
        )
        self.b_rewrite_n = max(
            1,
            int(os.getenv("SET_A_B_REWRITE_N", str(GROUP_REWRITE_DEFAULTS["B"]["rewrite_n"]))),
        )

        self.a_generator_model = os.getenv("SET_A_A_GENERATOR_MODEL", DEFAULT_DEEPSEEK_ENDPOINT)
        self.b_generator_model = os.getenv("SET_A_B_GENERATOR_MODEL", DEFAULT_DEEPSEEK_ENDPOINT)
        self.c_generator_model = os.getenv("SET_A_C_GENERATOR_MODEL", DEFAULT_DEEPSEEK_ENDPOINT)

        self.judge_model = os.getenv("SET_A_JUDGE_MODEL", DEFAULT_DOUBAO_ENDPOINT)

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Set-A input not found: {self.input_path}")

        if not self.runtime_cfg.ark_api_key:
            raise RuntimeError("ARK_API_KEY missing. Please configure ARK_API_KEY.")

        self.rewrite_gateway = SetARewriteGateway(
            llm_api_base=self.runtime_cfg.llm_api_base,
            ark_api_key=self.runtime_cfg.ark_api_key,
            ab_model=self.ab_rewrite_model,
            a_rewrite_n=self.a_rewrite_n,
            b_rewrite_n=self.b_rewrite_n,
        )

        self.a_retriever = GroupARetriever(
            self.runtime_cfg.neo4j_uri,
            self.runtime_cfg.neo4j_user,
            self.runtime_cfg.neo4j_password,
            llm_model=self.ab_rewrite_model,
            llm_api_base=self.runtime_cfg.llm_api_base,
            ark_api_key=self.runtime_cfg.ark_api_key,
        )
        self.b_retriever = GroupBCollapsedTreeRetriever(
            uri=self.runtime_cfg.neo4j_uri,
            user=self.runtime_cfg.neo4j_user,
            password=self.runtime_cfg.neo4j_password,
            relationship_type=self.b_rel_type,
        )

        c_cfg = GroupCDualRetrieverConfig()
        c_cfg.state_weight_rho = 0.0
        self.c_retriever = GroupCDualRetriever(cfg=c_cfg)

        self.generators = {
            "A": self._make_generator(self.a_generator_model),
            "B": self._make_generator(self.b_generator_model),
            "C": self._make_generator(self.c_generator_model),
        }

        self.judge = (
            RAGEvaluator(
                model_name=self.judge_model,
                api_url=self.runtime_cfg.llm_api_base,
                api_key=self.runtime_cfg.ark_api_key,
            )
            if self.enable_judge
            else None
        )

    def _make_generator(self, model_name: str):
        generator = RAGGenerator(
            model_name=model_name,
            api_url=self.runtime_cfg.llm_api_base,
            api_key=self.runtime_cfg.ark_api_key,
        )
        # Force unified context budget for fair Set-A evaluation.
        generator.max_context_length = self.context_budget_chars
        return generator

    def close(self) -> None:
        self.a_retriever.close()
        self.b_retriever.close()
        self.c_retriever.close()

    def _load_set_a_samples(self) -> list[dict[str, Any]]:
        with open(self.input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError("Set-A input must be a JSON array.")

        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            eval_set = str(item.get("eval_set", "")).strip()
            max_hop = item.get("max_hop", None)
            noise_profile = str(item.get("noise_profile", "none")).strip().lower()

            if eval_set != "Set-A":
                continue
            if max_hop != 1:
                continue
            if noise_profile != "none":
                continue

            qid = str(item.get("id", "")).strip()
            if not qid:
                continue
            if self.only_qid and qid != self.only_qid:
                continue

            rows.append(item)

        if self.random_sample_n > 0:
            sample_n = min(self.random_sample_n, len(rows))
            rows = random.Random(self.random_seed).sample(rows, sample_n)
        elif self.limit_n > 0:
            rows = rows[: self.limit_n]

        return rows

    def _extract_b_layer0_sources(self, nodes: list[dict[str, Any]]) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            layer = int(node.get("layer", -1))
            src = str(node.get("source_file", "")).strip()
            if layer != 0:
                continue
            if not src or src == "cluster_summary":
                continue
            key = _normalize_source(src)
            if not key or key in seen:
                continue
            seen.add(key)
            sources.append(src)
        return sources

    def _lookup_c_snippet_sources(self, snippet_ids: list[str]) -> dict[str, str]:
        clean_ids = [str(s).strip() for s in snippet_ids if str(s).strip()]
        if not clean_ids:
            return {}

        cypher = """
        UNWIND $snippet_ids AS sid
        OPTIONAL MATCH (t:GroupC_TextSnippet {snippet_id: sid})
        OPTIONAL MATCH (c:GroupC_CodeSnippet {snippet_id: sid})
        RETURN sid AS snippet_id,
               coalesce(t.source_file, c.source_file, '') AS source_file
        """

        mapping: dict[str, str] = {}
        with self.c_retriever.driver.session() as session:
            records = list(session.run(cypher, snippet_ids=clean_ids))
        for record in records:
            sid = str(record.get("snippet_id", "")).strip()
            src = str(record.get("source_file", "")).strip()
            if sid:
                mapping[sid] = src
        return mapping

    def _retrieve_with_group(
        self,
        group: str,
        sample: dict[str, Any],
        rewrite: RewriteResult,
    ) -> RetrievalResult:
        question = str(sample.get("question", "")).strip()
        started = time.time()

        if group == "A":
            terms = rewrite.expanded_terms if rewrite.expanded_terms else [question]
            raw_chunks = self.a_retriever.flat_vector_search(
                terms,
                top_k_per_term=self.a_top_k_per_term,
                similarity_threshold=self.a_similarity_threshold,
            )
            formatted_chunks, retrieved_sources, total_chars = _build_context_chunks_with_budget(
                raw_chunks,
                budget_chars=self.context_budget_chars,
            )
            context_text = "\n".join(formatted_chunks)
            return RetrievalResult(
                ranked_items=raw_chunks,
                retrieved_sources=retrieved_sources,
                metric_sources=_extract_ranked_sources_from_items(raw_chunks),
                context_text=context_text,
                retrieval_latency_ms=(time.time() - started) * 1000,
                debug={
                    "chunk_count": len(raw_chunks),
                    "context_chars": total_chars,
                },
            )

        if group == "B":
            raw_nodes = self.b_retriever.retrieve_global_topk(
                user_query=rewrite.rewritten_query_main,
                top_k=self.b_top_k,
            )
            raw_layer0_count = sum(1 for n in raw_nodes if int(n.get("layer", -1)) == 0)
            layered_buckets, lineage_metrics = resolve_lineage_and_bucket(raw_nodes)
            assembled_context, selected_nodes, assemble_metrics = assemble_macro_to_micro_context(
                layered_buckets=layered_buckets,
                include_subsumed_layer0=self.b_include_subsumed_layer0,
                context_max_chars=self.context_budget_chars,
            )
            selected_layer0_count = sum(1 for n in selected_nodes if int(n.get("layer", -1)) == 0)
            context_sources = self._extract_b_layer0_sources(selected_nodes)
            raw_layer0_sources = self._extract_b_layer0_sources(raw_nodes)
            metric_sources = raw_layer0_sources if raw_layer0_sources else context_sources

            return RetrievalResult(
                ranked_items=raw_nodes,
                retrieved_sources=context_sources,
                metric_sources=metric_sources,
                context_text=assembled_context,
                retrieval_latency_ms=(time.time() - started) * 1000,
                debug={
                    "lineage_metrics": lineage_metrics,
                    "assemble_metrics": assemble_metrics,
                    "selected_nodes": len(selected_nodes),
                    "raw_layer0_count": raw_layer0_count,
                    "selected_layer0_count": selected_layer0_count,
                    "raw_layer0_source_count": len(raw_layer0_sources),
                    "context_layer0_source_count": len(context_sources),
                },
            )

        if group == "C":
            result_obj = self.c_retriever.retrieve(
                query=question,
                user_id=None,
                current_turn=None,
                qa_score=None,
                current_struggle=None,
            )
            result_dict = result_obj.to_dict()
            candidates = result_dict.get("candidates", [])
            snippet_ids = [str(c.get("snippet_id", "")).strip() for c in candidates]
            source_map = self._lookup_c_snippet_sources(snippet_ids)

            ranked_items: list[dict[str, Any]] = []
            retrieved_sources: list[str] = []
            metric_sources: list[str] = []
            seen_context_source: set[str] = set()
            seen_metric_source: set[str] = set()
            context_lines: list[str] = []
            context_chars = 0

            for rank, item in enumerate(candidates, start=1):
                sid = str(item.get("snippet_id", "")).strip()
                src = source_map.get(sid, "")
                content = str(item.get("content", "") or "")
                score = _safe_float(item.get("final_score", 0.0), default=0.0)

                ranked_items.append(
                    {
                        "rank": rank,
                        "snippet_id": sid,
                        "parent_syllabus_id": str(item.get("parent_syllabus_id", "")).strip(),
                        "source_file": src,
                        "score": score,
                    }
                )

                source_key = _normalize_source(src)
                if src and source_key and source_key not in seen_metric_source:
                    seen_metric_source.add(source_key)
                    metric_sources.append(src)

                line = f"[Source: {src or 'UNKNOWN_SOURCE'}]\n{content}"
                if context_chars + len(line) <= self.context_budget_chars:
                    context_lines.append(line)
                    context_chars += len(line)
                    if src and source_key and source_key not in seen_context_source:
                        seen_context_source.add(source_key)
                        retrieved_sources.append(src)

            return RetrievalResult(
                ranked_items=ranked_items,
                retrieved_sources=retrieved_sources,
                metric_sources=metric_sources,
                context_text="\n".join(context_lines),
                retrieval_latency_ms=(time.time() - started) * 1000,
                debug={
                    "group_c_debug": result_dict.get("debug_metrics", {}),
                    "candidate_count": len(candidates),
                    "context_chars": context_chars,
                    "source_lookup_count": len(source_map),
                },
            )

        raise ValueError(f"Unsupported group: {group}")

    def _generate_answer(
        self,
        group: str,
        question: str,
        context_text: str,
        answer_mode: str | None = None,
    ) -> tuple[str, str, float]:
        started = time.time()
        generator = self.generators[group]
        chunks = [context_text] if context_text else ["No relevant context found."]
        answer, context_used = generator.generate_response(
            question,
            chunks,
            answer_mode=answer_mode,
        )
        return answer, context_used, (time.time() - started) * 1000

    def _evaluate_sample_group(
        self,
        sample: dict[str, Any],
        group: str,
    ) -> dict[str, Any]:
        qid = str(sample.get("id", "")).strip()
        qtype = str(sample.get("type", "")).strip()
        question = str(sample.get("question", "")).strip()
        ground_truth = str(sample.get("ground_truth", "")).strip()
        golden_sources_raw = sample.get("golden_sources", [])
        gold_points_raw = sample.get("gold_answer_points", [])

        golden_sources = [str(src).strip() for src in golden_sources_raw if str(src).strip()]
        gold_points = [str(p).strip() for p in gold_points_raw if str(p).strip()]

        rewrite = self.rewrite_gateway.rewrite_for_group(group=group, raw_query=question)
        retrieval = self._retrieve_with_group(group=group, sample=sample, rewrite=rewrite)

        answer, context_used, generation_latency_ms = self._generate_answer(
            group=group,
            question=question,
            context_text=retrieval.context_text,
            answer_mode=qtype,
        )

        context_sources_norm = _ordered_unique_sources(retrieval.retrieved_sources)
        metric_sources_norm = _ordered_unique_sources(
            retrieval.metric_sources if retrieval.metric_sources else retrieval.retrieved_sources
        )
        golden_sources_norm = _ordered_unique_sources(golden_sources)

        context_sources_equiv_norm = _ordered_unique_source_equiv(context_sources_norm)
        metric_sources_equiv_norm = _ordered_unique_source_equiv(metric_sources_norm)
        golden_sources_equiv_norm = _ordered_unique_source_equiv(golden_sources_norm)

        retrieval_metrics = _compute_retrieval_metrics(
            ranked_sources_norm=metric_sources_equiv_norm,
            golden_sources_norm=golden_sources_equiv_norm,
            k=self.metric_k,
        )
        source_prf = _compute_source_prf(metric_sources_equiv_norm, golden_sources_equiv_norm)
        source_precision_at_k = _compute_source_precision_at_k(
            ranked_sources_norm=metric_sources_equiv_norm,
            golden_sources_norm=golden_sources_equiv_norm,
            k=self.metric_k,
        )

        answer_em = _compute_em(answer, ground_truth)
        answer_f1 = _compute_token_f1(answer, ground_truth)
        gold_points_coverage = _compute_gold_points_coverage(answer, gold_points)
        refusal_rate = 1.0 if _is_refusal_answer(answer) else 0.0

        judge_scores = None
        if self.judge is not None:
            judge_scores = self.judge.evaluate(
                question,
                context_used,
                answer,
                retrieved_sources=retrieval.retrieved_sources,
                golden_sources=golden_sources,
            )

        return {
            "id": qid,
            "type": qtype,
            "group": group,
            "eval_set": "Set-A",
            "question": question,
            "ground_truth": ground_truth,
            "rewrite": {
                "rewrite_mode": rewrite.rewrite_mode,
                "expanded_terms": rewrite.expanded_terms,
                "rewritten_query_main": rewrite.rewritten_query_main,
                "rewritten_query_aux": rewrite.rewritten_query_aux,
                "rewrite_latency_ms": round(rewrite.rewrite_latency_ms, 2),
                "rewrite_fallback": rewrite.rewrite_fallback,
            },
            "retrieval": {
                "retrieved_sources": retrieval.retrieved_sources,
                "retrieved_sources_norm": context_sources_norm,
                "retrieved_sources_equiv_norm": context_sources_equiv_norm,
                "metric_sources": retrieval.metric_sources,
                "metric_sources_norm": metric_sources_norm,
                "metric_sources_equiv_norm": metric_sources_equiv_norm,
                "golden_sources": golden_sources,
                "golden_sources_norm": golden_sources_norm,
                "golden_sources_equiv_norm": golden_sources_equiv_norm,
                "ranked_items_count": len(retrieval.ranked_items),
                "context_text_chars": len(retrieval.context_text),
                "retrieval_latency_ms": round(retrieval.retrieval_latency_ms, 2),
                "debug": {
                    **retrieval.debug,
                    "metric_k_requested": self.metric_k,
                    "metric_k_effective": min(self.metric_k, len(metric_sources_equiv_norm)),
                    "metric_source_count": len(metric_sources_equiv_norm),
                    "context_source_count": len(context_sources_equiv_norm),
                    "golden_source_count": len(golden_sources_equiv_norm),
                },
            },
            "generation": {
                "answer": answer,
                "generation_latency_ms": round(generation_latency_ms, 2),
                "answer_mode": qtype,
                "is_refusal": bool(refusal_rate > 0.0),
            },
            "metrics": {
                **retrieval_metrics,
                **source_prf,
                "source_precision_at_k": float(source_precision_at_k),
                "answer_em": float(answer_em),
                "answer_f1": float(answer_f1),
                "gold_points_coverage": gold_points_coverage,
                "refusal_rate": float(refusal_rate),
            },
            "judge_metrics": judge_scores,
            "errors": [],
        }

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        metric_keys = [
            "recall_at_k",
            "mrr",
            "hit_at_1",
            "ndcg_at_k",
            "source_precision",
            "source_precision_at_k",
            "source_recall",
            "source_f1",
            "answer_em",
            "answer_f1",
            "gold_points_coverage",
            "refusal_rate",
        ]

        judge_keys = [
            "Context_Recall",
            "Context_Precision",
            "Faithfulness",
            "Answer_Relevance",
            "Source_Alignment",
        ]

        groups = sorted({row.get("group", "") for row in rows if row.get("group")})
        types = sorted({row.get("type", "") for row in rows if row.get("type")})

        summary: dict[str, Any] = {
            "total_rows": len(rows),
            "metric_k": self.metric_k,
            "judge_enabled": self.enable_judge,
            "by_group": {},
            "by_group_and_type": {},
        }

        for group in groups:
            group_rows = [r for r in rows if r.get("group") == group]
            metric_summary: dict[str, Any] = {}
            for key in metric_keys:
                values = []
                for row in group_rows:
                    val = row.get("metrics", {}).get(key)
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                metric_summary[key] = _mean_std(values)

            judge_summary: dict[str, Any] = {}
            if self.enable_judge:
                for key in judge_keys:
                    values = []
                    for row in group_rows:
                        judge_row = row.get("judge_metrics") or {}
                        val = judge_row.get(key)
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                    judge_summary[key] = _mean_std(values)

            latency_values = []
            for row in group_rows:
                r_ms = row.get("retrieval", {}).get("retrieval_latency_ms")
                g_ms = row.get("generation", {}).get("generation_latency_ms")
                if isinstance(r_ms, (int, float)) and isinstance(g_ms, (int, float)):
                    latency_values.append(float(r_ms) + float(g_ms))

            summary["by_group"][group] = {
                "count": len(group_rows),
                "metrics": metric_summary,
                "judge_metrics": judge_summary,
                "total_latency_ms": _mean_std(latency_values),
            }

            for qtype in types:
                bucket_key = f"{group}:{qtype}"
                bucket_rows = [r for r in group_rows if r.get("type") == qtype]
                bucket_metric_summary: dict[str, Any] = {}
                for key in metric_keys:
                    values = []
                    for row in bucket_rows:
                        val = row.get("metrics", {}).get(key)
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                    bucket_metric_summary[key] = _mean_std(values)

                summary["by_group_and_type"][bucket_key] = {
                    "count": len(bucket_rows),
                    "metrics": bucket_metric_summary,
                }

        return summary

    def run(self) -> None:
        samples = self._load_set_a_samples()
        if not samples:
            logger.warning("No Set-A samples to evaluate.")
            return

        logger.info(
            "Set-A evaluation started: samples=%d, metric_k=%d, judge_enabled=%s",
            len(samples),
            self.metric_k,
            self.enable_judge,
        )

        output_rows: list[dict[str, Any]] = []

        for idx, sample in enumerate(samples, start=1):
            qid = str(sample.get("id", "")).strip()
            logger.info("Evaluating sample %d/%d: %s", idx, len(samples), qid)

            for group in ("A", "B", "C"):
                try:
                    row = self._evaluate_sample_group(sample=sample, group=group)
                except Exception as exc:
                    logger.error("Failed sample=%s group=%s, error=%s", qid, group, exc)
                    row = {
                        "id": qid,
                        "type": str(sample.get("type", "")).strip(),
                        "group": group,
                        "eval_set": "Set-A",
                        "question": str(sample.get("question", "")).strip(),
                        "rewrite": {},
                        "retrieval": {},
                        "generation": {},
                        "metrics": {},
                        "judge_metrics": None,
                        "errors": [str(exc)],
                    }
                output_rows.append(row)

        summary = self._aggregate(output_rows)

        os.makedirs(os.path.dirname(self.sample_out), exist_ok=True)
        os.makedirs(os.path.dirname(self.summary_out), exist_ok=True)

        with open(self.sample_out, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, ensure_ascii=False, indent=2)

        with open(self.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Set-A sample metrics saved: %s", self.sample_out)
        logger.info("Set-A summary metrics saved: %s", self.summary_out)


def main() -> None:
    runner = SetAEvaluatorRunner()
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
