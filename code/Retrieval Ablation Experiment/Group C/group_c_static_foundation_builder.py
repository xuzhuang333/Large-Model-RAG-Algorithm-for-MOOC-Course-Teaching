#该文件是Group C静态基础构建的核心脚本，负责从指定的课程材料目录中解析结构化Markdown文档、纯文本文件和字幕文件，提取文本和代码片段，并将它们与课程大纲节点关联。脚本还会生成面向教学的节点摘要，并将所有数据存储到Neo4j图数据库中，为后续的检索和分析提供基础。
from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
WORKSPACE_DIR = os.path.dirname(PROJECT_DIR)
GROUP_B_DIR = os.path.join(PROJECT_DIR, "Group B")

for path in (PROJECT_DIR, GROUP_B_DIR):
    if path not in sys.path:
        sys.path.append(path)

from chunking_utils import TokenCounter, build_sentence_complete_chunks
from group_c_md_structured_parser import parse_structured_markdown
from neo4j_ops import GroupCStaticRepository
from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    ark_chat_completion,
    get_adaptive_generation_params,
    load_runtime_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GroupC_Static_Foundation")


SRT_TS_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$")


@dataclass
class GroupCStaticConfig:
    dataset_root: str
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    text_token_limit: int = 120
    hard_token_cap: int = 400
    embedding_batch_size: int = 64
    clear_before_insert: bool = True
    create_indexes: bool = True
    build_upward_summary: bool = True
    summary_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    summary_api_base: str = DEFAULT_ARK_API_BASE
    summary_max_chars: int = 7000
    dry_run_parse_only: bool = False
    smoke_markdown_path: str | None = None


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_config_from_env() -> GroupCStaticConfig:
    dataset_root = os.getenv(
        "DATASET_ROOT",
        r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学",
    )

    legacy_clear_before_insert = os.getenv("GROUP_C_CLEAR_BEFORE_INSERT")
    if legacy_clear_before_insert and legacy_clear_before_insert.strip().lower() in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        logger.warning(
            "GROUP_C_CLEAR_BEFORE_INSERT=%s ignored; clear-before-insert is forced to true for overwrite reruns.",
            legacy_clear_before_insert,
        )

    return GroupCStaticConfig(
        dataset_root=os.path.abspath(dataset_root),
        embedding_model_name=os.getenv("GROUP_C_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        text_token_limit=max(32, int(os.getenv("GROUP_C_TEXT_TOKEN_LIMIT", "120"))),
        hard_token_cap=max(64, int(os.getenv("GROUP_C_HARD_TOKEN_CAP", "400"))),
        embedding_batch_size=max(1, int(os.getenv("GROUP_C_EMBEDDING_BATCH_SIZE", "64"))),
        clear_before_insert=True,
        create_indexes=_parse_bool(os.getenv("GROUP_C_CREATE_INDEXES"), default=True),
        build_upward_summary=_parse_bool(os.getenv("GROUP_C_BUILD_UPWARD_SUMMARY"), default=True),
        summary_model=os.getenv("GROUP_C_SUMMARY_MODEL", DEFAULT_DEEPSEEK_ENDPOINT),
        summary_api_base=os.getenv("ARK_API_BASE", DEFAULT_ARK_API_BASE),
        summary_max_chars=max(1000, int(os.getenv("GROUP_C_SUMMARY_MAX_CHARS", "7000"))),
        dry_run_parse_only=_parse_bool(os.getenv("GROUP_C_DRY_RUN_PARSE_ONLY"), default=False),
        smoke_markdown_path=os.getenv("GROUP_C_SMOKE_MARKDOWN_PATH"),
    )


def _stable_id(seed: str, prefix: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _extract_week_tag(abs_path: str) -> str | None:
    match = re.search(r"【第\d+周】", abs_path)
    return match.group(0) if match else None


def _clean_srt_text(raw_text: str) -> str:
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


def _iter_directory_paths(dataset_root: str) -> list[str]:
    paths: list[str] = []
    for dirpath, _, _ in os.walk(dataset_root):
        abs_dir = os.path.abspath(dirpath)
        paths.append(abs_dir)
    return sorted(paths)


def build_syllabus_rows(dataset_root: str) -> list[dict]:
    dir_paths = _iter_directory_paths(dataset_root)
    included = set(dir_paths)

    rows: list[dict] = []
    for abs_dir in dir_paths:
        parent_abs = os.path.abspath(os.path.dirname(abs_dir))
        parent_included = parent_abs if parent_abs in included else None

        rel = os.path.relpath(abs_dir, dataset_root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1

        rows.append(
            {
                "node_id": _stable_id(abs_dir, "GC_SYL"),
                "parent_node_id": _stable_id(parent_included, "GC_SYL") if parent_included else None,
                "name": os.path.basename(abs_dir) if rel != "." else os.path.basename(dataset_root),
                "abs_path": abs_dir,
                "parent_abs_path": parent_included,
                "depth": depth,
                "week_tag": _extract_week_tag(abs_dir),
            }
        )

    return rows


def _collect_markdown_snippets(
    abs_file_path: str,
    syllabus_node_id: str,
    token_counter: TokenCounter,
) -> tuple[list[dict], list[dict]]:
    with open(abs_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    parsed = parse_structured_markdown(markdown_text)

    source_file_abs = os.path.abspath(abs_file_path)
    source_type = "md"

    meta_source_file = parsed.metadata.get("source_file", "")
    meta_instructor = parsed.metadata.get("instructor", "")
    meta_document_type = parsed.metadata.get("document_type", "")
    meta_core_keywords = parsed.metadata.get("core_keywords", "")

    text_rows: list[dict] = []
    code_rows: list[dict] = []

    for snippet in parsed.snippets:
        trace = ",".join(snippet.citations)
        if snippet.snippet_kind == "text":
            payload_text = f"{snippet.context_prefix}\n{snippet.text_content}".strip()
            if not payload_text:
                continue
            snippet_id = _stable_id(
                f"TEXT::{source_file_abs}::{snippet.section_order}::{snippet.chunk_order}::{snippet.chunk_type}::{snippet.chunk_title}::{payload_text}",
                "GC_TXT",
            )
            text_rows.append(
                {
                    "snippet_id": snippet_id,
                    "syllabus_node_id": syllabus_node_id,
                    "text": payload_text,
                    "context_prefix": snippet.context_prefix,
                    "chunk_type": snippet.chunk_type,
                    "chunk_title": snippet.chunk_title,
                    "section_name": snippet.section_name,
                    "section_order": snippet.section_order,
                    "chunk_order": snippet.chunk_order,
                    "course_material_title": parsed.course_material_title,
                    "metadata_source_file": meta_source_file,
                    "metadata_instructor": meta_instructor,
                    "metadata_document_type": meta_document_type,
                    "metadata_core_keywords": meta_core_keywords,
                    "source_file": source_file_abs,
                    "source_type": source_type,
                    "trace": trace,
                    "token_count": token_counter.count_tokens(payload_text),
                    "summary_level": -1,
                    "is_generated_summary": False,
                }
            )
        else:
            payload_code = snippet.code_content.strip()
            if not payload_code:
                continue
            snippet_id = _stable_id(
                f"CODE::{source_file_abs}::{snippet.section_order}::{snippet.chunk_order}::{snippet.chunk_type}::{snippet.chunk_title}::{payload_code}",
                "GC_CODE",
            )
            code_rows.append(
                {
                    "snippet_id": snippet_id,
                    "syllabus_node_id": syllabus_node_id,
                    "code": payload_code,
                    "context_prefix": snippet.context_prefix,
                    "chunk_type": snippet.chunk_type,
                    "chunk_title": snippet.chunk_title,
                    "section_name": snippet.section_name,
                    "section_order": snippet.section_order,
                    "chunk_order": snippet.chunk_order,
                    "course_material_title": parsed.course_material_title,
                    "metadata_source_file": meta_source_file,
                    "metadata_instructor": meta_instructor,
                    "metadata_document_type": meta_document_type,
                    "metadata_core_keywords": meta_core_keywords,
                    "source_file": source_file_abs,
                    "source_type": source_type,
                    "trace": trace,
                    "token_count": token_counter.count_tokens(payload_code),
                }
            )

    return text_rows, code_rows


def _collect_plain_text_snippets(
    abs_file_path: str,
    syllabus_node_id: str,
    token_counter: TokenCounter,
    token_limit: int,
    hard_token_cap: int,
) -> list[dict]:
    ext = os.path.splitext(abs_file_path)[1].lower()
    source_type = ext.lstrip(".")

    with open(abs_file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = _clean_srt_text(raw) if ext == ".srt" else raw
    chunks = build_sentence_complete_chunks(
        text,
        token_limit=token_limit,
        token_counter=token_counter,
        hard_token_cap=hard_token_cap,
    )

    rows: list[dict] = []
    source_file_abs = os.path.abspath(abs_file_path)
    for idx, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue

        chunk_title = f"{source_type}_chunk_{idx}"
        context_prefix = f"[Section: Raw {source_type.upper()}] -> [Chunk: {chunk_title}]"
        payload_text = f"{context_prefix}\n{chunk_text}".strip()

        snippet_id = _stable_id(
            f"PLAIN::{source_file_abs}::{idx}::{payload_text}",
            "GC_TXT",
        )
        rows.append(
            {
                "snippet_id": snippet_id,
                "syllabus_node_id": syllabus_node_id,
                "text": payload_text,
                "context_prefix": context_prefix,
                "chunk_type": f"{source_type}_chunk",
                "chunk_title": chunk_title,
                "section_name": f"Raw {source_type.upper()}",
                "section_order": 0,
                "chunk_order": idx,
                "course_material_title": os.path.basename(source_file_abs),
                "metadata_source_file": os.path.basename(source_file_abs),
                "metadata_instructor": "",
                "metadata_document_type": "raw_text",
                "metadata_core_keywords": "",
                "source_file": source_file_abs,
                "source_type": source_type,
                "trace": "",
                "token_count": int(chunk.get("token_count", token_counter.count_tokens(payload_text))),
                "summary_level": -1,
                "is_generated_summary": False,
            }
        )

    return rows


def collect_content_rows(
    dataset_root: str,
    token_counter: TokenCounter,
    token_limit: int,
    hard_token_cap: int,
) -> tuple[list[dict], list[dict]]:
    text_rows: list[dict] = []
    code_rows: list[dict] = []

    for dirpath, _, filenames in os.walk(dataset_root):
        abs_dir = os.path.abspath(dirpath)

        syllabus_node_id = _stable_id(abs_dir, "GC_SYL")

        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in {".md", ".srt", ".txt"}:
                continue

            abs_file_path = os.path.abspath(os.path.join(abs_dir, filename))
            try:
                if ext == ".md":
                    md_text_rows, md_code_rows = _collect_markdown_snippets(
                        abs_file_path,
                        syllabus_node_id,
                        token_counter,
                    )
                    text_rows.extend(md_text_rows)
                    code_rows.extend(md_code_rows)
                else:
                    plain_rows = _collect_plain_text_snippets(
                        abs_file_path,
                        syllabus_node_id,
                        token_counter,
                        token_limit,
                        hard_token_cap,
                    )
                    text_rows.extend(plain_rows)
            except Exception as exc:
                logger.warning("Skip file due to parse error: %s, error=%s", abs_file_path, exc)

    return text_rows, code_rows


def attach_embeddings(
    text_rows: list[dict],
    code_rows: list[dict],
    model: SentenceTransformer,
    batch_size: int,
) -> None:
    if text_rows:
        texts = [row["text"] for row in text_rows]
        text_embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        for row, emb in zip(text_rows, text_embeddings):
            row["embedding"] = [float(x) for x in emb]

    if code_rows:
        code_texts = [f"{row['context_prefix']}\n{row['code']}" for row in code_rows]
        code_embeddings = model.encode(
            code_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        for row, emb in zip(code_rows, code_embeddings):
            row["embedding"] = [float(x) for x in emb]


def _summarize_node_text(
    node_name: str,
    context_text: str,
    model: str,
    api_base: str,
    api_key: str | None,
) -> str:
    prompt = f"""
你是课程知识图谱摘要助手。请基于给定文本为节点生成一个简洁、可检索、面向教学的中文摘要。
要求：
1. 仅基于输入文本，不引入外部知识。
2. 保留关键定义、差异点、步骤与约束。
3. 输出单段文本，不要使用标题或编号。

[Node]
{node_name}

[Context]
{context_text}

请输出摘要：
"""

    try:
        temperature, max_tokens = get_adaptive_generation_params(prompt, task="generate")
        return ark_chat_completion(
            model=model,
            prompt=prompt,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=180,
        ).strip()
    except Exception as exc:
        logger.warning("Summary generation fallback for node=%s, error=%s", node_name, exc)
        return context_text[:800].strip() or "根据现有资料无法生成摘要"


def build_upward_summaries(
    repo: GroupCStaticRepository,
    embedding_model: SentenceTransformer,
    token_counter: TokenCounter,
    cfg: GroupCStaticConfig,
    runtime_api_key: str | None,
) -> dict[str, int]:
    cleared_count = repo.clear_generated_summaries()
    stats = {
        "summaries_cleared": cleared_count,
        "summaries_created": 0,
        "nodes_scanned": 0,
        "code_links_propagated": 0,
    }

    nodes = repo.fetch_syllabus_nodes_depth_desc()
    for node in nodes:
        stats["nodes_scanned"] += 1

        node_id = node["node_id"]
        node_name = node["name"]

        own_texts = repo.fetch_direct_texts(node_id, include_generated=False)
        child_nodes = repo.fetch_children(node_id)

        child_summary_texts: list[str] = []
        for child in child_nodes:
            child_texts = repo.fetch_direct_texts(child["node_id"], include_generated=True)
            for item in child_texts:
                if item["is_generated_summary"]:
                    child_summary_texts.append(item["text"])
                    break

        source_parts: list[str] = []
        for item in child_summary_texts:
            source_parts.append(item)
        for item in own_texts:
            source_parts.append(item["text"])

        repo_links = repo.propagate_code_links_from_children(node_id)
        stats["code_links_propagated"] += repo_links

        if not source_parts:
            continue

        merged_context = "\n\n".join(source_parts)
        if len(merged_context) > cfg.summary_max_chars:
            merged_context = merged_context[: cfg.summary_max_chars]

        summary_text = _summarize_node_text(
            node_name=node_name,
            context_text=merged_context,
            model=cfg.summary_model,
            api_base=cfg.summary_api_base,
            api_key=runtime_api_key,
        )
        if not summary_text:
            continue

        summary_embedding = embedding_model.encode(
            [summary_text],
            normalize_embeddings=True,
            batch_size=1,
        )[0]

        summary_snippet_id = _stable_id(f"SUMMARY::{node_id}", "GC_TXT")

        summary_row = {
            "snippet_id": summary_snippet_id,
            "syllabus_node_id": node_id,
            "text": summary_text,
            "context_prefix": f"[Section: NodeSummary] -> [Chunk: {node_name}]",
            "chunk_type": "syllabus_summary",
            "chunk_title": f"Summary of {node_name}",
            "section_name": "NodeSummary",
            "section_order": 999,
            "chunk_order": 999,
            "course_material_title": node_name,
            "metadata_source_file": "",
            "metadata_instructor": "",
            "metadata_document_type": "summary_generated",
            "metadata_core_keywords": "",
            "source_file": f"{node['abs_path']}#summary",
            "source_type": "summary_generated",
            "trace": "",
            "token_count": token_counter.count_tokens(summary_text),
            "summary_level": int(node["depth"]),
            "is_generated_summary": True,
            "embedding": [float(x) for x in summary_embedding],
        }

        inserted = repo.upsert_text_snippets([summary_row])
        stats["summaries_created"] += inserted

    return stats


def run_parse_smoke_check(smoke_markdown_path: str) -> None:
    abs_md = os.path.abspath(smoke_markdown_path)
    with open(abs_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    parsed = parse_structured_markdown(md_text)
    text_count = sum(1 for s in parsed.snippets if s.snippet_kind == "text")
    code_count = sum(1 for s in parsed.snippets if s.snippet_kind == "code")
    logger.info(
        "Smoke parse ok: file=%s, course_title=%s, metadata_keys=%s, text_snippets=%d, code_snippets=%d",
        abs_md,
        parsed.course_material_title,
        sorted(parsed.metadata.keys()),
        text_count,
        code_count,
    )


def run_group_c_static_foundation() -> None:
    cfg = build_config_from_env()

    if cfg.smoke_markdown_path:
        run_parse_smoke_check(cfg.smoke_markdown_path)

    token_counter = TokenCounter(model_name=cfg.embedding_model_name)

    syllabus_rows = build_syllabus_rows(cfg.dataset_root)
    text_rows, code_rows = collect_content_rows(
        cfg.dataset_root,
        token_counter,
        cfg.text_token_limit,
        cfg.hard_token_cap,
    )

    logger.info(
        "Collected static content: syllabus_nodes=%d, text_snippets=%d, code_snippets=%d",
        len(syllabus_rows),
        len(text_rows),
        len(code_rows),
    )

    if cfg.dry_run_parse_only:
        logger.info("GROUP_C_DRY_RUN_PARSE_ONLY=true, skip Neo4j write and stop after parsing.")
        return

    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
        default_llm_api_base=cfg.summary_api_base,
    )

    embedding_model = SentenceTransformer(cfg.embedding_model_name)
    attach_embeddings(
        text_rows=text_rows,
        code_rows=code_rows,
        model=embedding_model,
        batch_size=cfg.embedding_batch_size,
    )

    repo = GroupCStaticRepository(
        uri=runtime_cfg.neo4j_uri,
        user=runtime_cfg.neo4j_user,
        password=runtime_cfg.neo4j_password,
    )

    try:
        if cfg.clear_before_insert:
            logger.info("Clearing existing Group C graph before insertion...")
            repo.clear_group_c_graph()

        if cfg.create_indexes:
            repo.create_vector_indexes()

        inserted_syllabus = repo.upsert_syllabus_nodes(syllabus_rows)
        rel_count = repo.connect_syllabus_hierarchy(syllabus_rows)
        inserted_text = repo.upsert_text_snippets(text_rows)
        inserted_code = repo.upsert_code_snippets(code_rows)

        logger.info(
            "Inserted base graph: syllabus_nodes=%d, hierarchy_edges=%d, text=%d, code=%d",
            inserted_syllabus,
            rel_count,
            inserted_text,
            inserted_code,
        )

        if cfg.build_upward_summary:
            summary_stats = build_upward_summaries(
                repo=repo,
                embedding_model=embedding_model,
                token_counter=token_counter,
                cfg=cfg,
                runtime_api_key=runtime_cfg.ark_api_key,
            )
            logger.info("Upward summary completed: %s", summary_stats)
    finally:
        repo.close()


if __name__ == "__main__":
    run_group_c_static_foundation()
