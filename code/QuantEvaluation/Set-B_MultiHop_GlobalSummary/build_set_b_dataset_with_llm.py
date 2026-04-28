from __future__ import annotations

import copy
import importlib.util
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QUANT_ROOT = os.path.dirname(CURRENT_DIR)
CODE_ROOT = os.path.dirname(QUANT_ROOT)
ABLATION_DIR = os.path.join(CODE_ROOT, "Retrieval Ablation Experiment")

SUPPORTED_EXTS = {".md", ".txt", ".srt"}
EXT_ORDER = {".md": 0, ".txt": 1, ".srt": 2}
SRT_TS_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$"
)
SECTION_CODE_PATTERN = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?)")

DEFAULT_DATASET_ROOT = (
    r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学"
)
DEFAULT_TEMPLATE_PATH = os.path.join(CURRENT_DIR, "input_template_set_b.json")
DEFAULT_OUTPUT_PATH = os.path.join(CURRENT_DIR, "set_b_auto_generated.json")
DEFAULT_SPLIT = "test"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SetB_Auto_Generator")


def _load_shared_utils_module():
    module_path = os.path.join(ABLATION_DIR, "shared_retrieval_utils.py")
    module_name = "shared_retrieval_utils"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load shared utils from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_shared_utils = _load_shared_utils_module()
DEFAULT_DOUBAO_ENDPOINT = _shared_utils.DEFAULT_DOUBAO_ENDPOINT
ark_chat_completion = _shared_utils.ark_chat_completion
get_adaptive_generation_params = _shared_utils.get_adaptive_generation_params
load_runtime_config = _shared_utils.load_runtime_config


@dataclass
class SourceMaterial:
    source_id: int
    relative_path: str
    subtopic_code: str
    subtopic_title: str
    content: str


@dataclass
class SubtopicBundle:
    code: str
    title: str
    abs_path: str
    rel_path: str
    sources: list[SourceMaterial]


@dataclass
class ParentTopicBundle:
    parent_code: str
    parent_title: str
    parent_abs: str
    parent_rel: str
    subtopics: list[SubtopicBundle]


@dataclass
class NormalizedSetBItem:
    question: str
    ground_truth: str
    required_concepts: list[str]
    supporting_facts: list[str]
    gold_answer_points: list[str]
    golden_source_indices: list[int]
    golden_parent_syllabus_ids: list[str]
    max_hop: int
    difficulty: str


def _parse_bool(raw_value: str | None, default: bool) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_windows_relpath(path_value: str) -> str:
    return path_value.replace("/", "\\")


def _clean_srt_text(raw_text: str) -> str:
    cleaned_lines: list[str] = []
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


def _read_text_file(file_path: str) -> str:
    encodings = ("utf-8", "utf-8-sig", "gb18030")
    raw_text = ""

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                raw_text = f.read()
            break
        except UnicodeDecodeError:
            continue

    if not raw_text:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".srt":
        return _clean_srt_text(raw_text)
    return raw_text


def _load_template_item(template_path: str) -> dict[str, Any]:
    with open(template_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Template must be a non-empty array: {template_path}")
    if not isinstance(payload[0], dict):
        raise ValueError(f"Template first item must be an object: {template_path}")
    return payload[0]


def _extract_section_code(name: str) -> str | None:
    match = SECTION_CODE_PATTERN.match(name or "")
    if not match:
        return None
    return match.group(1)


def _split_code_parts(code: str | None) -> tuple[int, ...]:
    if not code:
        return tuple()
    try:
        return tuple(int(x) for x in code.split("."))
    except Exception:
        return tuple()


def _extract_title(name: str, code: str | None) -> str:
    if not code:
        return name.strip()
    stripped = name.strip()
    rest = stripped[len(code) :].strip()
    return rest if rest else stripped


def _find_dirs_with_supported_files(dataset_root: str) -> list[str]:
    directories: list[str] = []
    for dirpath, _, filenames in os.walk(dataset_root):
        has_supported = any(
            os.path.splitext(filename)[1].lower() in SUPPORTED_EXTS
            for filename in filenames
        )
        if has_supported:
            directories.append(os.path.abspath(dirpath))
    return sorted(set(directories))


def _sort_key_for_code(code: str) -> tuple[int, ...]:
    parsed = _split_code_parts(code)
    return parsed if parsed else (10**9,)


def _collect_supported_files_recursive(
    subtopic_dir: str,
    max_files_per_subtopic: int,
) -> list[str]:
    files: list[str] = []

    for dirpath, _, filenames in os.walk(subtopic_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            files.append(os.path.join(dirpath, name))

    files.sort(
        key=lambda p: (
            EXT_ORDER.get(os.path.splitext(p)[1].lower(), 99),
            p.lower(),
        )
    )

    if max_files_per_subtopic > 0:
        return files[:max_files_per_subtopic]
    return files


def _build_parent_topic_bundles(
    dataset_root: str,
    max_files_per_subtopic: int,
    max_chars_per_source: int,
    min_chars_per_source: int,
) -> list[ParentTopicBundle]:
    dirs_with_files = _find_dirs_with_supported_files(dataset_root)

    parent_dirs: list[str] = []
    for path in dirs_with_files:
        code = _extract_section_code(os.path.basename(path))
        if len(_split_code_parts(code)) == 2:
            parent_dirs.append(path)

    bundles: list[ParentTopicBundle] = []
    for parent_dir in parent_dirs:
        parent_name = os.path.basename(parent_dir)
        parent_code = _extract_section_code(parent_name)
        if not parent_code:
            continue

        prefix = parent_dir + os.sep
        subtopic_dirs: list[str] = []
        for candidate in dirs_with_files:
            if not candidate.startswith(prefix):
                continue
            code = _extract_section_code(os.path.basename(candidate))
            parts = _split_code_parts(code)
            if len(parts) != 3:
                continue
            # Ensure x.x.x really belongs to current x.x parent.
            if not code.startswith(parent_code + "."):
                continue
            subtopic_dirs.append(candidate)

        subtopic_dirs = sorted(set(subtopic_dirs), key=lambda d: _sort_key_for_code(_extract_section_code(os.path.basename(d)) or ""))
        if len(subtopic_dirs) < 2:
            continue

        raw_subtopics: list[tuple[str, str, str, list[SourceMaterial]]] = []
        for subtopic_dir in subtopic_dirs:
            subtopic_name = os.path.basename(subtopic_dir)
            subtopic_code = _extract_section_code(subtopic_name)
            if not subtopic_code:
                continue

            subtopic_title = _extract_title(subtopic_name, subtopic_code)
            file_paths = _collect_supported_files_recursive(
                subtopic_dir,
                max_files_per_subtopic=max_files_per_subtopic,
            )

            sources: list[SourceMaterial] = []
            for file_path in file_paths:
                try:
                    text = _read_text_file(file_path).strip()
                except Exception as exc:
                    logger.warning("Skip unreadable file: %s, error=%s", file_path, exc)
                    continue

                if min_chars_per_source > 0 and len(text) < min_chars_per_source:
                    continue

                clipped = text if max_chars_per_source <= 0 else text[:max_chars_per_source]
                rel_path = _normalize_windows_relpath(os.path.relpath(file_path, dataset_root))
                sources.append(
                    SourceMaterial(
                        source_id=0,
                        relative_path=rel_path,
                        subtopic_code=subtopic_code,
                        subtopic_title=subtopic_title,
                        content=clipped,
                    )
                )

            if sources:
                rel_subtopic = _normalize_windows_relpath(os.path.relpath(subtopic_dir, dataset_root))
                raw_subtopics.append((subtopic_code, subtopic_title, rel_subtopic, sources))

        if len(raw_subtopics) < 2:
            continue

        # Assign source_id globally within one parent topic bundle.
        assigned_subtopics: list[SubtopicBundle] = []
        next_source_id = 1
        for subtopic_code, subtopic_title, rel_subtopic, sources in raw_subtopics:
            assigned_sources: list[SourceMaterial] = []
            for src in sources:
                assigned_sources.append(
                    SourceMaterial(
                        source_id=next_source_id,
                        relative_path=src.relative_path,
                        subtopic_code=src.subtopic_code,
                        subtopic_title=src.subtopic_title,
                        content=src.content,
                    )
                )
                next_source_id += 1

            assigned_subtopics.append(
                SubtopicBundle(
                    code=subtopic_code,
                    title=subtopic_title,
                    abs_path=os.path.join(dataset_root, rel_subtopic),
                    rel_path=rel_subtopic,
                    sources=assigned_sources,
                )
            )

        parent_rel = _normalize_windows_relpath(os.path.relpath(parent_dir, dataset_root))
        bundles.append(
            ParentTopicBundle(
                parent_code=parent_code,
                parent_title=_extract_title(parent_name, parent_code),
                parent_abs=parent_dir,
                parent_rel=parent_rel,
                subtopics=assigned_subtopics,
            )
        )

    return bundles


def _extract_first_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM output")

    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON object found in LLM output")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON must be an object")
    return parsed


def _normalize_difficulty(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"easy", "medium", "hard"}:
        return normalized

    zh_map = {
        "简单": "easy",
        "中等": "medium",
        "困难": "hard",
        "较难": "hard",
    }
    return zh_map.get(normalized, "medium")


def _sanitize_text_list(raw_value: Any, max_items: int, min_items: int, field_name: str) -> list[str]:
    if not isinstance(raw_value, list):
        raise ValueError(f"{field_name} must be a list")

    values: list[str] = []
    for item in raw_value:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
        if len(values) >= max_items:
            break

    if len(values) < min_items:
        raise ValueError(f"{field_name} must contain at least {min_items} items")

    return values


def _build_source_lookup(bundle: ParentTopicBundle) -> dict[int, SourceMaterial]:
    lookup: dict[int, SourceMaterial] = {}
    for subtopic in bundle.subtopics:
        for source in subtopic.sources:
            lookup[source.source_id] = source
    return lookup


def _normalize_set_b_item(item: Any, bundle: ParentTopicBundle) -> NormalizedSetBItem:
    if not isinstance(item, dict):
        raise ValueError("Set-B generated payload must be a JSON object")

    question = str(item.get("question", "")).strip()
    ground_truth = str(item.get("ground_truth", "")).strip()
    if not question:
        raise ValueError("question is empty")
    if not ground_truth:
        raise ValueError("ground_truth is empty")

    required_concepts = _sanitize_text_list(
        item.get("required_concepts", []),
        max_items=8,
        min_items=2,
        field_name="required_concepts",
    )
    supporting_facts = _sanitize_text_list(
        item.get("supporting_facts", []),
        max_items=8,
        min_items=2,
        field_name="supporting_facts",
    )
    gold_answer_points = _sanitize_text_list(
        item.get("gold_answer_points", []),
        max_items=6,
        min_items=2,
        field_name="gold_answer_points",
    )

    source_lookup = _build_source_lookup(bundle)
    raw_indices = item.get("golden_source_indices", [])
    if not isinstance(raw_indices, list):
        raise ValueError("golden_source_indices must be a list")

    indices: list[int] = []
    for value in raw_indices:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed in source_lookup and parsed not in indices:
            indices.append(parsed)
    if len(indices) < 2:
        raise ValueError("golden_source_indices must include at least 2 valid sources")

    used_subtopics: list[str] = []
    for idx in indices:
        code = source_lookup[idx].subtopic_code
        if code not in used_subtopics:
            used_subtopics.append(code)
    if len(used_subtopics) < 2:
        raise ValueError("golden_source_indices must cover at least 2 different x.x.x subtopics")

    raw_hop = item.get("max_hop", 2)
    try:
        max_hop = int(raw_hop)
    except (TypeError, ValueError):
        max_hop = 2
    max_hop = max(2, min(4, max_hop))

    difficulty = _normalize_difficulty(item.get("difficulty", ""))

    return NormalizedSetBItem(
        question=question,
        ground_truth=ground_truth,
        required_concepts=required_concepts,
        supporting_facts=supporting_facts,
        gold_answer_points=gold_answer_points,
        golden_source_indices=indices,
        golden_parent_syllabus_ids=used_subtopics,
        max_hop=max_hop,
        difficulty=difficulty,
    )


def _flatten_sources(bundle: ParentTopicBundle) -> list[SourceMaterial]:
    flattened: list[SourceMaterial] = []
    for subtopic in bundle.subtopics:
        flattened.extend(subtopic.sources)
    flattened.sort(key=lambda s: s.source_id)
    return flattened


def _build_generation_prompt(bundle: ParentTopicBundle) -> str:
    subtopic_lines: list[str] = []
    for subtopic in sorted(bundle.subtopics, key=lambda s: _sort_key_for_code(s.code)):
        subtopic_lines.append(
            f"- subtopic_id={subtopic.code}, subtopic_title={subtopic.title}, rel_path={subtopic.rel_path}"
        )

    source_blocks: list[str] = []
    for source in _flatten_sources(bundle):
        source_blocks.append(
            (
                f"[Source {source.source_id}] path={source.relative_path}; "
                f"subtopic_id={source.subtopic_code}; subtopic_title={source.subtopic_title}\n"
                "--- BEGIN CONTENT ---\n"
                f"{source.content}\n"
                "--- END CONTENT ---"
            )
        )

    return f"""
你是课程量化测评集构建助手。你将收到同一父主题（x.x）下多个子主题（x.x.x）的资料片段（md/txt/srt）。

任务：仅基于给定资料，生成 1 条 Set-B（MultiHop GlobalSummary）样本。
样本类型固定为 macro，且必须是“跨至少两个不同 x.x.x 子主题”的联合问题与联合答案。

父主题：{bundle.parent_code} {bundle.parent_title}
子主题关系：以下子主题都隶属于同一父主题，按编号形成并列且可联合推理的关系。
{subtopic_lines and chr(10).join(subtopic_lines) or '- 无'}

严格约束：
1. 问题与答案必须完全可由给定资料支持，禁止外部知识。
2. 必须跨至少 2 个不同 x.x.x 子主题，禁止单子主题问题。
3. required_concepts 至少 2 项，supporting_facts 至少 2 项，gold_answer_points 至少 2 项。
4. golden_source_indices 必须引用至少 2 个 source，且这些 source 来自至少 2 个不同子主题。
5. max_hop 必须是 2~4 的整数。
6. difficulty 只能是 easy / medium / hard。
7. 仅输出严格 JSON，不要输出解释，不要输出 markdown 代码块。

输出 JSON 结构必须是：
{{
  "question": "...",
  "ground_truth": "...",
  "required_concepts": ["...", "..."],
  "supporting_facts": ["...", "..."],
  "gold_answer_points": ["...", "..."],
  "golden_source_indices": [1, 2],
  "max_hop": 3,
  "difficulty": "medium"
}}

可用资料如下：
{chr(10).join(source_blocks)}
""".strip()


def _build_repair_prompt(previous_output: str) -> str:
    return f"""
请修复下面输出，使其符合 Set-B 的 JSON schema 和约束。

要求：
1. 只能输出一个 JSON 对象。
2. 必须包含字段：question、ground_truth、required_concepts、supporting_facts、gold_answer_points、golden_source_indices、max_hop、difficulty。
3. required_concepts、supporting_facts、gold_answer_points 都至少 2 项。
4. golden_source_indices 至少 2 个整数。
5. max_hop 必须是 2~4 的整数。
6. difficulty 只能是 easy/medium/hard。
7. 禁止输出任何额外文本。

待修复输出：
{previous_output}
""".strip()


def _request_set_b_item(
    bundle: ParentTopicBundle,
    model_name: str,
    api_base: str,
    api_key: str | None,
    timeout_sec: int,
) -> NormalizedSetBItem:
    attempts = 3
    previous_output = ""

    for attempt in range(1, attempts + 1):
        prompt = _build_generation_prompt(bundle) if attempt == 1 else _build_repair_prompt(previous_output)

        temperature, max_tokens = get_adaptive_generation_params(prompt, task="generate")
        max_tokens = max(520, max_tokens)

        llm_text = ark_chat_completion(
            model=model_name,
            prompt=prompt,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        previous_output = llm_text

        try:
            parsed = _extract_first_json_object(llm_text)
            return _normalize_set_b_item(parsed, bundle=bundle)
        except Exception as exc:
            logger.warning(
                "Invalid Set-B generation for parent=%s at attempt=%d, error=%s",
                bundle.parent_rel,
                attempt,
                exc,
            )

    raise RuntimeError(f"Failed to generate valid Set-B item for parent topic {bundle.parent_rel}")


def _normalize_record_to_template(
    record: dict[str, Any],
    template_item: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, default_value in template_item.items():
        if key in record:
            value = record[key]
        else:
            value = copy.deepcopy(default_value)

        if isinstance(default_value, list) and not isinstance(value, list):
            normalized[key] = copy.deepcopy(default_value)
            continue

        if default_value is not None and value is None:
            normalized[key] = copy.deepcopy(default_value)
            continue

        normalized[key] = value

    return normalized


def _build_set_b_id(serial: int) -> str:
    return f"SB{serial:04d}"


def _build_set_b_record(
    template_item: dict[str, Any],
    qid: str,
    generated_item: NormalizedSetBItem,
    bundle: ParentTopicBundle,
) -> dict[str, Any]:
    source_lookup = _build_source_lookup(bundle)
    golden_sources: list[str] = []
    for idx in generated_item.golden_source_indices:
        source = source_lookup.get(idx)
        if source and source.relative_path not in golden_sources:
            golden_sources.append(source.relative_path)

    record = copy.deepcopy(template_item)

    record["id"] = qid
    record["eval_set"] = "Set-B"
    record["type"] = "macro"
    record["question"] = generated_item.question
    record["ground_truth"] = generated_item.ground_truth
    record["required_concepts"] = generated_item.required_concepts
    record["supporting_facts"] = generated_item.supporting_facts
    record["gold_answer_points"] = generated_item.gold_answer_points
    record["golden_sources"] = golden_sources
    record["golden_parent_syllabus_ids"] = generated_item.golden_parent_syllabus_ids
    record["max_hop"] = generated_item.max_hop
    record["difficulty"] = generated_item.difficulty
    record["noise_profile"] = "none"
    record["notes"] = (
        f"auto_generated_from={bundle.parent_rel};"
        f"subtopics={','.join(generated_item.golden_parent_syllabus_ids)}"
    )

    # Non Set-B fields kept deterministic and neutral.
    record["user_id"] = ""
    record["current_turn"] = None
    record["qa_score"] = None
    record["current_struggle"] = None
    record["hard_negative_pairs"] = []
    record["trajectory_id"] = ""
    record["turn_id"] = None
    record["persona"] = ""
    record["expected_route"] = ""
    record["tags"] = []
    record["candidate_reference_materials"] = []

    return _normalize_record_to_template(record, template_item)


def main() -> None:
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="YOUR VALUE",
    )

    dataset_root = os.getenv("DATASET_ROOT", DEFAULT_DATASET_ROOT)
    template_path = os.getenv("SET_B_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH)
    output_path = os.getenv("SET_B_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)
    model_name = os.getenv("SET_B_GENERATOR_MODEL", DEFAULT_DOUBAO_ENDPOINT)

    max_files_per_subtopic = int(os.getenv("SET_B_MAX_FILES_PER_SUBTOPIC", "0"))
    max_chars_per_source = int(os.getenv("SET_B_MAX_CHARS_PER_SOURCE", "0"))
    min_chars_per_source = max(1, int(os.getenv("SET_B_MIN_CHARS_PER_SOURCE", "80")))
    split_default = os.getenv("SET_B_SPLIT", DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
    start_index = max(1, int(os.getenv("SET_B_START_INDEX", "1")))
    timeout_sec = max(30, int(os.getenv("SET_B_TIMEOUT_SEC", "180")))

    max_topics_raw = (os.getenv("SET_B_MAX_TOPICS") or "").strip()
    max_topics = max(1, int(max_topics_raw)) if max_topics_raw else None

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    template_item = _load_template_item(template_path)
    bundles = _build_parent_topic_bundles(
        dataset_root=dataset_root,
        max_files_per_subtopic=max_files_per_subtopic,
        max_chars_per_source=max_chars_per_source,
        min_chars_per_source=min_chars_per_source,
    )

    if max_topics is not None:
        bundles = bundles[:max_topics]

    if not bundles:
        logger.warning("No eligible Set-B parent-topic bundles found. Nothing generated.")
        return

    logger.info(
        "Set-B generation started: parent_topics=%d, model=%s",
        len(bundles),
        model_name,
    )

    output_records: list[dict[str, Any]] = []
    failed_topics: list[str] = []
    serial = start_index

    for idx, bundle in enumerate(bundles, start=1):
        logger.info(
            "Generating Set-B item %d/%d: parent=%s, subtopics=%d",
            idx,
            len(bundles),
            bundle.parent_rel,
            len(bundle.subtopics),
        )

        try:
            generated_item = _request_set_b_item(
                bundle=bundle,
                model_name=model_name,
                api_base=runtime_cfg.llm_api_base,
                api_key=runtime_cfg.ark_api_key,
                timeout_sec=timeout_sec,
            )

            record = _build_set_b_record(
                template_item=template_item,
                qid=_build_set_b_id(serial),
                generated_item=generated_item,
                bundle=bundle,
            )
            record["split"] = split_default
            output_records.append(record)
            serial += 1
        except Exception as exc:
            logger.error("Skip parent topic due to generation failure: %s, error=%s", bundle.parent_rel, exc)
            failed_topics.append(f"{bundle.parent_rel}\t{exc}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    logger.info(
        "Set-B generation finished: output=%s, samples=%d, topics_failed=%d",
        output_path,
        len(output_records),
        len(failed_topics),
    )

    if failed_topics:
        failed_path = f"{output_path}.failed_topics.txt"
        with open(failed_path, "w", encoding="utf-8") as f:
            for line in failed_topics:
                f.write(line + "\n")
        logger.warning("Failed topic list written to: %s", failed_path)


if __name__ == "__main__":
    main()
