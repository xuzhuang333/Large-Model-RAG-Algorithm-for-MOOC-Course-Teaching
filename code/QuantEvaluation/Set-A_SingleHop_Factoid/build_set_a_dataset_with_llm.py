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

DEFAULT_DATASET_ROOT = (
	r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学"
)
DEFAULT_TEMPLATE_PATH = os.path.join(CURRENT_DIR, "input_template_set_a.json")
DEFAULT_OUTPUT_PATH = os.path.join(CURRENT_DIR, "set_a_auto_generated.json")
DEFAULT_SPLIT = "test"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SetA_Auto_Generator")


def _load_shared_utils_module():
	module_path = os.path.join(ABLATION_DIR, "shared_retrieval_utils.py")
	module_name = "shared_retrieval_utils"
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load shared utils from: {module_path}")
	module = importlib.util.module_from_spec(spec)
	# Register before execution so decorators like @dataclass can resolve module globals.
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
	content: str


@dataclass
class TopicBundle:
	topic_abs: str
	topic_rel: str
	sources: list[SourceMaterial]


def _parse_bool(raw_value: str | None, default: bool) -> bool:
	if raw_value is None:
		return default
	return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_windows_relpath(path_value: str) -> str:
	return path_value.replace("/", "\\")


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


def _find_dirs_with_supported_files(dataset_root: str) -> list[str]:
	directories: list[str] = []
	for dirpath, _, filenames in os.walk(dataset_root):
		has_supported = any(
			os.path.splitext(name)[1].lower() in SUPPORTED_EXTS for name in filenames
		)
		if has_supported:
			directories.append(os.path.abspath(dirpath))
	return sorted(set(directories))


def _select_topic_dirs(dirs_with_files: list[str], leaf_only: bool) -> list[str]:
	if not leaf_only:
		return dirs_with_files

	selected: list[str] = []
	for current in dirs_with_files:
		prefix = current + os.sep
		has_child = any(
			other != current and other.startswith(prefix) for other in dirs_with_files
		)
		if not has_child:
			selected.append(current)
	return selected


def _select_source_files(topic_dir: str, max_files_per_topic: int) -> list[str]:
	files: list[str] = []
	for name in os.listdir(topic_dir):
		abs_path = os.path.join(topic_dir, name)
		if not os.path.isfile(abs_path):
			continue
		ext = os.path.splitext(name)[1].lower()
		if ext in SUPPORTED_EXTS:
			files.append(name)

	files.sort(key=lambda n: (EXT_ORDER.get(os.path.splitext(n)[1].lower(), 99), n.lower()))
	if max_files_per_topic <= 0:
		return files
	return files[:max_files_per_topic]


def _build_topic_bundles(
	dataset_root: str,
	leaf_only: bool,
	max_files_per_topic: int,
	max_chars_per_source: int,
	min_chars_per_source: int,
) -> list[TopicBundle]:
	dirs_with_files = _find_dirs_with_supported_files(dataset_root)
	topic_dirs = _select_topic_dirs(dirs_with_files, leaf_only=leaf_only)

	bundles: list[TopicBundle] = []
	for topic_dir in topic_dirs:
		selected_names = _select_source_files(topic_dir, max_files_per_topic=max_files_per_topic)
		sources: list[SourceMaterial] = []

		for idx, file_name in enumerate(selected_names, start=1):
			abs_path = os.path.join(topic_dir, file_name)
			try:
				text = _read_text_file(abs_path).strip()
			except Exception as exc:
				logger.warning("Skip unreadable file: %s, error=%s", abs_path, exc)
				continue

			if min_chars_per_source > 0 and len(text) < min_chars_per_source:
				continue

			clipped = text if max_chars_per_source <= 0 else text[:max_chars_per_source]
			rel_path = _normalize_windows_relpath(os.path.relpath(abs_path, dataset_root))
			sources.append(SourceMaterial(source_id=idx, relative_path=rel_path, content=clipped))

		if not sources:
			continue

		topic_rel = _normalize_windows_relpath(os.path.relpath(topic_dir, dataset_root))
		bundles.append(TopicBundle(topic_abs=topic_dir, topic_rel=topic_rel, sources=sources))

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


def _normalize_single_item(item: Any, sources_count: int) -> dict[str, Any]:
	if not isinstance(item, dict):
		raise ValueError("Each item (micro/macro) must be a JSON object")

	question = str(item.get("question", "")).strip()
	ground_truth = str(item.get("ground_truth", "")).strip()
	if not question:
		raise ValueError("question is empty")
	if not ground_truth:
		raise ValueError("ground_truth is empty")

	raw_indices = item.get("golden_source_indices", [])
	indices: list[int] = []
	if isinstance(raw_indices, list):
		for value in raw_indices:
			try:
				parsed = int(value)
			except (TypeError, ValueError):
				continue
			if 1 <= parsed <= sources_count and parsed not in indices:
				indices.append(parsed)

	raw_points = item.get("gold_answer_points", [])
	points: list[str] = []
	if isinstance(raw_points, list):
		for point in raw_points:
			text = str(point).strip()
			if text:
				points.append(text)
	points = points[:3]

	return {
		"question": question,
		"ground_truth": ground_truth,
		"golden_source_indices": indices,
		"gold_answer_points": points,
		"difficulty": _normalize_difficulty(item.get("difficulty", "")),
	}


def _build_generation_prompt(topic: TopicBundle) -> str:
	source_blocks = []
	for source in topic.sources:
		source_blocks.append(
			(
				f"[Source {source.source_id}] path={source.relative_path}\n"
				"--- BEGIN CONTENT ---\n"
				f"{source.content}\n"
				"--- END CONTENT ---"
			)
		)

	joined_sources = "\n\n".join(source_blocks)

	return f"""
你是课程测评集构建助手。你将收到同一教学主题目录下的资料片段（md/txt/srt）。

任务：仅基于给定资料，生成 2 条 Set-A（SingleHop Factoid）样本：
1. micro：细粒度事实问答（定义、规则、关键语法点）
2. macro：同一主题内的概括问答，但仍必须是单跳可答

严格约束：
1. 问题与答案必须完全可由给定资料直接支持，禁止外部知识。
2. 每条样本必须是单跳，不要要求跨多个主题推理。
3. 每条样本给出 golden_source_indices（从 source_id 中选择 1~3 个整数）。
4. 每条样本给出 gold_answer_points（最多 3 个简短要点）。
5. difficulty 只能是 easy / medium / hard。
6. 仅输出严格 JSON，不要输出解释，不要输出 markdown 代码块。

输出 JSON 结构必须是：
{{
  "micro": {{
	"question": "...",
	"ground_truth": "...",
	"golden_source_indices": [1],
	"gold_answer_points": ["..."],
	"difficulty": "easy"
  }},
  "macro": {{
	"question": "...",
	"ground_truth": "...",
	"golden_source_indices": [1],
	"gold_answer_points": ["..."],
	"difficulty": "medium"
  }}
}}

Topic directory: {topic.topic_rel}

Available sources:
{joined_sources}
""".strip()


def _build_repair_prompt(previous_output: str) -> str:
	return f"""
请修复下面输出，使其符合指定 JSON schema。

要求：
1. 只能输出一个 JSON 对象。
2. 顶层必须包含 micro 和 macro。
3. 每个对象必须包含 question、ground_truth、golden_source_indices、gold_answer_points、difficulty。
4. difficulty 只能是 easy/medium/hard。
5. 禁止输出任何额外文本。

待修复输出：
{previous_output}
""".strip()


def _request_set_a_pair(
	topic: TopicBundle,
	model_name: str,
	api_base: str,
	api_key: str | None,
	timeout_sec: int,
) -> dict[str, dict[str, Any]]:
	attempts = 2
	previous_output = ""

	for attempt in range(1, attempts + 1):
		prompt = (
			_build_generation_prompt(topic)
			if attempt == 1
			else _build_repair_prompt(previous_output)
		)
		temperature, max_tokens = get_adaptive_generation_params(prompt, task="generate")
		max_tokens = max(420, max_tokens)

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
			micro = _normalize_single_item(parsed.get("micro"), sources_count=len(topic.sources))
			macro = _normalize_single_item(parsed.get("macro"), sources_count=len(topic.sources))
			return {"micro": micro, "macro": macro}
		except Exception as exc:
			logger.warning(
				"Invalid generation for topic=%s at attempt=%d, error=%s",
				topic.topic_rel,
				attempt,
				exc,
			)

	raise RuntimeError(f"Failed to generate valid JSON for topic {topic.topic_rel}")


def _indices_to_paths(indices: list[int], sources: list[SourceMaterial]) -> list[str]:
	if not indices:
		return [sources[0].relative_path]

	picked: list[str] = []
	for idx in indices:
		if 1 <= idx <= len(sources):
			path = sources[idx - 1].relative_path
			if path not in picked:
				picked.append(path)

	return picked if picked else [sources[0].relative_path]


def _build_set_a_id(serial: int) -> str:
	return f"SA{serial:04d}"


def _normalize_record_to_template(
	record: dict[str, Any],
	template_item: dict[str, Any],
) -> dict[str, Any]:
	"""Return a record that strictly follows template keys and fallback defaults."""
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


def _build_set_a_record(
	template_item: dict[str, Any],
	qid: str,
	qtype: str,
	generated_item: dict[str, Any],
	topic: TopicBundle,
	split_default: str,
) -> dict[str, Any]:
	record = copy.deepcopy(template_item)

	record["id"] = qid
	record["eval_set"] = "Set-A"
	record["type"] = qtype
	record["question"] = generated_item["question"]
	record["ground_truth"] = generated_item["ground_truth"]
	record["gold_answer_points"] = generated_item["gold_answer_points"]
	record["golden_sources"] = _indices_to_paths(generated_item["golden_source_indices"], topic.sources)
	record["noise_profile"] = "none"
	record["max_hop"] = 1
	record["split"] = split_default
	record["difficulty"] = generated_item["difficulty"]
	record["notes"] = f"auto_generated_from={topic.topic_rel}"

	# Keep non-SetA fields deterministic and empty to avoid cross-set leakage.
	record["user_id"] = ""
	record["current_turn"] = None
	record["qa_score"] = None
	record["current_struggle"] = None
	record["required_concepts"] = []
	record["supporting_facts"] = []
	record["hard_negative_pairs"] = []
	record["trajectory_id"] = ""
	record["turn_id"] = None
	record["persona"] = ""
	record["expected_route"] = ""
	record["tags"] = []
	record["golden_parent_syllabus_ids"] = []
	record["candidate_reference_materials"] = []

	return _normalize_record_to_template(record, template_item)


def main() -> None:
	runtime_cfg = load_runtime_config(
		default_uri="bolt://localhost:7687",
		default_user="neo4j",
		default_password="12345678",
	)

	dataset_root = os.getenv("DATASET_ROOT", DEFAULT_DATASET_ROOT)
	template_path = os.getenv("SET_A_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH)
	output_path = os.getenv("SET_A_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)
	model_name = os.getenv("SET_A_GENERATOR_MODEL", DEFAULT_DOUBAO_ENDPOINT)

	leaf_only = _parse_bool(os.getenv("SET_A_LEAF_DIR_ONLY"), default=True)
	# <= 0 means "use all supported files under a leaf topic dir"
	max_files_per_topic = int(os.getenv("SET_A_MAX_FILES_PER_TOPIC", "0"))
	# <= 0 means "no truncation" so full md/txt/srt content is sent
	max_chars_per_source = int(os.getenv("SET_A_MAX_CHARS_PER_SOURCE", "0"))
	# <= 0 means no minimal-length filtering
	min_chars_per_source = int(os.getenv("SET_A_MIN_CHARS_PER_SOURCE", "1"))
	split_default = os.getenv("SET_A_SPLIT", DEFAULT_SPLIT).strip() or DEFAULT_SPLIT
	start_index = max(1, int(os.getenv("SET_A_START_INDEX", "1")))
	timeout_sec = max(30, int(os.getenv("SET_A_TIMEOUT_SEC", "180")))

	max_topics_raw = (os.getenv("SET_A_MAX_TOPICS") or "").strip()
	max_topics = max(1, int(max_topics_raw)) if max_topics_raw else None

	if not os.path.isdir(dataset_root):
		raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

	template_item = _load_template_item(template_path)
	topic_bundles = _build_topic_bundles(
		dataset_root=dataset_root,
		leaf_only=leaf_only,
		max_files_per_topic=max_files_per_topic,
		max_chars_per_source=max_chars_per_source,
		min_chars_per_source=min_chars_per_source,
	)

	if max_topics is not None:
		topic_bundles = topic_bundles[:max_topics]

	if not topic_bundles:
		logger.warning("No eligible topic bundles found. Nothing generated.")
		return

	logger.info(
		"Set-A generation started: topics=%d, model=%s, leaf_only=%s",
		len(topic_bundles),
		model_name,
		leaf_only,
	)

	output_records: list[dict[str, Any]] = []
	failed_topics: list[str] = []
	serial = start_index

	for idx, topic in enumerate(topic_bundles, start=1):
		logger.info(
			"Generating topic %d/%d: %s (sources=%d)",
			idx,
			len(topic_bundles),
			topic.topic_rel,
			len(topic.sources),
		)

		try:
			generated_pair = _request_set_a_pair(
				topic=topic,
				model_name=model_name,
				api_base=runtime_cfg.llm_api_base,
				api_key=runtime_cfg.ark_api_key,
				timeout_sec=timeout_sec,
			)
		except Exception as exc:
			logger.error("Skip topic due to generation failure: %s, error=%s", topic.topic_rel, exc)
			failed_topics.append(topic.topic_rel)
			continue

		micro_record = _build_set_a_record(
			template_item=template_item,
			qid=_build_set_a_id(serial),
			qtype="micro",
			generated_item=generated_pair["micro"],
			topic=topic,
			split_default=split_default,
		)
		serial += 1

		macro_record = _build_set_a_record(
			template_item=template_item,
			qid=_build_set_a_id(serial),
			qtype="macro",
			generated_item=generated_pair["macro"],
			topic=topic,
			split_default=split_default,
		)
		serial += 1

		output_records.append(micro_record)
		output_records.append(macro_record)

	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(output_records, f, ensure_ascii=False, indent=2)

	logger.info(
		"Set-A generation finished: output=%s, samples=%d, topics_ok=%d, topics_failed=%d",
		output_path,
		len(output_records),
		len(output_records) // 2,
		len(failed_topics),
	)

	if failed_topics:
		failed_path = f"{output_path}.failed_topics.txt"
		with open(failed_path, "w", encoding="utf-8") as f:
			for topic in failed_topics:
				f.write(topic + "\n")
		logger.warning("Failed topic list written to: %s", failed_path)


if __name__ == "__main__":
	main()
