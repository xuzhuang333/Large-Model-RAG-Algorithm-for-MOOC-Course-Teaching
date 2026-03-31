import json
import logging
import os
import re
import shutil
from datetime import datetime

import requests

# 一次性脚本日志：便于记录每条样本的处理与异常
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Fill_Ground_Truth_Once")


def try_load_json_with_repair(json_path: str):
    """
    先按标准 JSON 解析；若失败，则修复常见尾逗号后再解析。
    返回 (data, repaired_flag)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        return json.loads(raw), False
    except json.JSONDecodeError as exc:
        logger.warning("Standard JSON parse failed: %s", exc)

    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(repaired), True
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parsing still failed after repair: {exc}") from exc


def make_backup(file_path: str) -> str:
    """写回前生成备份，避免误覆盖。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak_{timestamp}"
    shutil.copyfile(file_path, backup_path)
    return backup_path


def clean_srt_text(raw_text: str) -> str:
    """清理 SRT 中的序号和时间戳，减少噪音。"""
    lines = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+$", s):
            continue
        if "-->" in s:
            continue
        lines.append(s)
    return "\n".join(lines)


def read_text_file_with_fallback(path: str) -> str:
    """按常见编码回退读取文本。"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def resolve_source_path(source: str, source_root: str, project_root: str) -> str | None:
    """
    解析 golden_sources 路径：
    1) 绝对路径
    2) 相对 source_root
    3) 相对 project_root（兜底）
    """
    if not source:
        return None

    normalized = source.replace("/", os.sep).replace("\\", os.sep)

    if os.path.isabs(normalized) and os.path.exists(normalized):
        return normalized

    candidate1 = os.path.join(source_root, normalized)
    if os.path.exists(candidate1):
        return candidate1

    candidate2 = os.path.join(project_root, normalized)
    if os.path.exists(candidate2):
        return candidate2

    return None


def build_official_material_text(item: dict, source_root: str, project_root: str) -> str:
    """读取并拼接该问题对应的 golden_sources 内容。"""
    blocks = []
    missing = []

    for src in item.get("golden_sources", []):
        real_path = resolve_source_path(src, source_root, project_root)
        if not real_path:
            missing.append(src)
            continue

        text = read_text_file_with_fallback(real_path)
        if real_path.lower().endswith(".srt"):
            text = clean_srt_text(text)

        blocks.append(f"[Source: {src}]\n{text}")

    if missing:
        logger.warning("Item %s has missing sources: %s", item.get("id"), missing)

    return "\n\n".join(blocks)


def extract_ground_truth(question: str, official_material: str, model: str, api_url: str) -> str:
    """调用本地 Llama 3.1（Ollama）提炼 ground_truth。"""
    if not official_material.strip():
        return "- 教材中未提及"

    prompt = f"""
你是一个冷酷、客观的数据提取机器。请根据以下【官方教材内容】，为【学生问题】提取标准答案。

绝对服从以下规则：
1. 必须且只能以“信息点（Bullet Points）”的形式输出，例如：“- 知识点A； - 知识点B”。
2. 剥离一切主观语气、连接词、过渡句和解释性废话。
3. 只保留回答问题所必需的最核心名词、步骤或结论。
4. 答案必须100%忠实于教材内容。如果教材未提及，只输出“- 教材中未提及”。

【学生问题】:
{question}

【官方教材内容】:
{official_material}

请输出：
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=180)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text if text else "- 教材中未提及"
    except Exception as exc:
        logger.error("LLM extraction failed: %s", exc)
        return "- 教材中未提及"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_json = os.getenv("ANNOTATE_JSON_PATH", os.path.join(script_dir, "qa_dataset_to_annotate.json"))
    output_json = os.getenv("ANNOTATE_OUTPUT_PATH", input_json)

    source_root = os.getenv(
        "GOLDEN_SOURCE_ROOT",
        r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学",
    )

    api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    model = os.getenv("JUDGE_MODEL", "llama3.1")
    # 一次性批处理脚本默认“覆盖更新”，避免 ground_truth 非空时被静默跳过
    # 若希望保留已有内容，可设置 SKIP_NONEMPTY_GROUND_TRUTH=true
    skip_nonempty = os.getenv("SKIP_NONEMPTY_GROUND_TRUTH", "false").strip().lower() == "true"

    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    data, repaired = try_load_json_with_repair(input_json)
    if repaired:
        logger.info("Input JSON repaired in-memory before processing.")

    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list.")

    backup_path = make_backup(input_json)
    logger.info("Backup created: %s", backup_path)

    for item in data:
        qid = item.get("id", "UNKNOWN_ID")
        question = item.get("question", "")
        if not question:
            logger.warning("Skipping item %s: empty question", qid)
            continue

        # 兼容 ground_truth 不是字符串的情况，避免因类型异常导致流程中断
        raw_ground_truth = item.get("ground_truth", "")
        existing_gt = raw_ground_truth.strip() if isinstance(raw_ground_truth, str) else str(raw_ground_truth).strip()
        if existing_gt and skip_nonempty:
            logger.info("Skipping item %s: ground_truth already exists (SKIP_NONEMPTY_GROUND_TRUTH=true)", qid)
            continue

        official_material = build_official_material_text(item, source_root, project_root)
        ground_truth = extract_ground_truth(question, official_material, model=model, api_url=api_url)

        item["ground_truth"] = ground_truth
        logger.info("Updated item %s ground_truth", qid)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info("Done. Ground truth written to: %s", output_json)


if __name__ == "__main__":
    main()
