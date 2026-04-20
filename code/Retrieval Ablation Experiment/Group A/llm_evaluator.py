import json
import logging
import os
import re
import sys

import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DOUBAO_ENDPOINT,
    ark_chat_completion,
    get_adaptive_generation_params,
)

logger = logging.getLogger("Llama3.1_Jury")


class RAGEvaluator:
    def __init__(self, model_name=DEFAULT_DOUBAO_ENDPOINT, api_url=DEFAULT_ARK_API_BASE, api_key=None):
        self.model_name = model_name
        # 兼容旧参数名 api_url，当前语义是 API Base
        self.api_url = api_url
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        logger.info(f"Initialized LLM-as-a-Judge with model: {self.model_name}")

    def _parse_json_scores(self, text):
        """从模型返回文本中提取首个 JSON 对象，避免被解释性文本干扰。"""
        raw = (text or "").strip()
        if not raw:
            raise ValueError("Empty evaluator output")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("No JSON object found in evaluator output")
        return json.loads(match.group(0))

    def evaluate(self, query, context, answer, retrieved_sources=None, golden_sources=None):
        """
        Evaluates the RAG pipeline output across 5 dimensions (Score 1-5).
        Returns a JSON object containing the metrics.
        """
        retrieved_sources = retrieved_sources or []
        golden_sources = golden_sources or []
        source_alignment_rule = (
            "若 [Golden_Sources] 为空，请将 Source_Alignment 输出为 -1。"
            "否则请按检索来源与黄金来源的一致性评分（1-5），并严格使用以下匹配规则："
            "1) 路径标准化后完全相同，视为命中；"
            "2) 同一路径仅扩展名不同（.md/.txt/.srt）也视为命中；"
            "3) 仅当黄金来源是 summary（包含 '#summary' 或 '单元小结'）时，"
            "允许同单元号匹配（如 3.2.*）；"
            "4) 若黄金来源不是 summary，仅同单元号不算命中；"
            "5) 请尽量按一对一匹配统计命中后再做整体评分。"
        )

        prompt = f"""
        你是一个公正的评判员，负责评估检索增强生成（RAG）系统的性能。
        你将获得一个问题、检索到的上下文、生成的答案、检索来源列表、黄金来源列表。
        请对以下五个指标进行评分：
        - 前四项评分范围为 1 到 5 分（1 分最差，5 分最佳）
        - Source_Alignment 评分规则见下方

        1. Context_Recall: 上下文是否包含回答问题所需的全部必要信息?
        2. Context_Precision: 上下文是否与问题高度相关，且没有不必要的噪声?
        3. Faithfulness: 答案是否严格基于上下文，没有幻觉（即没有编造上下文不存在的内容）?
        4. Answer_Relevance: 答案是否直接且有效地解决了问题?
        5. Source_Alignment: 检索来源与黄金来源的一致程度。
           {source_alignment_rule}
              评分锚点：5=几乎完全一致；4=大部分一致；3=部分一致；2=少量一致；1=基本不一致。

        [Question]: {query}
        [Context]: {context}
        [Answer]: {answer}
        [Retrieved_Sources]: {retrieved_sources}
        [Golden_Sources]: {golden_sources}

        Output your evaluation strictly in the following JSON format without any other text:
        {{"Context_Recall": <score>, "Context_Precision": <score>, "Faithfulness": <score>, "Answer_Relevance": <score>, "Source_Alignment": <score_or_-1>}}
        """

        try:
            temperature, max_tokens = get_adaptive_generation_params(prompt, task="evaluate")
            result_text = ark_chat_completion(
                model=self.model_name,
                prompt=prompt,
                api_base=self.api_url,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=180,
            )
            scores = self._parse_json_scores(result_text)
            # 兜底填充：保证返回字段完整，便于后续统计
            scores.setdefault("Context_Recall", 0)
            scores.setdefault("Context_Precision", 0)
            scores.setdefault("Faithfulness", 0)
            scores.setdefault("Answer_Relevance", 0)
            scores.setdefault("Source_Alignment", -1 if not golden_sources else 0)
            logger.info(f"Evaluation completed: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Evaluation failed or invalid JSON: {e}")
            return {
                "Context_Recall": 0,
                "Context_Precision": 0,
                "Faithfulness": 0,
                "Answer_Relevance": 0,
                "Source_Alignment": -1 if not golden_sources else 0,
            }