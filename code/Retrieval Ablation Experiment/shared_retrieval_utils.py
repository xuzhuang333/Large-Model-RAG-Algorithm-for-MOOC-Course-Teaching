import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger("Shared_Retrieval_Utils")

DEFAULT_ARK_API_BASE = "YOUR LLM VALUE"
DEFAULT_DEEPSEEK_ENDPOINT = "YOUR LLM VALUE"
DEFAULT_DOUBAO_ENDPOINT = "YOUR LLM VALUE"


@dataclass
class RewriteConfig:
    group_name: str
    rewrite_n: int = 3
    llm_model: str = DEFAULT_DEEPSEEK_ENDPOINT
    api_base: str = DEFAULT_ARK_API_BASE
    api_key: str | None = None
    timeout_sec: int = 90
    temperature: float | None = None
    max_tokens: int | None = None


def get_adaptive_generation_params(prompt: str, task: str) -> tuple[float, int]:
    """
    基于提示词长度做保守的自适应参数控制，优先保证可跑批和 token 成本可控。
    task: rewrite | generate | evaluate
    """
    prompt_chars = len(prompt)

    if task == "rewrite":
        # 改写任务输出很短，严格限制输出 token，减少无效开销
        temperature = 0.2
        max_tokens = max(48, min(128, 48 + prompt_chars // 120))
    elif task == "evaluate":
        # 评测任务只需结构化评分，输出应短且稳定
        temperature = 0.1
        max_tokens = max(120, min(220, 120 + prompt_chars // 220))
    else:
        # 生成任务需要一定表达空间，但仍做上限保护
        temperature = 0.35
        max_tokens = max(180, min(520, 180 + prompt_chars // 55))

    return temperature, max_tokens


def ark_chat_completion(
    model: str,
    prompt: str,
    api_base: str,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_sec: int = 120,
) -> str:
    """
    统一封装火山方舟 OpenAI 兼容接口调用。
    仅走云端 API，不保留本地 LLM 逻辑。
    """
    key = api_key or os.getenv("ARK_API_KEY")
    if not key:
        raise RuntimeError("Missing ARK_API_KEY in environment.")

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices") or []
    if choices and choices[0].get("message"):
        content = choices[0]["message"].get("content", "")
        return content.strip() if isinstance(content, str) else str(content).strip()

    # 部分兼容返回的兜底字段
    fallback = data.get("output_text") or data.get("response") or ""
    return fallback.strip() if isinstance(fallback, str) else str(fallback).strip()


class QueryRewriter:
    """Reusable query rewrite utility for A/B/C retrievers."""

    def __init__(self, config: RewriteConfig):
        self.config = config

    def rewrite(self, user_query: str, n: int | None = None) -> list[str]:
        target_n = n if n is not None else self.config.rewrite_n
        prompt = f"""
        你是一个计算机课程的专业助教。用户的提问比较口语化，请将其转化为 {target_n} 个最可能出现在计算机专业教材目录中的标准中文术语或中文短语。
        用户提问: \"{user_query}\"
        要求:
        1. 只输出这 {target_n} 个中文术语，用英文逗号分隔。
        2. 不要包含任何解释、序号或多余的文字。
        3. 术语要精准、学术化。
        4. 遇到时间类（如第几周）或章节类（如1.1.1、1.1、1）这种请完整保存用户输入
        """

        try:
            temperature, max_tokens = get_adaptive_generation_params(prompt, task="rewrite")
            llm_output = ark_chat_completion(
                model=self.config.llm_model,
                prompt=prompt,
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                temperature=self.config.temperature if self.config.temperature is not None else temperature,
                max_tokens=self.config.max_tokens if self.config.max_tokens is not None else max_tokens,
                timeout_sec=self.config.timeout_sec,
            )
            keywords = [k.strip() for k in llm_output.replace("，", ",").split(",") if k.strip()]
            if not keywords:
                return [user_query]
            return keywords[:target_n]
        except Exception as exc:
            logger.error(
                "[%s] Query rewrite failed: %s. Falling back to original query.",
                self.config.group_name,
                exc,
            )
            return [user_query]


GROUP_REWRITE_DEFAULTS = {
    "A": {"rewrite_n": 3, "llm_model": DEFAULT_DEEPSEEK_ENDPOINT},
    "B": {"rewrite_n": 4, "llm_model": DEFAULT_DEEPSEEK_ENDPOINT},
    "C": {"rewrite_n": 5, "llm_model": DEFAULT_DEEPSEEK_ENDPOINT},
}


GROUP_RESOURCE_NAMES = {
    "A": {
        "node_label": "BaselineChunk",
        "index_name": "baseline_chunk_vector_index",
        "retrieval_strategy": "flat_vector_baseline",
    },
    "B": {
        "node_label": "RaptorTreeNodeB",
        "index_name": "group_b_tree_vector_index",
        "retrieval_strategy": "tree_retrieval",
    },
    "C": {
        "node_label": "GraphClusterNodeC",
        "index_name": "group_c_cluster_vector_index",
        "retrieval_strategy": "graph_cluster_retrieval",
    },
}


@dataclass
class RuntimeConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_api_base: str
    ark_api_key: str | None


def load_runtime_config(
    default_uri: str = "bolt://localhost:7687",
    default_user: str = "neo4j",
    default_password: str = "YOUR VALUE",
    default_llm_api_base: str = DEFAULT_ARK_API_BASE,
) -> RuntimeConfig:
    """
    统一读取运行时配置。
    Neo4j 仍可保留默认值复现实验；LLM 仅走云端 API。
    """
    return RuntimeConfig(
        neo4j_uri=os.getenv("NEO4J_URI", default_uri),
        neo4j_user=os.getenv("NEO4J_USER", default_user),
        neo4j_password=os.getenv("NEO4J_PASSWORD", default_password),
        llm_api_base=os.getenv("ARK_API_BASE", default_llm_api_base),
        ark_api_key=os.getenv("ARK_API_KEY"),
    )
