import requests
import logging
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    ark_chat_completion,
    get_adaptive_generation_params,
)

# English logs and comments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Universal_LLM_Generator")


class RAGGenerator:
    def __init__(self, model_name=DEFAULT_DEEPSEEK_ENDPOINT, api_url=DEFAULT_ARK_API_BASE, api_key=None):
        # 仅使用云端模型接入点（Endpoint ID）
        self.model_name = model_name
        # 兼容旧参数名 api_url，这里实际表示 API Base（例如 https://ark.cn-beijing.volces.com/api/v3）
        self.api_url = api_url
        # 上下文长度控制：默认 2000 字符，可由环境变量调整
        self.max_context_length = int(os.getenv("A_CONTEXT_MAX_CHARS", "2000"))
        # 优先使用显式传入 key，否则从 ARK_API_KEY 读取
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        logger.info(f"Initialized Universal Generator with model: {self.model_name}")

    def _infer_answer_mode(self, query, answer_mode=None):
        forced = str(answer_mode or "").strip().lower()
        if forced in {"micro", "macro"}:
            return forced

        query_text = str(query or "").strip()
        macro_hints = (
            "总结",
            "概述",
            "整体",
            "框架",
            "比较",
            "区别",
            "联系",
            "原理",
            "流程",
            "步骤",
            "为什么",
            "有哪些",
            "主要",
            "核心",
            "如何理解",
        )
        micro_hints = (
            "是什么",
            "哪一个",
            "哪个",
            "多少",
            "是否",
            "定义",
            "全称",
            "参数",
            "返回",
            "缩写",
        )

        if any(hint in query_text for hint in macro_hints):
            return "macro"
        if any(hint in query_text for hint in micro_hints):
            return "micro"

        return "micro" if len(query_text) <= 24 else "macro"

    def _build_prompt(self, query, context_str, answer_mode):
        common_rules = """
        通用要求：
        1. 必须且只能基于提供的参考资料进行回答。
        2. 如果参考资料中没有相关信息，请直接回答“根据现有资料无法回答”。
        3. 不要添加任何未提供的背景知识或假设。
        4. 绝对禁止编造信息或过度推测。只能回答参考资料中明确提到的内容。
        5. 绝对禁止使用任何未提供的外部知识或常识推理。
        """.strip()

        if answer_mode == "macro":
            format_rules = """
            回答模板（macro）：
            - 先用1句话给出总述。
            - 再按“要点1/要点2/要点3 ...”分点回答。
            - 每个要点尽量对应参考资料中的一个清晰事实。
            - 若证据不足，明确指出“哪一部分资料不足”。
            """.strip()
        else:
            format_rules = """
            回答模板（micro）：
            - 首句直接给出结论（尽量短）。
            - 如需补充，最多补1-2句关键依据。
            - 不要展开与问题无关的背景介绍。
            """.strip()

        return f"""
        你是严谨的计算机课程助教。请阅读参考资料并回答学生问题。

        {common_rules}

        {format_rules}

        【参考资料】:
        {context_str}

        【学生问题】:
        {query}

        请回答：
        """

    def generate_response(self, query, retrieved_chunks, answer_mode=None):
        """
        Generates an answer based STRICTLY on the retrieved context.
        This function will be reused across Group A, B, and C.
        """
        # 组装上下文：只保留完整 chunk，不做硬截断，避免语义割裂
        context_str = ""
        current_length = 0

        for i, chunk in enumerate(retrieved_chunks):
            # chunk can be a dict or string depending on your retriever output
            text = chunk['text'] if isinstance(chunk, dict) else chunk
            chunk_text = f"\n[Doc {i + 1}]:\n{text}\n"

            if current_length + len(chunk_text) > self.max_context_length:
                logger.warning("Context budget reached. Stop adding more chunks to preserve chunk integrity.")
                break

            context_str += chunk_text
            current_length += len(chunk_text)

        if not context_str.strip():
            context_str = "No relevant context found."

        mode = self._infer_answer_mode(query=query, answer_mode=answer_mode)
        prompt = self._build_prompt(query=query, context_str=context_str, answer_mode=mode)

        try:
            # 根据输入长度自适应控制温度和输出 token，兼顾效果与成本
            temperature, max_tokens = get_adaptive_generation_params(prompt, task="generate")
            answer = ark_chat_completion(
                model=self.model_name,
                prompt=prompt,
                api_base=self.api_url,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=180,
            )
            return answer, context_str
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return "Error during LLM generation.", context_str