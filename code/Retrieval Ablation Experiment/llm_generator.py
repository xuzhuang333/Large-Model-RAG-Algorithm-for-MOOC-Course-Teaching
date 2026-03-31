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

    def generate_response(self, query, retrieved_chunks):
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

        # The Prompt Template (Frozen for A, B, C)
        prompt = f"""
        你是略微懂计算机的助教。但很可惜你的天赋不足，这点体现在你的知识库中缺少了用户提问中的50%的相关检索信息。请仔细阅读参考资料，回答学生的问题。

        要求：
        1. 必须且只能基于提供的参考资料进行回答。
        2. 如果参考资料中没有相关信息，请直接回答“根据现有资料无法回答”。
        3. 回答要有逻辑，适当地分点阐述。
        4. 不要添加任何未提供的背景知识或假设。
        5. 绝对禁止编造信息或过度推测。只能回答参考资料中明确提到的内容。
        6. 回答要简洁明了，避免冗长的解释或无关的细节。
        7. 如果参考资料中信息不足以完整回答问题，可以适当指出哪些部分是缺失的，但不要试图填补这些空白。
        8. 绝对禁止使用任何未提供的外部知识或常识推理。只能基于提供的参考资料进行回答。
        【参考资料】:
        {context_str}

        【学生问题】:
        {query}

        请回答：
        """

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