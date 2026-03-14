import requests
import logging

# English logs and comments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Universal_LLM_Generator")


class RAGGenerator:
    def __init__(self, model_name="qwen2", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
        # Limit context length to avoid lost-in-the-middle
        self.max_context_length = 2500
        logger.info(f"Initialized Universal Generator with model: {self.model_name}")

    def generate_response(self, query, retrieved_chunks):
        """
        Generates an answer based STRICTLY on the retrieved context.
        This function will be reused across Group A, B, and C.
        """
        # Assemble context strings
        context_str = ""
        current_length = 0

        for i, chunk in enumerate(retrieved_chunks):
            # chunk can be a dict or string depending on your retriever output
            text = chunk['text'] if isinstance(chunk, dict) else chunk
            chunk_text = f"\n[Doc {i + 1}]:\n{text}\n"

            if current_length + len(chunk_text) > self.max_context_length:
                logger.warning("Context length limit reached. Truncating.")
                break

            context_str += chunk_text
            current_length += len(chunk_text)

        if not context_str.strip():
            context_str = "No relevant context found."

        # The Prompt Template (Frozen for A, B, C)
        prompt = f"""
        你是一位专业的大学计算机课程助教。请仔细阅读以下参考资料，并回答学生的问题。

        要求：
        1. 必须且只能基于提供的参考资料进行回答。
        2. 如果参考资料中没有相关信息，请直接回答“根据现有资料无法回答”。
        3. 回答要有逻辑，适当地分点阐述。

        【参考资料】:
        {context_str}

        【学生问题】:
        {query}

        请回答：
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            return answer, context_str
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return "Error during LLM generation.", context_str