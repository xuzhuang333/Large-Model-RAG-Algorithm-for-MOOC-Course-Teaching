import json
import logging

import requests

logger = logging.getLogger("Llama3.1_Jury")


class RAGEvaluator:
    def __init__(self, model_name="llama3.1", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url
        logger.info(f"Initialized LLM-as-a-Judge with model: {self.model_name}")

    def evaluate(self, query, context, answer):
        """
        Evaluates the RAG pipeline output across 4 dimensions (Score 1-5).
        Returns a JSON object containing the metrics.
        """
        prompt = f"""
        You are an impartial judge evaluating a Retrieval-Augmented Generation (RAG) system.
        You will be given a Question, the retrieved Context, and the generated Answer.
        Please rate the following 4 metrics on a scale of 1 to 5 (1 being worst, 5 being best):

        1. Context_Recall: Does the Context contain ALL the necessary information to answer the Question?
        2. Context_Precision: Is the Context highly relevant to the Question, without unnecessary noise?
        3. Faithfulness: Is the Answer strictly based on the Context without hallucinations?
        4. Answer_Relevance: Does the Answer directly and effectively address the Question?

        [Question]: {query}
        [Context]: {context}
        [Answer]: {answer}

        Output your evaluation strictly in the following JSON format without any other text:
        {{"Context_Recall": <score>, "Context_Precision": <score>, "Faithfulness": <score>, "Answer_Relevance": <score>}}
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"  # Force Ollama to output valid JSON
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result_text = response.json().get("response", "")
            scores = json.loads(result_text)
            logger.info(f"Evaluation completed: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Evaluation failed or invalid JSON: {e}")
            return {"Context_Recall": 0, "Context_Precision": 0, "Faithfulness": 0, "Answer_Relevance": 0}