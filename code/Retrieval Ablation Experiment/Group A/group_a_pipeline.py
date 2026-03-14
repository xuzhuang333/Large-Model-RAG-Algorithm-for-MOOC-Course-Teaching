import json
import logging
import os
import requests
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 导入通用生成器和裁判
from llm_generator import RAGGenerator
from llm_evaluator import RAGEvaluator

# English logs and comments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroupA_Pipeline_V2")

class GroupARetriever:
    def __init__(self, uri, user, password, llm_model="qwen2"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Loading BGE model for flat vector retrieval...")
        self.bge_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self.llm_model = llm_model
        self.ollama_api_url = "http://localhost:11434/api/generate"

    def close(self):
        self.driver.close()

    def call_local_llm_rewrite(self, user_query, n=3):
        """Query Rewrite: Transforming spoken query into academic terms."""
        logger.info(f"Query Rewrite: Optimizing user query '{user_query}'...")
        prompt = f"""
        你是一个计算机课程的专业助教。用户的提问比较口语化，请将其转化为 {n} 个最可能出现在计算机专业教材目录中的标准中文术语或中文短语。
        用户提问: "{user_query}"
        要求:
        1. 只输出这 {n} 个中文术语，用英文逗号分隔。
        2. 不要包含任何解释、序号或多余的文字。
        3. 术语要精准、学术化。
        4. 遇到时间类（如第几周）或章节类（如1.1.1、1.1、1）这种请完整保存用户输入
        """
        payload = {"model": self.llm_model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()
            llm_output = response.json().get("response", "").strip()
            keywords = [k.strip() for k in llm_output.replace("，", ",").split(",") if k.strip()]
            if not keywords:
                keywords = [user_query]
            logger.info(f"Expanded Terms: {keywords[:n]}")
            return keywords[:n]
        except Exception as e:
            logger.error(f"LLM rewrite failed: {e}. Falling back to original query.")
            return [user_query]

    def flat_vector_search(self, search_terms, top_k_per_term=3, similarity_threshold=0.5):
        """
        Group A Specific: Multi-path retrieval from BaselineChunk
        """
        all_results = {}

        with self.driver.session() as session:
            for term in search_terms:
                query_vector = self.bge_model.encode([term], normalize_embeddings=True)[0]
                query_vector_list = [float(x) for x in query_vector]

                # Match the exact properties we defined in the new BaselineIndexerV3
                cypher_query = """
                CALL db.index.vector.queryNodes('baseline_chunk_vector_index', $k, $query_vec)
                YIELD node, score
                RETURN node.text AS text, node.source_file AS source, score
                """
                results = session.run(cypher_query, k=top_k_per_term, query_vec=query_vector_list)

                for record in results:
                    text = record['text']
                    score = record['score']
                    source = record['source']

                    if score < similarity_threshold:
                        continue

                    if text not in all_results:
                        all_results[text] = {"text": text, "source": source, "score": score}
                    else:
                        if score > all_results[text]['score']:
                            all_results[text]['score'] = score

        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)
        logger.info(f"Flat retrieval found {len(sorted_results)} chunks above threshold.")
        return sorted_results

def run_group_a_experiment():
    retriever = GroupARetriever("bolt://localhost:7687", "neo4j", "12345678")  # 修改密码
    generator = RAGGenerator(model_name="qwen2")
    evaluator = RAGEvaluator(model_name="llama3.1")

    # 注意：确保这里的问题只包含“第1周”的内容！
    dataset_path = r"E:\graduate_project\code\QAdata\qa_dataset.json" 
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    results_log = []

    for item in qa_data:
        q_id = item["id"]
        q_type = item["type"]
        raw_query = item["question"]

        print(f"\n{'=' * 50}")
        logger.info(f"Testing [{q_id} | {q_type}]: {raw_query}")

        # Step 1: Query Rewrite
        expanded_terms = retriever.call_local_llm_rewrite(raw_query)

        # Step 2: Retrieve (Group A Strategy)
        raw_chunks = retriever.flat_vector_search(expanded_terms, top_k_per_term=3, similarity_threshold=0.5)
        
        # 【重大改进】：将 source 组装进文本中，让裁判和大模型知道信息来源
        formatted_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Format: [Source: 1.1/xxx.md] \n Content...
            formatted_text = f"[Source: {chunk['source']}]\n{chunk['text']}"
            formatted_chunks.append(formatted_text)

        # Step 3: Generate
        answer, context_used = generator.generate_response(raw_query, formatted_chunks)
        print(f"\n[Generated Answer]:\n{answer}\n")

        # Step 4: Evaluate
        scores = evaluator.evaluate(raw_query, context_used, answer)

        # Save record
        record = {
            "id": q_id,
            "type": q_type,
            "question": raw_query,
            "expanded_terms": expanded_terms,
            "evaluation_scores": scores,
            "generated_answer": answer # 顺便把答案也存下来方便人工核对
        }
        results_log.append(record)

    output_file = "eval_results_GroupA.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)

    logger.info(f"Group A experiment completed. Results saved to {output_file}")
    retriever.close()

if __name__ == "__main__":
    run_group_a_experiment()