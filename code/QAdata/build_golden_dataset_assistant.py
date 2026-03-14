import json
import logging
import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 英文日志和注释
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Golden_Dataset_Assistant")


class AnnotationAssistant:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Loading BGE model to fetch candidate chunks...")
        self.bge_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    def fetch_candidates(self, query, top_k=100):
        """
        获取广泛的候选块以协助人工标注。
        """
        query_vector = self.bge_model.encode([query], normalize_embeddings=True)[0]
        query_vector_list = [float(x) for x in query_vector]

        candidates = []
        with self.driver.session() as session:
            # 查询A组基线索引以生成候选块
            cypher_query = """
            CALL db.index.vector.queryNodes('baseline_chunk_vector_index', $k, $query_vec)
            YIELD node, score
            RETURN node.text AS text, node.source_file AS source, score
            """
            results = session.run(cypher_query, k=top_k, query_vec=query_vector_list)
            for i, record in enumerate(results):
                candidates.append({
                    "candidate_id": i + 1,
                    "source": record["source"],
                    "score": round(record["score"], 4),
                    "text": record["text"]
                })
        return candidates

    def close(self):
        self.driver.close()


def generate_annotation_task():
    # 初始化连接（更新密码！）
    assistant = AnnotationAssistant("bolt://localhost:7687", "neo4j", "12345678")

    input_file = "qa_dataset.json"
    output_file = "qa_dataset_to_annotate.json"

    if not os.path.exists(input_file):
        logger.error(f"Cannot find {input_file}. Please create it with your raw questions first.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    annotated_data = []

    for item in qa_data:
        query = item["question"]
        logger.info(f"Fetching candidates for: {query}")

        candidates = assistant.fetch_candidates(query, top_k=1000)

        # 构建人工标注的模板
        task_item = {
            "id": item["id"],
            "type": item["type"],
            "question": query,
            "ground_truth": "",  # 人工任务：填写此项！
            "golden_sources": [],
            # 人工任务：添加包含正确答案的来源名称（例如"1.1.1 单元开篇/1.txt"）
            "candidate_reference_materials": candidates  # AI辅助：阅读这些材料以撰写正确答案
        }
        annotated_data.append(task_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotated_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Annotation task file generated: {output_file}")
    logger.info("Please open the file, read the candidates, and fill in 'ground_truth' and 'golden_sources'!")
    assistant.close()


if __name__ == "__main__":
    generate_annotation_task()