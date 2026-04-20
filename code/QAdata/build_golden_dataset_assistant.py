import json
import logging
import importlib.util
import os
import sys
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 让 QAdata 脚本可复用实验目录中的共享配置工具
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ABLATION_DIR = os.path.join(PROJECT_ROOT, "Retrieval Ablation Experiment")


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
GROUP_RESOURCE_NAMES = _shared_utils.GROUP_RESOURCE_NAMES
load_runtime_config = _shared_utils.load_runtime_config

# 统一日志格式，便于后续排查数据集构建问题
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Golden_Dataset_Assistant")


class AnnotationAssistant:
    def __init__(self, uri, user, password, index_name):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # A 组候选召回索引名从共享命名配置读取，避免硬编码漂移
        self.index_name = index_name
        logger.info("Loading BGE model to fetch candidate chunks...")
        self.bge_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

    def fetch_candidates(self, query, top_k=100):
        """
        获取候选块以协助人工标注。
        逻辑：问题 -> 向量化 -> A组索引召回TopK -> 生成候选列表。
        """
        query_vector = self.bge_model.encode([query], normalize_embeddings=True)[0]
        query_vector_list = [float(x) for x in query_vector]

        candidates = []
        with self.driver.session() as session:
            # 查询A组基线索引以生成候选块
            cypher_query = """
            CALL db.index.vector.queryNodes($index_name, $k, $query_vec)
            YIELD node, score
            RETURN node.text AS text, node.source_file AS source, score
            """
            results = session.run(
                cypher_query,
                index_name=self.index_name,
                k=top_k,
                query_vec=query_vector_list,
            )
            for i, record in enumerate(results):
                candidates.append({
                    "candidate_id": i + 1,
                    "source": record["source"] or "UNKNOWN_SOURCE",
                    "score": round(record["score"], 4),
                    "text": record["text"] or ""
                })
        return candidates

    def close(self):
        self.driver.close()


def generate_annotation_task():
    # 默认保持可复现实验参数，同时允许环境变量覆盖
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
    )
    index_name = GROUP_RESOURCE_NAMES["A"]["index_name"]
    assistant = AnnotationAssistant(
        runtime_cfg.neo4j_uri,
        runtime_cfg.neo4j_user,
        runtime_cfg.neo4j_password,
        index_name=index_name,
    )

    # 使用脚本所在目录定位输入输出，避免受运行时工作目录影响
    input_file = os.getenv("QA_DATASET_PATH", os.path.join(CURRENT_DIR, "qa_dataset.json"))
    output_file = os.getenv("QA_ANNOTATE_OUTPUT_PATH", os.path.join(CURRENT_DIR, "qa_dataset_to_annotate.json"))
    top_k = int(os.getenv("ANNOTATION_TOP_K", "1000"))

    if not os.path.exists(input_file):
        logger.error(f"Cannot find {input_file}. Please create it with your raw questions first.")
        assistant.close()
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        annotated_data = []

        for item in qa_data:
            # 缺字段时跳过该样本，避免整批中断
            q_id = item.get("id")
            q_type = item.get("type")
            query = item.get("question")
            if not query:
                logger.warning(f"Skipping malformed item without question: {item}")
                continue

            logger.info(f"Fetching candidates for [{q_id} | {q_type}]: {query}")

            candidates = assistant.fetch_candidates(query, top_k=top_k)

            # 构建人工标注模板
            task_item = {
                "id": q_id,
                "type": q_type,
                "question": query,
                "ground_truth": "",  # 人工任务：填写标准答案
                "golden_sources": [],  # 人工任务：填写证据来源路径列表
                # AI辅助候选：用于支持人工撰写 ground_truth 与 golden_sources
                "candidate_reference_materials": candidates,
            }
            annotated_data.append(task_item)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotated_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Annotation task file generated: {output_file}")
        logger.info("Please open the file, read the candidates, and fill in 'ground_truth' and 'golden_sources'!")
    finally:
        assistant.close()


if __name__ == "__main__":
    generate_annotation_task()