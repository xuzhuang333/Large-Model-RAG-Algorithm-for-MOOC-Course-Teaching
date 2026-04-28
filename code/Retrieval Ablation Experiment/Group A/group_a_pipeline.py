import json
import logging
import os
import sys
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
#这个代码文件用于Group A 的整体流程控制，包含了从查询重写、检索、生成到评测的完整闭环。

# 导入通用生成器和裁判
from llm_generator import RAGGenerator
from llm_evaluator import RAGEvaluator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from shared_retrieval_utils import (
    DEFAULT_ARK_API_BASE,
    DEFAULT_DEEPSEEK_ENDPOINT,
    DEFAULT_DOUBAO_ENDPOINT,
    GROUP_REWRITE_DEFAULTS,
    QueryRewriter,
    RewriteConfig,
    load_runtime_config,
)

# English logs and comments
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroupA_Pipeline_V2")

class GroupARetriever:
    def __init__(
        self,
        uri,
        user,
        password,
        llm_model=DEFAULT_DEEPSEEK_ENDPOINT,
        llm_api_base=DEFAULT_ARK_API_BASE,
        ark_api_key=None,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Loading BGE model for flat vector retrieval...")
        self.bge_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.ark_api_key = ark_api_key
        rewrite_defaults = GROUP_REWRITE_DEFAULTS["A"]

        # Query Rewrite 仅走云端火山方舟，不再使用本地 LLM。
        self.rewriter = QueryRewriter(
            RewriteConfig(
                group_name="A",
                rewrite_n=rewrite_defaults["rewrite_n"],
                llm_model=llm_model,
                api_base=self.llm_api_base,
                api_key=self.ark_api_key,
            )
        )

    def close(self):
        self.driver.close()

    def call_local_llm_rewrite(self, user_query, n=3):
        """Query Rewrite: Transforming spoken query into academic terms."""
        logger.info(f"Query Rewrite: Optimizing user query '{user_query}'...")
        keywords = self.rewriter.rewrite(user_query, n=n)
        logger.info(f"Expanded Terms: {keywords[:n]}")
        return keywords[:n]

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


def _build_context_chunks_with_budget(raw_chunks, budget_chars=2000):
    """
    按预算拼接检索上下文：
    - 严格控制总长度不超过 budget_chars
    - 只保留完整 chunk，不做硬截断，避免语义割裂
    - 一旦下一块会超预算，直接舍弃该块并停止
    """
    formatted_chunks = []
    retrieved_sources = []
    total_chars = 0

    for chunk in raw_chunks:
        source = chunk.get("source", "UNKNOWN_SOURCE")
        text = chunk.get("text", "")
        formatted_text = f"[Source: {source}]\n{text}"
        chunk_chars = len(formatted_text)

        if total_chars + chunk_chars > budget_chars:
            logger.info(
                "Context budget reached (%s chars). Dropped current complete chunk to preserve semantics.",
                budget_chars,
            )
            break

        formatted_chunks.append(formatted_text)
        retrieved_sources.append(source)
        total_chars += chunk_chars

    return formatted_chunks, retrieved_sources, total_chars


def _load_golden_sources_map(annotate_dataset_path):
    """
    从标注数据集中读取每个问题的 golden_sources，供评测阶段做来源对齐比较。
    """
    if not os.path.exists(annotate_dataset_path):
        logger.warning("Annotated dataset not found for source alignment: %s", annotate_dataset_path)
        return {}

    try:
        with open(annotate_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to load annotated dataset: %s", exc)
        return {}

    result = {}
    for item in data:
        qid = str(item.get("id", "")).strip()
        if not qid:
            continue
        sources = item.get("golden_sources", [])
        result[qid] = sources if isinstance(sources, list) else []
    return result

def run_group_a_experiment():
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="YOUR VALUE",
        default_llm_api_base=DEFAULT_ARK_API_BASE,
    )

    if not runtime_cfg.ark_api_key:
        logger.error("ARK_API_KEY not found. Please configure it in environment variables.")
        return

    retriever = GroupARetriever(
        runtime_cfg.neo4j_uri,
        runtime_cfg.neo4j_user,
        runtime_cfg.neo4j_password,
        llm_model=os.getenv("A_REWRITE_MODEL", DEFAULT_DEEPSEEK_ENDPOINT),
        llm_api_base=runtime_cfg.llm_api_base,
        ark_api_key=runtime_cfg.ark_api_key,
    )
    generator = RAGGenerator(
        model_name=os.getenv("A_GENERATOR_MODEL", DEFAULT_DEEPSEEK_ENDPOINT),
        api_url=runtime_cfg.llm_api_base,
        api_key=runtime_cfg.ark_api_key,
    )
    evaluator = RAGEvaluator(
        model_name=os.getenv("A_EVALUATOR_MODEL", DEFAULT_DOUBAO_ENDPOINT),
        api_url=runtime_cfg.llm_api_base,
        api_key=runtime_cfg.ark_api_key,
    )

    # 当前默认支持全周问题集，不再限制第1周。
    dataset_path = os.getenv("QA_DATASET_PATH", r"E:\graduate_project\code\QAdata\qa_dataset.json")
    annotate_dataset_path = os.getenv(
        "QA_ANNOTATE_DATASET_PATH",
        r"E:\graduate_project\code\QAdata\qa_dataset_to_annotate.json",
    )
    rewrite_n = int(os.getenv("A_REWRITE_N", str(GROUP_REWRITE_DEFAULTS["A"]["rewrite_n"])))
    top_k_per_term = int(os.getenv("A_TOP_K_PER_TERM", "3"))
    similarity_threshold = float(os.getenv("A_SIMILARITY_THRESHOLD", "0.5"))
    context_budget_chars = int(os.getenv("A_CONTEXT_BUDGET_CHARS", "2000"))
    only_qid = os.getenv("ONLY_QID", "").strip()

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # 预加载 golden_sources 映射，避免每轮循环重复读文件
    golden_sources_map = _load_golden_sources_map(annotate_dataset_path)

    results_log = []
    matched_count = 0

    for item in qa_data:
        q_id = item["id"]
        q_type = item["type"]
        raw_query = item["question"]

        # 支持单题调试：ONLY_QID=Q001 时仅运行该问题
        if only_qid and q_id != only_qid:
            continue

        matched_count += 1

        print(f"\n{'=' * 50}")
        logger.info(f"Testing [{q_id} | {q_type}]: {raw_query}")

        # Step 1: Query Rewrite
        expanded_terms = retriever.call_local_llm_rewrite(raw_query, n=rewrite_n)

        # Step 2: Retrieve (Group A Strategy)
        raw_chunks = retriever.flat_vector_search(
            expanded_terms,
            top_k_per_term=top_k_per_term,
            similarity_threshold=similarity_threshold,
        )

        # 检索上下文预算控制：严格限制总字符数，且保持 chunk 语义完整
        formatted_chunks, retrieved_sources, context_chars = _build_context_chunks_with_budget(
            raw_chunks,
            budget_chars=context_budget_chars,
        )
        logger.info("Context selected chars=%s, chunks=%s", context_chars, len(formatted_chunks))

        # Step 3: Generate
        answer, context_used = generator.generate_response(raw_query, formatted_chunks)
        print(f"\n[Generated Answer]:\n{answer}\n")

        # Step 4: Evaluate
        golden_sources = golden_sources_map.get(q_id, [])
        scores = evaluator.evaluate(
            raw_query,
            context_used,
            answer,
            retrieved_sources=retrieved_sources,
            golden_sources=golden_sources,
        )

        # Save record
        record = {
            "id": q_id,
            "type": q_type,
            "question": raw_query,
            "expanded_terms": expanded_terms,
            "evaluation_scores": scores,
            "generated_answer": answer,  # 顺便把答案也存下来方便人工核对
            "retrieved_sources": retrieved_sources,
            "golden_sources": golden_sources,
        }
        results_log.append(record)

    if only_qid and matched_count == 0:
        logger.warning("ONLY_QID=%s not found in dataset.", only_qid)

    output_file = os.getenv("A_OUTPUT_FILE", "eval_results_GroupA.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)

    logger.info(f"Group A experiment completed. Results saved to {output_file}")
    retriever.close()

if __name__ == "__main__":
    run_group_a_experiment()