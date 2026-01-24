import logging
import requests
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Smart_RAG")


class SmartRAGRetriever:
    def __init__(self, uri, user, password, llm_model="qwen2"):
        """
        初始化：连接 Neo4j，加载 BGE 向量模型，配置 LLM
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        logger.info("正在加载 BGE 向量模型...")
        self.bge_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

        self.llm_model = llm_model
        # Ollama 默认的本地 API 地址
        self.ollama_api_url = "http://localhost:11434/api/generate"
        logger.info(f"系统初始化完成。使用本地 LLM: {llm_model}")

    def close(self):
        self.driver.close()

    def call_local_llm_rewrite(self, user_query, n=3):
        """
        核心步骤 1：利用本地 LLM 将用户的口语化问题，凝练成 n 个专业教材术语
        """
        logger.info(f"正在思考如何优化问题: '{user_query}'...")

        # 这是一个精心设计的 Prompt (提示词工程)
        prompt = f"""
        你是一个计算机课程的专业助教。用户的提问比较口语化，请将其转化为 {n} 个最可能出现在计算机专业教材目录中的标准术语或短语。

        用户提问: "{user_query}"

        要求:
        1. 只输出这 {n} 个术语，用英文逗号分隔。
        2. 不要包含任何解释、序号或多余的文字。
        3. 术语要精准、学术化。

        示例输入: "python怎么入门"
        示例输出: Python语言概述,程序设计基本方法,开发环境配置

        请回答:
        """

        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response_json = response.json()
            llm_output = response_json.get("response", "").strip()

            # 处理 LLM 的输出，分割成列表
            # 替换可能出现的中文逗号
            keywords = [k.strip() for k in llm_output.replace("，", ",").split(",") if k.strip()]

            # 如果 LLM 没生成够，把原始问题也加进去保底
            if not keywords:
                keywords = [user_query]

            logger.info(f"LLM 联想出的专业术语: {keywords}")
            return keywords[:n]  # 确保只返回前 n 个

        except Exception as e:
            logger.error(f"调用本地 LLM 失败: {e}。将使用原始问题搜索。")
            return [user_query]

    def vector_search(self, search_terms, top_k_per_term=3, similarity_threshold=0.6):
        """
        [改进版] 核心步骤 2 & 3：基于阈值的多路召回策略

        Args:
            search_terms: LLM 扩展出的关键词列表
            top_k_per_term: 每个关键词去查几个备选节点
            similarity_threshold: 相似度门槛 (0~1)，低于这个分数的节点会被直接丢弃
        """
        all_results = {}  # 字典用于去重：Key=节点名, Value=节点详细信息

        with self.driver.session() as session:
            for term in search_terms:
                # 2.1 将术语转换为向量
                query_vector = self.bge_model.encode([term], normalize_embeddings=True)[0]
                query_vector_list = [float(x) for x in query_vector]

                # 2.2 去 Neo4j 搜索 (每个词查 Top-K 个)
                cypher_query = """
                CALL db.index.vector.queryNodes('concept_name_vector_index', $k, $query_vec)
                YIELD node, score
                RETURN node.name AS name, node.source_folder AS path, score
                """

                results = session.run(cypher_query, k=top_k_per_term, query_vec=query_vector_list)

                for record in results:
                    node_name = record['name']
                    score = record['score']

                    # --- 核心改进逻辑 START ---

                    # 1. 阈值过滤：分数不够高，直接忽略，防止引入无关噪音
                    if score < similarity_threshold:
                        continue

                    # 2. 结果融合 (Result Fusion)
                    # 如果这个节点是第一次出现，直接加入结果集
                    if node_name not in all_results:
                        all_results[node_name] = {
                            'name': node_name,
                            'score': score,
                            'path': record['path'],
                            'matched_terms': [term]  # 记录它是被哪个词命中的
                        }
                    else:
                        # 如果这个节点之前已经被别的词搜到了
                        # 1. 我们保留最高的那个分数 (Max-Score Strategy)
                        if score > all_results[node_name]['score']:
                            all_results[node_name]['score'] = score
                        # 2. 记录一下它被多个词同时命中了 (这侧面说明它很重要)
                        if term not in all_results[node_name]['matched_terms']:
                            all_results[node_name]['matched_terms'].append(term)

                    # --- 核心改进逻辑 END ---

        # 3. 将字典转为列表，并按分数从高到低排序，供下一步使用
        sorted_results = sorted(all_results.values(), key=lambda x: x['score'], reverse=True)

        logger.info(f"多路召回完成：共找到 {len(sorted_results)} 个符合阈值({similarity_threshold})的知识点。")
        return sorted_results

    def process_query(self, user_prompt):
        print(f"\n👤 用户提问: {user_prompt}")
        print("-" * 50)

        # 1. LLM 扩展
        expanded_terms = self.call_local_llm_rewrite(user_prompt)

        # 2. 向量搜索 (带阈值和融合)
        results = self.vector_search(expanded_terms, similarity_threshold=0.6)

        # 3. 展示结果
        print(f"\n📚 检索结果 (Top {len(results)}):")
        for i, res in enumerate(results):
            # 获取节点名称，防止字典里没有 'name' 键 (虽然逻辑上应该有)
            node_name = res.get('name', 'Unknown Node')

            print(f"[{i + 1}] 匹配节点: {node_name}")
            print(f"    相似度: {res['score']:.4f}")

            terms_str = ", ".join(res['matched_terms'])
            print(f"    来源术语: {terms_str}")
            # -------------------

            print(f"    物理路径: {res['path']}")
            print("-" * 30)


# --- 修正 vector_search 中的一行代码 ---
# 在 all_results[node_name] = { ... } 中，建议加上 'name': node_name 以便打印
# (上面的代码如果不改，print 的时候只能从 key 获取，或者在这里加进去)

if __name__ == "__main__":
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"

    # 确保安装了 Ollama 并且运行过 `ollama run qwen2`
    searcher = SmartRAGRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, llm_model="qwen2")

    # 测试案例
    searcher.process_query("Python怎么入门呀？")

    searcher.close()