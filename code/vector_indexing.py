import logging
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 配置日志输出，方便看到现在的进度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Vector_Indexer")

"""
产生向量标签
"""
class VectorIndexer:
    def __init__(self, uri, user, password):
        """
        初始化连接和模型
        """
        # 1. 连接 Neo4j 数据库
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # 2. 加载 Embedding 模型
        # 我们使用 BGE-Small 中文模型，它速度快、效果好，且输出维度是 512
        # 第一次运行会自动下载模型权重，大概需要几分钟，请耐心等待
        logger.info("正在加载 BGE 模型 (BAAI/bge-small-zh-v1.5)...")
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        logger.info("模型加载成功！准备开始工作。")

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def add_embeddings_to_nodes(self):
        """
        核心功能：取出所有节点 -> 计算向量 -> 存回数据库
        """
        with self.driver.session() as session:
            # 第一步：查找所有还没有向量数据的 Concept 节点
            # 这种 WHERE IS NULL 的写法可以支持断点续传
            logger.info("🔍 正在查询数据库中需要向量化的节点...")
            result = session.run("MATCH (n:Concept) WHERE n.embedding IS NULL RETURN n.name AS name")
            nodes = [record["name"] for record in result]

            count = len(nodes)
            logger.info(f"📊 发现 {count} 个节点需要生成向量。")

            if count == 0:
                logger.info("所有节点都已经有向量了，无需重复操作。")
                return

            # 第二步：批量计算向量 (Embedding)
            # model.encode 会返回一个列表，里面包含了每个名字对应的 512 维向量
            # normalize_embeddings=True 表示进行归一化，这样以后计算余弦相似度更准
            logger.info("🧠 正在计算向量 (Embedding)... 这可能需要一点时间")
            embeddings = self.model.encode(nodes, normalize_embeddings=True)

            # 第三步：将计算好的向量写回 Neo4j
            logger.info("💾 正在将向量写入 Neo4j 数据库...")

            # 我们使用 zip 将名字和向量配对，逐个更新
            for name, vector in zip(nodes, embeddings):
                # Neo4j 要求向量必须是 List[float] 格式，这里做一个强制类型转换
                vector_list = [float(x) for x in vector]

                # 使用 Cypher 语句更新节点属性
                # db.create.setNodeVectorProperty 是 Neo4j 5.x 推荐的高效写入向量的方法
                query = """
                MATCH (n:Concept {name: $name})
                CALL db.create.setNodeVectorProperty(n, 'embedding', $vector)
                """
                session.run(query, name=name, vector=vector_list)

            logger.info(f"✅ 成功更新了 {count} 个节点的向量数据！")

    def create_vector_index(self):
        """
        创建向量索引 (Vector Index)
        这一步至关重要，没有索引，以后的搜索就是全表扫描，速度会很慢。
        """
        index_name = "concept_name_vector_index"
        with self.driver.session() as session:
            logger.info(f"⚙️ 正在检查或创建向量索引: {index_name}...")

            # Cypher 语句解释：
            # FOR (c:Concept) ON (c.embedding): 针对 Concept 标签的 embedding 属性建索引
            # vector.dimensions: 512 -> 必须和 BGE 模型的输出维度一致
            # vector.similarity_function: 'cosine' -> 使用余弦相似度进行搜索
            query = """
            CREATE VECTOR INDEX concept_name_vector_index IF NOT EXISTS
            FOR (c:Concept)
            ON (c.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 512,
             `vector.similarity_function`: 'cosine'
            }}
            """
            try:
                session.run(query)
                logger.info(f"✅ 向量索引 '{index_name}' 准备就绪。")
            except Exception as e:
                logger.error(f"❌ 创建索引失败: {e}")


# --- 程序入口 ---
if __name__ == "__main__":
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"

    indexer = VectorIndexer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 1. 计算并存储向量
        indexer.add_embeddings_to_nodes()

        # 2. 创建索引 (有了这个才能进行搜索)
        indexer.create_vector_index()

    finally:
        indexer.close()