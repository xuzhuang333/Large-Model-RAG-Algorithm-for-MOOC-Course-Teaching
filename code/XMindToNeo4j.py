import os
import json
import zipfile
import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XMind_Neo4j_Indexer")


class XMindToNeo4j:
    def __init__(self, uri, user, password):
        """
                初始化数据库连接
                :param uri: Neo4j 的连接地址 (通常是 bolt://localhost:7687)
                :param user: 用户名
                :param password: 密码
                """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("连接数据库成功.")

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def parse_xmind_content(self, xmind_path):
        """
                解析 XMind 文件内容，其核心数据存储在 content.json 中
        """

        try:
            with zipfile.ZipFile(xmind_path, 'r') as z:
                #打开压缩包内的 content.json 文件
                with z.open('content.json') as f:
                    data = json.load(f)
                    return data  #返回的是一个列表，包含思维导图的所有画布(Sheets)
        except Exception as e:
            logger.error(f"Failed to parse XMind file at {xmind_path}: {e}")
            return None

    def store_hierarchy(self, tx, topic, folder_path, parent_name=None):
        """
                递归函数：将思维导图的层级结构存入 Neo4j
                :param tx: Neo4j 的事务对象
                :param topic: 当前处理的主题节点（JSON 格式）
                :param folder_path: 该 XMind 文件所在的物理路径（用于后续检索对应文件）
                :param parent_name: 父节点的名称
        """
        title = topic.get('title', 'Untitled')

        # 使用 MERGE 语句创建节点：如果节点已存在则跳过，不存在则创建
        # 同时将物理文件夹路径存入 source_folder 属性
        query = """
        MERGE (c:Concept {name: $name})
        SET c.source_folder = $folder
        RETURN c
        """
        tx.run(query, name=title, folder=folder_path)

        if parent_name:
            # 如果存在父节点，则创建“HAS_SUBTOPIC”（拥有子主题）的关系
            rel_query = """
            MATCH (p:Concept {name: $p_name}), (c:Concept {name: $c_name})
            MERGE (p)-[:HAS_SUBTOPIC]->(c)
            """
            tx.run(rel_query, p_name=parent_name, c_name=title)

        # 递归处理所有子主题
        # XMind 的子主题存储在 children -> attached 下
        children = topic.get('children', {}).get('attached', [])
        for child in children:
            self.store_hierarchy(tx, child, folder_path, title)

    def process_directory(self, root_dir):
        """
                遍历目标目录，寻找并处理所有的 .xmind 文件
                :param root_dir: 数据集的根目录路径
        """
        logger.info(f"Starting directory walk in: {root_dir}")

        with self.driver.session() as session:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(".xmind"):
                        file_path = os.path.join(root, file)
                        logger.info(f"Processing file: {file_path}")

                        sheets = self.parse_xmind_content(file_path)
                        if sheets:
                            for sheet in sheets:
                                root_topic = sheet.get('rootTopic')
                                if root_topic:
                                    session.execute_write(
                                        self.store_hierarchy,
                                        root_topic,
                                        root
                                    )
        logger.info("Batch indexing completed.")


# --- Execution ---
if __name__ == "__main__":
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"

    ROOT_DATA_DIR = r"E:\graduate_project\reference material"

    indexer = XMindToNeo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        indexer.process_directory(ROOT_DATA_DIR)
    finally:
        indexer.close()