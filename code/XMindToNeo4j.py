import os
import json
import zipfile
import logging
import hashlib
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Global_Hierarchy_Indexer")


class GlobalHierarchyIndexer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Database connection established.")

    def close(self):
        self.driver.close()

    def parse_xmind_content(self, xmind_path):
        """Parse content.json from XMind file."""
        try:
            with zipfile.ZipFile(xmind_path, 'r') as z:
                if 'content.json' in z.namelist():
                    with z.open('content.json') as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse XMind {xmind_path}: {e}")
        return None

    def store_xmind_hierarchy(self, tx, topic, xmind_path, folder_path, parent_id):
        """
        Store XMind internal structure.
        xmind_path: 当前 XMind 文件的完整路径（用于生成唯一节点 ID）
        folder_path: 所在文件夹路径（用于备选 ID 生成）
        parent_id: 父节点 ID（可以是文件夹节点或父主题节点）
        """
        topic_id = topic.get('id')
        title = topic.get('title', 'Untitled')

        # 结合 XMind 文件路径和内部 ID（或标题）生成全局唯一 ID
        unique_str = f"{xmind_path}|{topic_id if topic_id else title}"
        node_id = hashlib.md5(unique_str.encode('utf-8')).hexdigest()

        # 创建 XMind concept 节点（保留原始 title 用于显示）
        query = """
        MERGE (c:Concept {node_id: $id})
        SET c.name = $name, c.source_folder = $folder, c.node_type = 'XMind_Node'
        RETURN c
        """
        tx.run(query, id=node_id, name=title, folder=folder_path)

        # 链接到父节点
        rel_query = """
        MATCH (p:Concept {node_id: $p_id}), (c:Concept {node_id: $c_id})
        MERGE (p)-[:HAS_SUBTOPIC]->(c)
        """
        tx.run(rel_query, p_id=parent_id, c_id=node_id)

        # 处理子主题（传递相同的 xmind_path）
        children = topic.get('children', {}).get('attached', [])
        for child in children:
            self.store_xmind_hierarchy(tx, child, xmind_path, folder_path, node_id)

    def sync_directory_to_graph(self, root_dir):
        """
        Build the graph starting from the physical directory structure.
        Folders become nodes, establishing the macro-hierarchy of the course.
        """
        logger.info(f"Scanning directory hierarchy from: {root_dir}")

        with self.driver.session() as session:
            for root, dirs, files in os.walk(root_dir):
                folder_name = os.path.basename(root)
                if not folder_name:
                    continue

                # Use the absolute path hash as the unique ID for the folder node
                folder_id = hashlib.md5(root.encode('utf-8')).hexdigest()

                # 1. Create the Folder Concept node
                session.run("""
                MERGE (f:Concept {node_id: $id})
                SET f.name = $name, f.source_folder = $path, f.node_type = 'Folder'
                """, id=folder_id, name=folder_name, path=root)

                # 2. Link this Folder to its Parent Folder (if within root_dir)
                parent_dir = os.path.dirname(root)
                if parent_dir >= root_dir and parent_dir != root:
                    parent_id = hashlib.md5(parent_dir.encode('utf-8')).hexdigest()
                    session.run("""
                    MATCH (p:Concept {node_id: $p_id}), (c:Concept {node_id: $c_id})
                    MERGE (p)-[:HAS_SUBTOPIC]->(c)
                    """, p_id=parent_id, c_id=folder_id)

                # 3. Mount XMind micro-hierarchy under this Folder node
                for file in files:
                    if file.endswith(".xmind"):
                        xmind_path = os.path.join(root, file)
                        sheets = self.parse_xmind_content(xmind_path)
                        if sheets:
                            for sheet in sheets:
                                root_topic = sheet.get('rootTopic')
                                if root_topic:
                                    session.execute_write(
                                        self.store_xmind_hierarchy,
                                        root_topic,
                                        xmind_path,  # 新增参数：XMind 文件路径
                                        root,  # 文件夹路径
                                        folder_id  # 父节点（当前文件夹）
                                    )
        logger.info("Global hierarchy graph construction completed successfully.")


if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"  # 替换为实际密码

    ROOT_DATA_DIR = r"E:\graduate_project\reference material"

    indexer = GlobalHierarchyIndexer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        indexer.sync_directory_to_graph(ROOT_DATA_DIR)
    finally:
        indexer.close()