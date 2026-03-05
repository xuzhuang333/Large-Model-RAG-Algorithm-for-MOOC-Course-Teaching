import zipfile
import json
import os
import logging
from neo4j import GraphDatabase

# Configure logging (English logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("XMind_Patcher")


class XMindPatcher:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("XMind Patcher initialized.")

    def close(self):
        self.driver.close()

    def merge_node_to_neo4j(self, title, source_folder):
        """Creates or updates a Concept node in Neo4j."""
        with self.driver.session() as session:
            query = """
            MERGE (n:Concept {name: $title})
            SET n.source_folder = $source_folder
            RETURN n
            """
            session.run(query, title=title, source_folder=source_folder)

    def create_relationship(self, parent_title, child_title):
        """Creates a HAS_SUBTOPIC relationship between two Concept nodes."""
        with self.driver.session() as session:
            query = """
            MATCH (p:Concept {name: $parent_title})
            MATCH (c:Concept {name: $child_title})
            MERGE (p)-[:HAS_SUBTOPIC]->(c)
            """
            session.run(query, parent_title=parent_title, child_title=child_title)

    def parse_children_recursive(self, parent_title, children_data, source_folder):
        """Recursively parses the children, strictly handling the 'attached' dictionary structure."""
        # Check if children_data is a dictionary and contains 'attached'
        if isinstance(children_data, dict) and 'attached' in children_data:
            children_list = children_data['attached']
        elif isinstance(children_data, list):
            # Fallback for older XMind versions where children might directly be a list
            children_list = children_data
        else:
            return

        for child in children_list:
            child_title = child.get('title')
            if not child_title:
                continue

            # 1. Create the child node
            self.merge_node_to_neo4j(child_title, source_folder)

            # 2. Link child to parent
            self.create_relationship(parent_title, child_title)
            logger.info(f"Linked: [{parent_title}] -> [{child_title}]")

            # 3. Recursion for deeper levels
            child_children = child.get('children')
            if child_children:
                self.parse_children_recursive(child_title, child_children, source_folder)

    def patch_xmind_file(self, xmind_path, expected_root_name):
        """Extracts JSON from XMind and patches the graph."""
        logger.info(f"Patching XMind file: {xmind_path}")
        source_folder = os.path.dirname(xmind_path)

        try:
            with zipfile.ZipFile(xmind_path, 'r') as xmind_zip:
                if 'content.json' in xmind_zip.namelist():
                    with xmind_zip.open('content.json') as f:
                        content = json.loads(f.read().decode('utf-8'))

                        if isinstance(content, list) and len(content) > 0:
                            root_topic = content[0].get('rootTopic', {})
                            xmind_title = root_topic.get('title', 'UNKNOWN')

                            # 1. Merge the expected root name (e.g., "4.4.1 单元开篇")
                            self.merge_node_to_neo4j(expected_root_name, source_folder)

                            # 2. Merge the actual XMind title (e.g., "程序的分支结构")
                            self.merge_node_to_neo4j(xmind_title, source_folder)

                            # 3. Create a bridge relationship
                            # This connects the folder name to the actual XMind concept
                            if expected_root_name != xmind_title:
                                self.create_relationship(expected_root_name, xmind_title)
                                logger.info(
                                    f"Bridged physical folder [{expected_root_name}] to XMind root [{xmind_title}]")

                            # 4. Start recursive parsing from the XMind root
                            children_data = root_topic.get('children')
                            if children_data:
                                self.parse_children_recursive(xmind_title, children_data, source_folder)

                            logger.info(f"Successfully patched graph for: {expected_root_name}")
        except Exception as e:
            logger.error(f"Failed to patch {xmind_path}: {e}")


if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"  # 改为你的密码

    patcher = XMindPatcher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # 填入那几个漏掉的 XMind 文件的真实路径和对应的预期根节点名
    missing_files = [
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第3周】基本数据类型\3.3 字符串类型及操作\3.3.1 单元开篇\3.3.1 单元开篇.xmind",
            "expected_name": "3.3.1 单元开篇"
        },
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第4周】程序的控制结构\4.1 程序的分支结构\4.1.1 单元开篇\4.1.1 单元开篇.xmind",
            "expected_name": "4.1.1 单元开篇"
        },
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第4周】程序的控制结构\4.3 程序的循环结构\4.3.1 单元开篇\4.3.1 单元开篇.xmind",
            "expected_name": "4.3.1 单元开篇"
        },
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第6周】组合数据类型\6.1 集合类型及操作\6.1.1 单元开篇\6.1.1 单元开篇.xmind",
            "expected_name": "6.1.1 单元开篇"
        },
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第6周】组合数据类型\6.2 序列类型及操作\6.2.1 单元开篇\6.2.1 单元开篇.xmind",
            "expected_name": "6.2.1 单元开篇"
        },
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第6周】组合数据类型\6.4 字典类型及操作\6.4.1 单元开篇\6.4.1 单元开篇.xmind",
            "expected_name": "6.4.1 单元开篇"
        },
        {
            "path": r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学\【第6周】组合数据类型\6.4 字典类型及操作\6.4.1 单元开篇\6.4.1 单元开篇.xmind",
            "expected_name": "6.4.1 单元开篇"
        },
        # 你可以继续在这里添加 4.3.1 的路径
    ]

    for item in missing_files:
        patcher.patch_xmind_file(item["path"], item["expected_name"])

    patcher.close()