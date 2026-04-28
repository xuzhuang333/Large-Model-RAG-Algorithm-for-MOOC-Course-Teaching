import os
import logging
from neo4j import GraphDatabase

# Configure logging (English as requested)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Retrieval_Test")


class KnowledgeRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_node_context(self, node_name):
        """
        Step 1: Search Neo4j for the node and get its folder path.
        Step 2: Access the file system to list relevant files.
        """
        folder_path = None

        # --- 1. Query Neo4j ---
        with self.driver.session() as session:
            # Cypher query to find the path based on node name
            query = """
            MATCH (n:Concept {name: $name})
            RETURN n.source_folder AS path
            LIMIT 1
            """
            result = session.run(query, name=node_name).single()

            if result and result["path"]:
                folder_path = result["path"]
                logger.info(f"✅ Found node '{node_name}' linked to path: {folder_path}")
            else:
                logger.warning(f"❌ Node '{node_name}' not found or has no path.")
                return

        # --- 2. Access File System (The 'Augmentation' part of RAG) ---
        if folder_path and os.path.exists(folder_path):
            logger.info(f"📂 Accessing directory: {folder_path}")

            # List all relevant files for RAG (txt, pdf)
            files = os.listdir(folder_path)
            found_data = False

            for f in files:
                full_path = os.path.join(folder_path, f)

                # Case A: Text file - we can read it directly as a preview
                if f.endswith(".txt"):
                    found_data = True
                    logger.info(f"📄 Found Text Context: {f}")
                    try:
                        with open(full_path, 'r', encoding='utf-8') as text_file:
                            content_preview = text_file.read()[:200]  # Read first 200 chars
                            print(
                                f"\n--- [Content Preview: {f}] ---\n{content_preview}...\n-----------------------------\n")
                    except Exception as e:
                        logger.error(f"Could not read text file: {e}")

                # Case B: PDF file - just log existence for now (parsing comes later)
                elif f.endswith(".pdf"):
                    found_data = True
                    logger.info(f"📚 Found PDF Context: {f} (Ready for parsing)")

            if not found_data:
                logger.warning("Directory exists but contains no .txt or .pdf files for context.")
        else:
            logger.error(f"Physical path does not exist on disk: {folder_path}")


# --- Execution ---
if __name__ == "__main__":
    # Update with your Neo4j credentials
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "YOUR VALUE"  # 替换为 YOUR VALUE

    retriever = KnowledgeRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Simulation: User inputs a specific concept name
    # Based on your screenshots, "1.1.1 单元开篇" is a valid node
    target_node = "程序设计基本方法"

    print(f"🔍 Simulating search for: '{target_node}'")
    retriever.get_node_context(target_node)

    retriever.close()