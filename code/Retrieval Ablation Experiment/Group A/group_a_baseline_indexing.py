import os
import re
import sys
import logging
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

#这个代码文件用于Group A 的Baseline Indexing，负责从物理目录中提取文本、进行粗暴的硬切分，并将结果存入Neo4j中。这个版本的切分完全不考虑语义边界，旨在展示Baseline方法的局限性。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from shared_retrieval_utils import GROUP_RESOURCE_NAMES, load_runtime_config

# English logs and comments as explicitly requested
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroupA_Indexer_V3_Physical_Walk")

class BaselineIndexerV3:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Loading BGE model (BAAI/bge-small-zh-v1.5) for Baseline embedding...")
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        
        # Brutal Hard Chunking Parameters (To demonstrate Baseline flaws)
        self.chunk_size = 500
        self.chunk_overlap = 50
        logger.info(f"Initialized with Hard Chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def close(self):
        self.driver.close()

    def clear_database_and_index(self):
        """
        Wipes out ALL old BaselineChunk nodes and drops the vector index.
        Ensures a completely clean slate for the new experiment.
        """
        logger.info("Purging all existing BaselineChunk data from Neo4j...")
        index_name = GROUP_RESOURCE_NAMES["A"]["index_name"]
        with self.driver.session() as session:
            session.run("MATCH (c:BaselineChunk) DETACH DELETE c")
            session.run(f"DROP INDEX {index_name} IF EXISTS")
        logger.info("Database purge completed successfully.")

    def create_vector_index(self):
        """Creates the isolated vector index for Group A."""
        index_name = GROUP_RESOURCE_NAMES["A"]["index_name"]
        logger.info(f"Creating {index_name}...")
        with self.driver.session() as session:
            query = """
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (c:BaselineChunk)
            ON (c.embedding)
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: 512,
             `vector.similarity_function`: 'cosine'
            }}}}
            """
            session.run(query.format(index_name=index_name))
        logger.info("Vector index created.")

    def clean_srt_text(self, raw_text):
        """Removes timestamps and indices from SRT files."""
        cleaned_lines = []
        for line in raw_text.split('\n'):
            line = line.strip()
            if not line or re.match(r'^\d+$', line) or '-->' in line:
                continue
            cleaned_lines.append(line)
        return " ".join(cleaned_lines)

    def brutal_hard_chunk(self, text):
        """
        Slices the text strictly by character count.
        Intentionally ignores Markdown boundaries, tables, and LaTeX formulas.
        """
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

    def process_physical_directory(self, root_dataset_dir):
        """
        Walks through the physical directory bypassing Neo4j's logical hierarchy.
        This catches ALL .txt, .srt, and .md files (including Level 2 summaries).
        """
        logger.info(f"Scanning physical directory: {root_dataset_dir}")
            
        all_chunks_data = []
        processed_files_count = 0

        for dirpath, dirnames, filenames in os.walk(root_dataset_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                extracted_text = ""

                try:
                    # We strictly process only these three file types
                    if filename.endswith(".txt"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            extracted_text = f.read()
                    elif filename.endswith(".srt"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            extracted_text = self.clean_srt_text(f.read())
                    elif filename.endswith(".md"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            extracted_text = f.read()
                    else:
                        continue # Ignore other files (.pdf, .xmind, etc.)

                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    continue

                if extracted_text.strip():
                    processed_files_count += 1
                    # 2. Brutal chunking
                    file_chunks = self.brutal_hard_chunk(extracted_text)
                    
                    # Store relative path as source for traceability
                    rel_path = os.path.relpath(file_path, root_dataset_dir)
                    for chunk in file_chunks:
                        all_chunks_data.append({
                            "text": chunk.strip(),
                            "source": rel_path
                        })

        logger.info(f"Extraction complete. Processed {processed_files_count} files, generating {len(all_chunks_data)} raw chunks.")
        
        # 3. Batch insert to Neo4j
        if all_chunks_data:
            self.insert_chunks_to_db(all_chunks_data)

    def insert_chunks_to_db(self, chunks_data):
        """Embeds text and inserts BaselineChunk nodes into Neo4j."""
        logger.info("Computing embeddings and inserting into Neo4j...")
        texts = [item["text"] for item in chunks_data]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        with self.driver.session() as session:
            for i, item in enumerate(chunks_data):
                vector_list = [float(x) for x in embeddings[i]]
                query = """
                CREATE (c:BaselineChunk {
                    text: $text,
                    source_file: $source
                })
                WITH c
                CALL db.create.setNodeVectorProperty(c, 'embedding', $vector)
                """
                session.run(query, text=item["text"], source=item["source"], vector=vector_list)
        logger.info("Insertion complete.")

if __name__ == "__main__":
    runtime_cfg = load_runtime_config(
        default_uri="bolt://localhost:7687",
        default_user="neo4j",
        default_password="12345678",
    )

    # Update this to your actual root dataset directory
    DATASET_ROOT = os.getenv(
        "DATASET_ROOT",
        r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学",
    )
    
    indexer = BaselineIndexerV3(
        runtime_cfg.neo4j_uri,
        runtime_cfg.neo4j_user,
        runtime_cfg.neo4j_password,
    )
    
    try:
        indexer.clear_database_and_index()
        indexer.create_vector_index()
        indexer.process_physical_directory(DATASET_ROOT)
        logger.info("🎉 Group A Baseline Indexing (V3) successfully finished!")
    finally:
        indexer.close()