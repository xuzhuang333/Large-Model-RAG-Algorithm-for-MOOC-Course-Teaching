import os
import re
import logging
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

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
        with self.driver.session() as session:
            session.run("MATCH (c:BaselineChunk) DETACH DELETE c")
            session.run("DROP INDEX baseline_chunk_vector_index IF EXISTS")
        logger.info("Database purge completed successfully.")

    def create_vector_index(self):
        """Creates the isolated vector index for Group A."""
        logger.info("Creating baseline_chunk_vector_index...")
        with self.driver.session() as session:
            query = """
            CREATE VECTOR INDEX baseline_chunk_vector_index IF NOT EXISTS
            FOR (c:BaselineChunk)
            ON (c.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 512,
             `vector.similarity_function`: 'cosine'
            }}
            """
            session.run(query)
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

    def process_physical_directory(self, root_dataset_dir, target_filter=None):
        """
        Walks through the physical directory bypassing Neo4j's logical hierarchy.
        This catches ALL .txt, .srt, and .md files (including Level 2 summaries).
        """
        logger.info(f"Scanning physical directory: {root_dataset_dir}")
        if target_filter:
            logger.info(f"FILTER ACTIVE: Only processing paths containing '{target_filter}'")
            
        all_chunks_data = []
        processed_files_count = 0

        for dirpath, dirnames, filenames in os.walk(root_dataset_dir):
            # 1. Apply the target filter (e.g., only "Week 1")
            # This makes the algorithm universal; just pass target_filter=None later to process everything.
            if target_filter and target_filter not in dirpath:
                continue

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
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"  # Please verify your password

    # Update this to your actual root dataset directory
    DATASET_ROOT = r"E:\graduate_project\reference material\Python语言程序设计_北京理工大学"
    
    # Universal Filter Design: Set to None when you want to process ALL weeks.
    TARGET_WEEK_FILTER = "【第1周】Python基本语法元素"

    indexer = BaselineIndexerV3(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        indexer.clear_database_and_index()
        indexer.create_vector_index()
        indexer.process_physical_directory(DATASET_ROOT, target_filter=TARGET_WEEK_FILTER)
        logger.info("🎉 Group A Baseline Indexing (V3) successfully finished!")
    finally:
        indexer.close()