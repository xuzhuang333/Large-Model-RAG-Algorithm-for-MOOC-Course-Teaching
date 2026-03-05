import os
import re
import json
import logging
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 配置日志输出格式（日志内容保持英文以符合规范）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GroupA_Indexer")


class BaselineIndexer:
    def __init__(self, uri, user, password):
        """
        初始化基线索引器。
        连接图数据库并加载用于 Group A 向量化的 BGE 模型。
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Loading BGE model (BAAI/bge-small-zh-v1.5) for Group A...")
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

        # 朴素硬切分（Hard Chunking）的超参数
        self.chunk_size = 500
        self.chunk_overlap = 50
        logger.info("Baseline Indexer initialized.")

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def get_all_source_folders(self):
        """
        从已有的 Concept 节点中提取所有绑定的物理文件夹路径。
        确保 A、B、C 三组实验处理的数据源绝对一致。
        """
        folders = []
        with self.driver.session() as session:
            result = session.run("MATCH (n:Concept) WHERE n.source_folder IS NOT NULL RETURN n.source_folder AS folder")
            for record in result:
                folders.append(record["folder"])
        logger.info(f"Found {len(folders)} folders to process.")
        return list(set(folders))  # 去重

    def clean_srt_text(self, raw_text):
        """
        对 SRT 字幕文件进行基础清洗，去除时间戳和序号，仅保留文本。
        """
        lines = raw_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 跳过空行、纯数字序号行和包含 '-->' 的时间戳行
            if not line or re.match(r'^\d+$', line) or '-->' in line:
                continue
            cleaned_lines.append(line)
        return " ".join(cleaned_lines)

    def parse_json_to_raw_text(self, json_path):
        """
        读取结构化的 JSON 标注文件，但故意将其扁平化为一段毫无层级的纯文本。
        以此模拟传统朴素 RAG 无法理解 PDF 排版的劣势。
        """
        raw_text = ""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 遍历所有页面和文本行，无视 title/text 类型，直接拼接内容
                for page in data:
                    for line_group in page.get("text_content_line_list", []):
                        for item in line_group:
                            raw_text += item.get("content", "") + " "
                    raw_text += "\n"
            logger.info(f"Successfully flattened JSON into raw text: {os.path.basename(json_path)}")
        except Exception as e:
            logger.error(f"Failed to read JSON: {e}")

        return raw_text

    def hard_chunk_text(self, text):
        """
        执行滑动窗口硬切分（Naive sliding window chunking）。
        无视语义边界，按照固定长度暴力截断文本。
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

    def process_and_index_folder(self, folder_path):
        """
        读取指定文件夹下的 TXT、SRT 和 JSON 文件。
        提取纯文本后进行硬切分，并将切分后的数据块存入数据库。
        """
        if not os.path.exists(folder_path):
            logger.warning(f"Path not found: {folder_path}")
            return

        all_chunks_data = []  # 存放字典的列表: {'text': chunk_text, 'source': filename}

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            extracted_text = ""

            try:
                # 处理纯文本教案
                if filename.endswith(".txt"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        extracted_text = f.read()

                # 处理视频字幕
                elif filename.endswith(".srt"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        extracted_text = self.clean_srt_text(f.read())

                # 处理课件 JSON（替代 PDF）
                elif filename.endswith(".json") and "content.json" not in filename:
                    # 排除 XMind 自带的 content.json，只处理课件标注 JSON
                    extracted_text = self.parse_json_to_raw_text(file_path)

            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
                continue

            # 如果提取到了文本，执行硬切分
            if extracted_text.strip():
                file_chunks = self.hard_chunk_text(extracted_text)
                for chunk in file_chunks:
                    all_chunks_data.append({
                        "text": chunk.strip(),
                        "source": os.path.join(os.path.basename(folder_path), filename)
                    })

        # 如果当前文件夹产生了数据块，则执行向量化和入库操作
        if all_chunks_data:
            self.insert_chunks_to_db(all_chunks_data)

    def insert_chunks_to_db(self, chunks_data):
        """
        计算所有数据块的嵌入向量（Embedding），并在 Neo4j 中创建 BaselineChunk 节点。
        """
        texts = [item["text"] for item in chunks_data]
        logger.info(f"Calculating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        with self.driver.session() as session:
            for i, item in enumerate(chunks_data):
                vector_list = [float(x) for x in embeddings[i]]

                # 创建带有 BaselineChunk 标签的孤立节点，不建立任何图谱连线
                query = """
                CREATE (c:BaselineChunk {
                    text: $text,
                    source_file: $source
                })
                WITH c
                CALL db.create.setNodeVectorProperty(c, 'embedding', $vector)
                """
                session.run(query, text=item["text"], source=item["source"], vector=vector_list)

    def create_baseline_index(self):
        """
        为 Group A 的数据块创建专属的向量索引。
        确保与后续实验组的检索逻辑完全物理隔离。
        """
        index_name = "baseline_chunk_vector_index"
        with self.driver.session() as session:
            logger.info(f"Creating vector index: {index_name}")
            query = """
            CREATE VECTOR INDEX baseline_chunk_vector_index IF NOT EXISTS
            FOR (c:BaselineChunk)
            ON (c.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 512,
             `vector.similarity_function`: 'cosine'
            }}
            """
            try:
                session.run(query)
                logger.info("Baseline vector index ready.")
            except Exception as e:
                logger.error(f"Failed to create index: {e}")


if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"  # 请在这里替换为您的实际数据库密码

    indexer = BaselineIndexer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 第一步：确保向量索引优先建立
        indexer.create_baseline_index()

        # 第二步：获取所有目标文件夹路径
        folders = indexer.get_all_source_folders()

        # 第三步：逐个文件夹处理，进行扁平化读取、切分、向量化和入库
        for i, folder in enumerate(folders):
            logger.info(f"Processing folder {i + 1}/{len(folders)}: {os.path.basename(folder)}")
            indexer.process_and_index_folder(folder)

        logger.info("Group A Baseline Indexing Complete!")

    finally:
        indexer.close()