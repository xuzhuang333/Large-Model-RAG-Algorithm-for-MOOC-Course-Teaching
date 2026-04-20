from __future__ import annotations

import re

from neo4j import GraphDatabase
#这个文件定义了Group B的Leaf Repository类，封装了与Neo4j数据库交互的细节，包括节点的插入、查询和索引管理等操作。这个类是Group B中处理底层数据存储和访问的核心组件，为上层的树构建和摘要生成提供支持。

class GroupBLeafRepository:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        node_label: str,
        index_name: str,
        embedding_dim: int = 512,
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_label = node_label
        self.index_name = index_name
        self.embedding_dim = embedding_dim

    def close(self) -> None:
        self.driver.close()

    def clear_group_b_nodes_and_index(self) -> None:
        with self.driver.session() as session:
            session.run(f"MATCH (n:{self.node_label}) DETACH DELETE n")
            session.run(f"DROP INDEX {self.index_name} IF EXISTS")

    def create_vector_index(self) -> None:
        query = """
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{node_label})
        ON (n.embedding)
        OPTIONS {{indexConfig: {{
         `vector.dimensions`: {embedding_dim},
         `vector.similarity_function`: 'cosine'
        }}}}
        """
        with self.driver.session() as session:
            session.run(
                query.format(
                    index_name=self.index_name,
                    node_label=self.node_label,
                    embedding_dim=self.embedding_dim,
                )
            )

    def insert_leaf_nodes(self, rows: list[dict]) -> int:
        if not rows:
            return 0

        query = f"""
        UNWIND $rows AS row
        MERGE (n:{self.node_label} {{node_id: row.node_id}})
        ON CREATE SET n.created_at = datetime()
        SET
            n.text = row.text,
            n.layer = 0,
            n.source_file = row.source_file,
            n.source_type = row.source_type,
            n.chunk_order = row.chunk_order,
            n.token_count = row.token_count,
            n.sentence_count = row.sentence_count,
            n.updated_at = datetime()
        WITH n, row
        CALL db.create.setNodeVectorProperty(n, 'embedding', row.embedding)
        RETURN count(DISTINCT n) AS inserted
        """
        with self.driver.session() as session:
            result = session.run(query, rows=rows)
            record = result.single()
            return int(record["inserted"]) if record else 0

    def fetch_nodes_by_layer(self, layer: int) -> list[dict]:
        query = f"""
        MATCH (n:{self.node_label})
        WHERE n.layer = $layer AND n.embedding IS NOT NULL
        RETURN
            n.node_id AS node_id,
            n.text AS text,
            n.layer AS layer,
            coalesce(n.token_count, 0) AS token_count,
            n.embedding AS embedding
        ORDER BY n.node_id ASC
        """
        with self.driver.session() as session:
            result = session.run(query, layer=layer)
            rows: list[dict] = []
            for record in result:
                rows.append(
                    {
                        "node_id": record["node_id"],
                        "text": record["text"],
                        "layer": int(record["layer"]),
                        "token_count": int(record["token_count"]),
                        "embedding": [float(x) for x in record["embedding"]],
                    }
                )
            return rows

    def insert_summary_nodes(
        self,
        rows: list[dict],
        relationship_type: str = "GROUP_B_PARENT_OF",
    ) -> int:
        if not rows:
            return 0

        rel = self._sanitize_relationship_type(relationship_type)
        query = f"""
        UNWIND $rows AS row
        MERGE (p:{self.node_label} {{node_id: row.node_id}})
        ON CREATE SET p.created_at = datetime()
        SET
            p.text = row.text,
            p.layer = row.layer,
            p.source_file = 'cluster_summary',
            p.source_type = 'summary',
            p.chunk_order = row.cluster_order,
            p.token_count = row.token_count,
            p.sentence_count = 1,
            p.child_count = row.child_count,
            p.updated_at = datetime()
        WITH p, row
        CALL db.create.setNodeVectorProperty(p, 'embedding', row.embedding)
        WITH p, row
        UNWIND row.child_node_ids AS child_id
        MATCH (c:{self.node_label} {{node_id: child_id}})
        MERGE (p)-[:{rel}]->(c)
        RETURN count(DISTINCT p) AS inserted
        """
        with self.driver.session() as session:
            result = session.run(query, rows=rows)
            record = result.single()
            return int(record["inserted"]) if record else 0

    @staticmethod
    def _sanitize_relationship_type(relationship_type: str) -> str:
        candidate = (relationship_type or "").strip().upper()
        if not candidate:
            return "GROUP_B_PARENT_OF"
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", candidate):
            return candidate
        return "GROUP_B_PARENT_OF"