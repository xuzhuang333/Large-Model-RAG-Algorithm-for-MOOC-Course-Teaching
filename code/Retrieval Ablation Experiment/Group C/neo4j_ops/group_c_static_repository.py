from __future__ import annotations

import re
from typing import Any

from neo4j import GraphDatabase


class GroupCStaticRepository:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        syllabus_label: str = "GroupC_SyllabusNode",
        text_label: str = "GroupC_TextSnippet",
        code_label: str = "GroupC_CodeSnippet",
        text_index_name: str = "group_c_text_vector_index",
        code_index_name: str = "group_c_code_vector_index",
        embedding_dim: int = 512,
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.syllabus_label = syllabus_label
        self.text_label = text_label
        self.code_label = code_label
        self.text_index_name = text_index_name
        self.code_index_name = code_index_name
        self.embedding_dim = embedding_dim

    def close(self) -> None:
        self.driver.close()

    def clear_group_c_graph(self) -> None:
        with self.driver.session() as session:
            session.run(f"MATCH (n:{self.code_label}) DETACH DELETE n")
            session.run(f"MATCH (n:{self.text_label}) DETACH DELETE n")
            session.run(f"MATCH (n:{self.syllabus_label}) DETACH DELETE n")
            session.run(f"DROP INDEX {self.text_index_name} IF EXISTS")
            session.run(f"DROP INDEX {self.code_index_name} IF EXISTS")

    def create_vector_indexes(self) -> None:
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
                    index_name=self.text_index_name,
                    node_label=self.text_label,
                    embedding_dim=self.embedding_dim,
                )
            )
            session.run(
                query.format(
                    index_name=self.code_index_name,
                    node_label=self.code_label,
                    embedding_dim=self.embedding_dim,
                )
            )

    def upsert_syllabus_nodes(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        query = f"""
        UNWIND $rows AS row
        MERGE (s:{self.syllabus_label} {{node_id: row.node_id}})
        ON CREATE SET s.created_at = datetime()
        SET
            s.name = row.name,
            s.abs_path = row.abs_path,
            s.parent_abs_path = row.parent_abs_path,
            s.depth = row.depth,
            s.week_tag = row.week_tag,
            s.updated_at = datetime()
        RETURN count(DISTINCT s) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"]) if record else 0

    def connect_syllabus_hierarchy(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        query = f"""
        UNWIND $rows AS row
        WITH row WHERE row.parent_node_id IS NOT NULL
        MATCH (p:{self.syllabus_label} {{node_id: row.parent_node_id}})
        MATCH (c:{self.syllabus_label} {{node_id: row.node_id}})
        MERGE (p)-[:HAS_SUBTOPIC]->(c)
        RETURN count(*) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"]) if record else 0

    def upsert_text_snippets(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        query = f"""
        UNWIND $rows AS row
        MERGE (t:{self.text_label} {{snippet_id: row.snippet_id}})
        ON CREATE SET t.created_at = datetime()
        SET
            t.syllabus_node_id = row.syllabus_node_id,
            t.text = row.text,
            t.context_prefix = row.context_prefix,
            t.chunk_type = row.chunk_type,
            t.chunk_title = row.chunk_title,
            t.section_name = row.section_name,
            t.section_order = row.section_order,
            t.chunk_order = row.chunk_order,
            t.course_material_title = row.course_material_title,
            t.metadata_source_file = row.metadata_source_file,
            t.metadata_instructor = row.metadata_instructor,
            t.metadata_document_type = row.metadata_document_type,
            t.metadata_core_keywords = row.metadata_core_keywords,
            t.source_file = row.source_file,
            t.source_type = row.source_type,
            t.trace = row.trace,
            t.token_count = row.token_count,
            t.summary_level = row.summary_level,
            t.is_generated_summary = row.is_generated_summary,
            t.updated_at = datetime()
        WITH t, row
        CALL db.create.setNodeVectorProperty(t, 'embedding', row.embedding)
        WITH t, row
        MATCH (s:{self.syllabus_label} {{node_id: row.syllabus_node_id}})
        MERGE (s)-[:HAS_TEXT]->(t)
        RETURN count(DISTINCT t) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"]) if record else 0

    def upsert_code_snippets(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        query = f"""
        UNWIND $rows AS row
        MERGE (c:{self.code_label} {{snippet_id: row.snippet_id}})
        ON CREATE SET c.created_at = datetime()
        SET
            c.syllabus_node_id = row.syllabus_node_id,
            c.code = row.code,
            c.context_prefix = row.context_prefix,
            c.chunk_type = row.chunk_type,
            c.chunk_title = row.chunk_title,
            c.section_name = row.section_name,
            c.section_order = row.section_order,
            c.chunk_order = row.chunk_order,
            c.course_material_title = row.course_material_title,
            c.metadata_source_file = row.metadata_source_file,
            c.metadata_instructor = row.metadata_instructor,
            c.metadata_document_type = row.metadata_document_type,
            c.metadata_core_keywords = row.metadata_core_keywords,
            c.source_file = row.source_file,
            c.source_type = row.source_type,
            c.trace = row.trace,
            c.token_count = row.token_count,
            c.updated_at = datetime()
        WITH c, row
        CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
        WITH c, row
        MATCH (s:{self.syllabus_label} {{node_id: row.syllabus_node_id}})
        MERGE (s)-[:HAS_CODE]->(c)
        RETURN count(DISTINCT c) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"]) if record else 0

    def clear_generated_summaries(self) -> int:
        count_query = f"""
        MATCH (t:{self.text_label})
        WHERE coalesce(t.is_generated_summary, false) = true
        RETURN count(t) AS count
        """
        delete_query = f"""
        MATCH (t:{self.text_label})
        WHERE coalesce(t.is_generated_summary, false) = true
        DETACH DELETE t
        """
        with self.driver.session() as session:
            record = session.run(count_query).single()
            deleted = int(record["count"]) if record else 0
            if deleted > 0:
                session.run(delete_query)
            return deleted

    def clear_prerequisite_edges(self) -> int:
        count_query = f"""
        MATCH (:{self.syllabus_label})-[r:REQUIRES_PREREQUISITE]->(:{self.syllabus_label})
        RETURN count(r) AS count
        """
        delete_query = f"""
        MATCH (:{self.syllabus_label})-[r:REQUIRES_PREREQUISITE]->(:{self.syllabus_label})
        DELETE r
        """
        with self.driver.session() as session:
            record = session.run(count_query).single()
            deleted = int(record["count"]) if record else 0
            if deleted > 0:
                session.run(delete_query)
            return deleted

    def fetch_syllabus_nodes_depth_desc(self) -> list[dict[str, Any]]:
        query = f"""
        MATCH (s:{self.syllabus_label})
        RETURN s.node_id AS node_id, s.name AS name, s.abs_path AS abs_path, s.depth AS depth, s.week_tag AS week_tag
        ORDER BY s.depth DESC, s.abs_path ASC
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [
                {
                    "node_id": r["node_id"],
                    "name": r["name"],
                    "abs_path": r["abs_path"],
                    "depth": int(r["depth"]),
                    "week_tag": r["week_tag"],
                }
                for r in result
            ]

    def fetch_direct_texts(self, node_id: str, include_generated: bool = True) -> list[dict[str, Any]]:
        where = "" if include_generated else "AND coalesce(t.is_generated_summary, false) = false"
        query = f"""
        MATCH (s:{self.syllabus_label} {{node_id: $node_id}})-[:HAS_TEXT]->(t:{self.text_label})
        WHERE 1=1 {where}
        RETURN t.snippet_id AS snippet_id, t.text AS text, coalesce(t.is_generated_summary, false) AS is_generated_summary,
               coalesce(t.summary_level, -1) AS summary_level
        ORDER BY coalesce(t.summary_level, -1) DESC, t.chunk_order ASC
        """
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            rows: list[dict[str, Any]] = []
            for r in result:
                rows.append(
                    {
                        "snippet_id": r["snippet_id"],
                        "text": r["text"],
                        "is_generated_summary": bool(r["is_generated_summary"]),
                        "summary_level": int(r["summary_level"]),
                    }
                )
            return rows

    def fetch_children(self, node_id: str) -> list[dict[str, Any]]:
        query = f"""
        MATCH (p:{self.syllabus_label} {{node_id: $node_id}})-[:HAS_SUBTOPIC]->(c:{self.syllabus_label})
        RETURN c.node_id AS node_id, c.name AS name, c.depth AS depth
        ORDER BY c.depth DESC, c.name ASC
        """
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            return [
                {"node_id": r["node_id"], "name": r["name"], "depth": int(r["depth"])}
                for r in result
            ]

    def propagate_code_links_from_children(self, node_id: str) -> int:
        query = f"""
        MATCH (p:{self.syllabus_label} {{node_id: $node_id}})-[:HAS_SUBTOPIC]->(c:{self.syllabus_label})-[:HAS_CODE]->(code:{self.code_label})
        MERGE (p)-[:HAS_CODE]->(code)
        RETURN count(*) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, node_id=node_id).single()
            return int(record["count"]) if record else 0

    def fetch_weekly_summary_nodes(self) -> list[dict[str, Any]]:
        query = f"""
        MATCH (s:{self.syllabus_label})-[:HAS_TEXT]->(t:{self.text_label})
        WHERE coalesce(t.is_generated_summary, false) = true AND t.text IS NOT NULL AND trim(t.text) <> ''
        RETURN s.node_id AS node_id, s.name AS name, s.week_tag AS week_tag, s.abs_path AS abs_path, t.text AS summary_text
        ORDER BY s.week_tag ASC, s.depth ASC, s.name ASC
        """
        with self.driver.session() as session:
            result = session.run(query)
            rows: list[dict[str, Any]] = []
            for r in result:
                rows.append(
                    {
                        "node_id": r["node_id"],
                        "name": r["name"],
                        "week_tag": r["week_tag"],
                        "abs_path": r["abs_path"],
                        "summary_text": r["summary_text"],
                    }
                )
            return rows

    def upsert_prerequisite_edges(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0

        query = f"""
        UNWIND $rows AS row
        MATCH (a:{self.syllabus_label} {{node_id: row.prerequisite_node_id}})
        MATCH (b:{self.syllabus_label} {{node_id: row.target_node_id}})
        MERGE (a)-[r:REQUIRES_PREREQUISITE]->(b)
        SET r.reason = row.reason,
            r.confidence = row.confidence,
            r.updated_at = datetime()
        RETURN count(*) AS count
        """
        with self.driver.session() as session:
            record = session.run(query, rows=rows).single()
            return int(record["count"]) if record else 0

    @staticmethod
    def sanitize_week_tag(abs_path: str) -> str | None:
        match = re.search(r"【第\d+周】", abs_path)
        return match.group(0) if match else None
