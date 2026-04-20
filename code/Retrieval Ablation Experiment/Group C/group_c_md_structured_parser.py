#该文件用于解析结构化的Markdown文档，提取课程材料标题、元数据和内容片段（如文本块、代码块、比较表等）。解析后的数据结构便于后续处理和分析。
from __future__ import annotations

import re
from dataclasses import dataclass

COURSE_TITLE_RE = re.compile(r"^(?:#\s*)?\[Course_Material\]\s*(.+?)\s*$")
METADATA_HEADER_RE = re.compile(r"^(?:##\s*)?\[Metadata\]\s*$")
SECTION_RE = re.compile(r"^(?:##\s*)?\[Section:\s*(.+?)\]\s*$")
CHUNK_RE = re.compile(
    r"^(?:###\s*)?\[(Chunk|Comparison_Table|Code_Snippet|Concept_Deconstruction|Procedure):\s*(.+?)\]\s*$"
)
KEY_VALUE_RE = re.compile(r"^-\s*(?:\[cite_start\])?\*\*(.+?)\*\*:\s*(.+)$")
CITATION_RE = re.compile(r"\[cite:\s*([^\]]+)\]")
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedSnippet:
    snippet_kind: str  # text | code
    chunk_type: str
    chunk_title: str
    section_name: str
    section_order: int
    chunk_order: int
    context_prefix: str
    text_content: str
    code_content: str
    raw_markdown: str
    citations: list[str]


@dataclass
class ParsedMarkdownDocument:
    course_material_title: str
    metadata: dict[str, str]
    snippets: list[ParsedSnippet]


def _normalize_line_for_text(line: str) -> str:
    cleaned = line.replace("[cite_start]", "").strip()
    return cleaned


def _canonical_tag_line(line: str) -> str:
    stripped = line.lstrip()
    # Support list-prefix forms such as "- - [Course_Material] ..."
    stripped = re.sub(r"^(?:-\s*)+", "", stripped)
    return stripped.strip()


def _extract_citations(text: str) -> list[str]:
    citations: list[str] = []
    for match in CITATION_RE.findall(text):
        norm = match.strip()
        if norm and norm not in citations:
            citations.append(norm)
    return citations


def _parse_metadata(lines: list[str], start_idx: int) -> tuple[dict[str, str], int]:
    metadata: dict[str, str] = {}
    idx = start_idx

    while idx < len(lines):
        line = lines[idx].rstrip("\n")
        canonical = _canonical_tag_line(line)
        if SECTION_RE.match(canonical) or CHUNK_RE.match(canonical) or COURSE_TITLE_RE.match(canonical):
            break
        if canonical.startswith("## ") and not METADATA_HEADER_RE.match(canonical):
            break

        kv_match = KEY_VALUE_RE.match(line.strip())
        if kv_match:
            key = kv_match.group(1).strip().lower().replace(" ", "_")
            value = _normalize_line_for_text(kv_match.group(2))
            metadata[key] = value

        idx += 1

    return metadata, idx


def _parse_table_rows(body_lines: list[str]) -> str:
    table_lines = [ln.rstrip("\n") for ln in body_lines if "|" in ln]
    if len(table_lines) < 2:
        return ""

    header_cells = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    parsed_rows: list[str] = []

    for row_line in table_lines[2:]:
        row_cells = [cell.strip() for cell in row_line.strip().strip("|").split("|")]
        if len(row_cells) != len(header_cells):
            continue
        row_repr = " ; ".join(
            f"{header_cells[i]}={row_cells[i]}" for i in range(len(header_cells))
        )
        parsed_rows.append(row_repr)

    return "\n".join(parsed_rows)


def parse_structured_markdown(markdown_text: str) -> ParsedMarkdownDocument:
    lines = markdown_text.splitlines()
    course_material_title = ""
    metadata: dict[str, str] = {}
    snippets: list[ParsedSnippet] = []

    idx = 0
    section_name = "UNSPECIFIED_SECTION"
    section_order = 0
    chunk_order = 0

    while idx < len(lines):
        line = lines[idx].rstrip("\n")
        canonical = _canonical_tag_line(line)

        course_match = COURSE_TITLE_RE.match(canonical)
        if course_match:
            course_material_title = course_match.group(1).strip()
            idx += 1
            continue

        if METADATA_HEADER_RE.match(canonical):
            parsed_metadata, next_idx = _parse_metadata(lines, idx + 1)
            metadata.update(parsed_metadata)
            idx = next_idx
            continue

        section_match = SECTION_RE.match(canonical)
        if section_match:
            section_order += 1
            section_name = section_match.group(1).strip()
            idx += 1
            continue

        chunk_match = CHUNK_RE.match(canonical)
        if chunk_match:
            chunk_order += 1
            raw_chunk_type = chunk_match.group(1).strip()
            chunk_title = chunk_match.group(2).strip()
            chunk_type = raw_chunk_type.lower()

            body_lines: list[str] = []
            idx += 1
            while idx < len(lines):
                lookahead = _canonical_tag_line(lines[idx].rstrip("\n"))
                if CHUNK_RE.match(lookahead) or SECTION_RE.match(lookahead):
                    break
                body_lines.append(lines[idx])
                idx += 1

            raw_body = "\n".join(body_lines).strip()
            citations = _extract_citations(raw_body)
            context_prefix = f"[Section: {section_name}] -> [{raw_chunk_type}: {chunk_title}]"

            snippet_kind = "code" if raw_chunk_type == "Code_Snippet" else "text"
            text_content = ""
            code_content = ""

            if snippet_kind == "code":
                code_blocks = [blk.strip() for blk in CODE_FENCE_RE.findall(raw_body) if blk.strip()]
                code_content = "\n\n".join(code_blocks) if code_blocks else raw_body
            else:
                normalized_lines = [_normalize_line_for_text(x) for x in body_lines if x.strip()]
                text_content = "\n".join(normalized_lines).strip()
                if raw_chunk_type == "Comparison_Table":
                    table_repr = _parse_table_rows(body_lines)
                    if table_repr:
                        text_content = f"{text_content}\n\n[Table Rows]\n{table_repr}".strip()

            snippets.append(
                ParsedSnippet(
                    snippet_kind=snippet_kind,
                    chunk_type=chunk_type,
                    chunk_title=chunk_title,
                    section_name=section_name,
                    section_order=section_order,
                    chunk_order=chunk_order,
                    context_prefix=context_prefix,
                    text_content=text_content,
                    code_content=code_content,
                    raw_markdown=raw_body,
                    citations=citations,
                )
            )
            continue

        idx += 1

    return ParsedMarkdownDocument(
        course_material_title=course_material_title,
        metadata=metadata,
        snippets=snippets,
    )
