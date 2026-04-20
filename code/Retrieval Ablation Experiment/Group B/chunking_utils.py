from __future__ import annotations

import logging
import re
from functools import lru_cache

from transformers import AutoTokenizer

#这个文件定义了Group B中与文本切分和Token计数相关的工具函数和类，主要用于将文本切分成适合模型输入的块，并统计文本的Token数量。这些工具在Group B的树构建和摘要生成过程中被广泛使用，以确保输入文本符合模型的上下文长度限制，同时尽可能保持语义完整性。

logger = logging.getLogger("GroupB_Chunking")

DEFAULT_BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_MODEL_MAX_TOKENS = 512
DEFAULT_EXTREME_FORCE_SPLIT_TOKEN_CAP = 400


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    logger.info("Loading tokenizer from model: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


class TokenCounter:
    """Token counter backed by the embedding model tokenizer."""

    def __init__(self, model_name: str = DEFAULT_BGE_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = _load_tokenizer(model_name)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids)

    def get_model_token_limit(self, fallback: int = BGE_MODEL_MAX_TOKENS) -> int:
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        try:
            value = int(model_max_length) if model_max_length is not None else fallback
        except Exception:
            value = fallback

        # Some tokenizers expose a huge sentinel value instead of real context length.
        if value <= 0 or value > 100000:
            return fallback
        return value

    def force_split_by_token_cap(self, text: str, token_cap: int) -> list[str]:
        if not text:
            return []

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= token_cap:
            return [text]

        parts: list[str] = []
        for start in range(0, len(token_ids), token_cap):
            sub_ids = token_ids[start : start + token_cap]
            sub_text = self.tokenizer.decode(
                sub_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if sub_text:
                parts.append(sub_text)

        return parts


def split_into_sentences(text: str) -> list[str]:
    compact = text.replace("\r\n", "\n").replace("\r", "\n")
    compact = re.sub(r"\n{2,}", "\n", compact)
    compact = re.sub(r"[ \t]+", " ", compact)

    sentences: list[str] = []
    buffer: list[str] = []
    breakers = {"。", "！", "？", "!", "?", "；", ";", "\n"}

    for idx, ch in enumerate(compact):
        buffer.append(ch)

        if ch == ".":
            next_char = compact[idx + 1] if idx + 1 < len(compact) else ""
            if next_char.isspace() or next_char == "":
                sentence = "".join(buffer).strip()
                if sentence:
                    sentences.append(sentence)
                buffer = []
            continue

        if ch in breakers:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)

    return sentences


def build_sentence_complete_chunks(
    text: str,
    token_limit: int,
    token_counter: TokenCounter,
    hard_token_cap: int = DEFAULT_EXTREME_FORCE_SPLIT_TOKEN_CAP,
    audit_stats: dict[str, int] | None = None,
) -> list[dict]:
    """
    Build chunks with sentence integrity.
    If a single sentence exceeds token_limit, keep it intact as one chunk.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[dict] = []
    current_sentences: list[str] = []
    current_tokens = 0

    model_limit = token_counter.get_model_token_limit(fallback=BGE_MODEL_MAX_TOKENS)
    effective_hard_cap = min(hard_token_cap, model_limit)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sent_tokens = token_counter.count_tokens(sentence)

        # Keep normal over-limit sentences intact, but enforce a hard split for extreme lengths.
        if sent_tokens > effective_hard_cap:
            if audit_stats is not None:
                audit_stats["extreme_forced_split_sentences"] = (
                    audit_stats.get("extreme_forced_split_sentences", 0) + 1
                )

            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(
                    {
                        "text": chunk_text,
                        "token_count": current_tokens,
                        "sentence_count": len(current_sentences),
                    }
                )
                current_sentences = []
                current_tokens = 0

            forced_parts = token_counter.force_split_by_token_cap(sentence, effective_hard_cap)
            if audit_stats is not None:
                audit_stats["extreme_forced_split_subchunks"] = (
                    audit_stats.get("extreme_forced_split_subchunks", 0) + len(forced_parts)
                )
            for part in forced_parts:
                part_tokens = token_counter.count_tokens(part)
                chunks.append(
                    {
                        "text": part,
                        "token_count": part_tokens,
                        "sentence_count": 1,
                    }
                )
            continue

        if sent_tokens > token_limit:
            if current_sentences:
                chunk_text = " ".join(current_sentences).strip()
                chunks.append(
                    {
                        "text": chunk_text,
                        "token_count": current_tokens,
                        "sentence_count": len(current_sentences),
                    }
                )
                current_sentences = []
                current_tokens = 0

            chunks.append(
                {
                    "text": sentence,
                    "token_count": sent_tokens,
                    "sentence_count": 1,
                }
            )
            continue

        if not current_sentences:
            current_sentences = [sentence]
            current_tokens = sent_tokens
            continue

        if current_tokens + sent_tokens <= token_limit:
            current_sentences.append(sentence)
            current_tokens += sent_tokens
            continue

        chunk_text = " ".join(current_sentences).strip()
        chunks.append(
            {
                "text": chunk_text,
                "token_count": current_tokens,
                "sentence_count": len(current_sentences),
            }
        )
        current_sentences = [sentence]
        current_tokens = sent_tokens

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append(
            {
                "text": chunk_text,
                "token_count": current_tokens,
                "sentence_count": len(current_sentences),
            }
        )

    return chunks