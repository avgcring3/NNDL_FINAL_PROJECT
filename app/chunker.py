from __future__ import annotations

import re
from typing import Iterable

from app.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [re.sub(r"[ \t]+", " ", part).strip() for part in parts if part.strip()]


def _split_long_unit(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    step = max(1, max_chars - max(0, overlap))
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def _overlap_suffix(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    suffix = text[-overlap:].strip()
    first_space = suffix.find(" ")
    if first_space > 0:
        suffix = suffix[first_space + 1 :].strip()
    return suffix


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into paragraph-aware chunks capped by max_chars."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[str] = []
    current = ""

    for paragraph in _paragraphs(text):
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_unit(paragraph, max_chars, overlap))
            continue

        if not current:
            current = paragraph
            continue

        candidate = f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current.strip())
        suffix = _overlap_suffix(current, overlap)
        candidate = f"{suffix}\n\n{paragraph}" if suffix else paragraph
        current = candidate if len(candidate) <= max_chars else paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_documents(
    documents: Iterable[dict],
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    chunked: list[dict] = []
    for document in documents:
        doc_id = document["doc_id"]
        for index, text in enumerate(chunk_text(document["text"], max_chars, overlap), start=1):
            chunked.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{index:03d}",
                    "doc_id": doc_id,
                    "name": document["name"],
                    "source_file": document["source_file"],
                    "text": text,
                }
            )
    return chunked
