from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.chunker import chunk_documents
from app.config import (
    CHUNKS_PATH,
    INDEX_CHUNKS_PATH,
    INDEX_DIR,
    INDEX_MATRIX_PATH,
    INDEX_VECTORIZER_PATH,
)
from scripts.ingest import ingest_documents


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_index() -> tuple[int, int]:
    documents = ingest_documents()
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were produced")

    write_jsonl(CHUNKS_PATH, chunks)
    write_jsonl(INDEX_CHUNKS_PATH, chunks)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        norm="l2",
    )
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, INDEX_VECTORIZER_PATH)
    sparse.save_npz(INDEX_MATRIX_PATH, matrix)
    return matrix.shape


if __name__ == "__main__":
    chunk_count, feature_count = build_index()
    print(f"Indexed matrix: {chunk_count} chunks x {feature_count} features")
