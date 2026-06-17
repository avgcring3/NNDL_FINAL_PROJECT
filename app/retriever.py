from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from app.config import INDEX_CHUNKS_PATH, INDEX_MATRIX_PATH, INDEX_VECTORIZER_PATH


@dataclass(frozen=True)
class RagIndex:
    vectorizer: object
    matrix: sparse.spmatrix
    chunks: list[dict]


def index_exists(index_dir: Path | None = None) -> bool:
    if index_dir is None:
        return (
            INDEX_VECTORIZER_PATH.exists()
            and INDEX_MATRIX_PATH.exists()
            and INDEX_CHUNKS_PATH.exists()
        )
    return (
        (index_dir / "vectorizer.pkl").exists()
        and (index_dir / "matrix.npz").exists()
        and (index_dir / "chunks.jsonl").exists()
    )


def load_index(index_dir: Path | None = None) -> RagIndex:
    vectorizer_path = INDEX_VECTORIZER_PATH
    matrix_path = INDEX_MATRIX_PATH
    chunks_path = INDEX_CHUNKS_PATH
    if index_dir is not None:
        vectorizer_path = index_dir / "vectorizer.pkl"
        matrix_path = index_dir / "matrix.npz"
        chunks_path = index_dir / "chunks.jsonl"

    vectorizer = joblib.load(vectorizer_path)
    matrix = sparse.load_npz(matrix_path)
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def retrieve(query: str, index: RagIndex | None = None, top_k: int = 5) -> list[dict]:
    if not query.strip():
        return []

    rag_index = index or load_index()
    query_vector = rag_index.vectorizer.transform([query])
    scores = (rag_index.matrix @ query_vector.T).toarray().ravel()
    if scores.size == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    results: list[dict] = []
    for rank, row_index in enumerate(top_indices, start=1):
        chunk = rag_index.chunks[int(row_index)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[row_index]),
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "name": chunk["name"],
                "source_file": chunk["source_file"],
                "text": chunk["text"],
            }
        )
    return results
