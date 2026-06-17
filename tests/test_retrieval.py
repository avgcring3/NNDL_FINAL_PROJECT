import json

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retriever import load_index, retrieve


def _write_test_index(tmp_path):
    chunks = [
        {
            "chunk_id": "doc_price_chunk_001",
            "doc_id": "doc_price",
            "name": "Pricing",
            "source_file": "test.json",
            "text": "RideFlow estimates taxi price from distance duration demand weather and traffic.",
        },
        {
            "chunk_id": "doc_ui_chunk_001",
            "doc_id": "doc_ui",
            "name": "UI",
            "source_file": "test.json",
            "text": "The Streamlit dashboard shows route maps forecast charts and price coefficients.",
        },
    ]
    vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    joblib.dump(vectorizer, tmp_path / "vectorizer.pkl")
    sparse.save_npz(tmp_path / "matrix.npz", matrix)
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(chunk) for chunk in chunks),
        encoding="utf-8",
    )


def test_retrieve_returns_relevant_top_result(tmp_path) -> None:
    _write_test_index(tmp_path)
    index = load_index(tmp_path)
    results = retrieve("taxi price weather demand", index=index, top_k=2)
    assert results[0]["doc_id"] == "doc_price"
    assert results[0]["score"] > 0


def test_retrieve_unrelated_query_has_zero_score(tmp_path) -> None:
    _write_test_index(tmp_path)
    index = load_index(tmp_path)
    results = retrieve("volcano archaeology pottery", index=index, top_k=2)
    assert results[0]["score"] == 0
