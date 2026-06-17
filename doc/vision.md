# Vision

## Technologies

- Python for all pipeline code.
- TF-IDF from scikit-learn for indexing.
- Sparse matrix storage from SciPy.
- Joblib for vectorizer serialization.
- Streamlit for the UI.
- Pytest for automated checks.

## Index

The index is built from `data/raw/datasets.json`.

1. `scripts/ingest.py` converts dataset records to `data/processed/documents.jsonl`.
2. `app/chunker.py` splits documents into paragraph-aware chunks.
3. `scripts/build_index.py` fits a TF-IDF vectorizer.
4. Artifacts are saved to `data/index/vectorizer.pkl`, `data/index/matrix.npz`, and `data/index/chunks.jsonl`.

## Search

The retriever transforms a user query with the saved vectorizer and ranks chunks by cosine similarity. Because TF-IDF vectors are L2-normalized, matrix dot product is enough for cosine ranking.

## MVP Boundaries

- No external LLM call.
- No vector database.
- No automatic dataset download.
- No semantic embeddings.
- No reranker.

## Run

```bash
uv sync
uv run python scripts/build_index.py
uv run streamlit run app/main.py
```
