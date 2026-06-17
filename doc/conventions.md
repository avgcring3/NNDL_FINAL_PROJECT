# Conventions

## Scope

Keep changes small and directly connected to the homework RAG pipeline. Do not refactor the existing RideFlow forecasting code unless the RAG task requires it.

## Modules

- `scripts/ingest.py`: raw dataset to normalized documents.
- `app/chunker.py`: chunking logic.
- `scripts/build_index.py`: index construction.
- `app/retriever.py`: index loading and retrieval.
- `app/generator.py`: extractive answer and refusal policy.
- `app/main.py`: Streamlit UI.

## Answer Rules

- Answer only from retrieved chunks.
- Include sources with `doc_id` and score.
- Refuse when all retrieved chunks are below the relevance threshold.
- Do not invent facts outside the RideFlow corpus.

## Tests

Tests should cover chunk size constraints, metadata preservation, relevant retrieval, unrelated retrieval, and refusal behavior.

## Data And Dependencies

The MVP dataset is committed as JSON. Generated index artifacts are rebuildable and ignored by git.
