# Tasklist

| Iteration | Result | Check | Status |
| --- | --- | --- | --- |
| 00 Scaffold | Project files, app package, scripts, tests folder | `python -c "import app.config"` | Done |
| 01 Demo data | `data/raw/datasets.json` with 10+ records | `python scripts/prepare_datasets.py` | Done |
| 02 Ingestion | `documents.jsonl` generated | `python scripts/ingest.py` | Done |
| 03 Chunking | Paragraph-aware chunks | `pytest tests/test_chunking.py -v` | Done |
| 04 Index | TF-IDF vectorizer, sparse matrix, chunks | `python scripts/build_index.py` | Done |
| 05 Retrieval | Top-k chunks with scores and metadata | `python scripts/check_retrieval.py` | Done |
| 06 Demo answer | Extractive answer and refusal | `python scripts/check_generator.py` | Done |
| 07 Streamlit UI | Question input, answer, sources, fragments | `streamlit run app/main.py` | Done |
| 08 Tests and README | Reproducible docs and green tests | `pytest tests -v` | Done |

## MVP Completion Criteria

- At least 10 raw text records.
- Build command creates index artifacts.
- Relevant questions return sourced answers.
- Negative question refuses.
- Automated tests pass.
