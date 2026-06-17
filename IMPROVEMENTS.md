# Improvements

## Implemented In MVP

### Relevance threshold and refusal policy

The generator uses a minimum retrieval score before answering. If no retrieved chunk passes the threshold, it returns a refusal instead of fabricating an answer.

Changed files:

- `app/generator.py`
- `app/prompts.py`
- `tests/test_generator.py`

Check:

```bash
python scripts/check_generator.py
pytest tests/test_generator.py -v
```

## Planned Improvements

### Embeddings instead of TF-IDF

Sentence embeddings would improve semantic matching for questions that use different wording from the source corpus. This would change `scripts/build_index.py` and `app/retriever.py`.

Check with a fixed set of demo questions and compare whether relevant documents appear in top-3 more often.

### Better Streamlit diagnostics

The UI can add score filters, highlighted matching terms, query history, and a compact source table. This would change `app/main.py`.

Check manually with the three demo questions and one negative question.
