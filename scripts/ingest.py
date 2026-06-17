from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import DOCUMENTS_PATH, RAW_DATA_PATH


def load_dataset(path: Path = RAW_DATA_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("datasets")
    if not isinstance(records, list) or not records:
        raise ValueError("datasets.json must contain a non-empty 'datasets' array")
    return records


def ingest_documents(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = DOCUMENTS_PATH,
) -> list[dict]:
    records = load_dataset(input_path)
    documents: list[dict] = []
    for index, record in enumerate(records, start=1):
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        doc_id = record.get("id") or f"doc_{index:03d}"
        documents.append(
            {
                "doc_id": doc_id,
                "name": record.get("name") or record.get("title") or doc_id,
                "source_file": input_path.name,
                "text": text,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")
    return documents


if __name__ == "__main__":
    docs = ingest_documents()
    print(f"Wrote {len(docs)} documents to {DOCUMENTS_PATH}")
