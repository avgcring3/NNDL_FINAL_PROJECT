import json
from pathlib import Path


def test_dataset_meets_excellent_scale() -> None:
    path = Path("data/raw/datasets.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["datasets"]
    assert len(records) >= 1000
    assert data["record_count"] == len(records)


def test_dataset_records_are_unique_texts() -> None:
    path = Path("data/raw/datasets.json")
    records = json.loads(path.read_text(encoding="utf-8"))["datasets"]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    assert all(record["text"].strip() for record in records)
