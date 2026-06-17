from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "datasets.json"
DEMAND_CSV_PATH = ROOT_DIR / "data" / "processed" / "demo_moscow_hourly_demand.csv"


def demand_level(value: int) -> str:
    if value >= 85:
        return "very high"
    if value >= 70:
        return "high"
    if value >= 50:
        return "medium"
    return "low"


def time_bucket(hour: int) -> str:
    if 6 <= hour <= 10:
        return "morning commute"
    if 11 <= hour <= 16:
        return "daytime"
    if 17 <= hour <= 21:
        return "evening commute"
    return "night"


def load_seed_records(path: Path = RAW_DATA_PATH) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("datasets", [])
    return [record for record in records if str(record.get("id", "")).startswith("doc_")]


def build_demand_records(limit: int) -> list[dict]:
    records: list[dict] = []
    with DEMAND_CSV_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader, start=1):
            if len(records) >= limit:
                break

            timestamp = datetime.fromisoformat(row["hour"])
            demand = int(row["demand"])
            level = demand_level(demand)
            bucket = time_bucket(timestamp.hour)
            zone_name = row["zone_name"]
            zone_id = row["zone_id"]
            weekday = timestamp.strftime("%A")
            date_text = timestamp.strftime("%Y-%m-%d %H:%M")

            records.append(
                {
                    "id": f"demand_{index:06d}",
                    "name": f"Moscow demand observation {index:06d}",
                    "text": (
                        f"RideFlow demand observation for Moscow zone {zone_id} "
                        f"({zone_name}) at {date_text}. The observed hourly taxi "
                        f"demand is {demand}, which is classified as {level}. "
                        f"The timestamp falls on {weekday} during the {bucket} "
                        "period. This text record is indexed so the RAG system "
                        "can answer questions about demand levels, zones, dates, "
                        "time buckets, and how demand can affect future taxi price "
                        "estimation."
                    ),
                }
            )
    return records


def write_dataset(seed_records: list[dict], demand_records: list[dict]) -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Local RideFlow NN project materials and generated Moscow hourly demand CSV",
        "description": (
            "Educational text corpus for the RAG homework. It combines overview "
            "documents about RideFlow NN with text records generated from the "
            "project's reproducible Moscow hourly demand dataset."
        ),
        "record_count": len(seed_records) + len(demand_records),
        "datasets": seed_records + demand_records,
    }
    RAW_DATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=1200,
        help="Number of demand CSV rows to convert into text records.",
    )
    args = parser.parse_args()

    if args.limit < 1000:
        raise ValueError("--limit must be at least 1000 for the excellent-grade dataset")

    seed_records = load_seed_records()
    demand_records = build_demand_records(args.limit)
    write_dataset(seed_records, demand_records)
    print(
        f"Dataset is ready: {len(seed_records) + len(demand_records)} records "
        f"({len(seed_records)} overview + {len(demand_records)} demand) in {RAW_DATA_PATH}"
    )


if __name__ == "__main__":
    main()
