from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "datasets.json"


def main() -> None:
    data = json.loads(RAW_DATA_PATH.read_text(encoding="utf-8"))
    count = len(data["datasets"])
    print(f"Dataset is ready: {count} records in {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
