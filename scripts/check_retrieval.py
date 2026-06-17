from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.retriever import retrieve


if __name__ == "__main__":
    for query in [
        "future taxi trip price coefficients",
        "Moscow taxi demand forecasting",
        "ancient roman pottery",
    ]:
        print(f"\nQuery: {query}")
        for item in retrieve(query, top_k=3):
            print(f"- {item['doc_id']} score={item['score']:.3f} name={item['name']}")
