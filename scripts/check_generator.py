from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.generator import generate_answer
from app.config import DEMO_QUESTIONS
from app.retriever import retrieve


if __name__ == "__main__":
    for question in DEMO_QUESTIONS:
        print(f"\nQuestion: {question}")
        response = generate_answer(question, retrieve(question, top_k=5))
        print(response["answer"])
        print("Sources:", [source["doc_id"] for source in response["sources"]])
