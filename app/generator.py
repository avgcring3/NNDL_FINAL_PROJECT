from __future__ import annotations

import re
from collections import Counter

from app.config import MIN_RELEVANCE_SCORE
from app.prompts import REFUSAL_MESSAGE

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "does",
    "for",
    "from",
    "how",
    "into",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _best_sentence(question_tokens: set[str], text: str) -> str:
    candidates = _sentences(text)
    if not candidates:
        return text.strip()
    scores = Counter()
    for index, sentence in enumerate(candidates):
        scores[index] = len(question_tokens & _tokens(sentence))
    best_index, best_score = scores.most_common(1)[0]
    if best_score == 0:
        return candidates[0]
    return candidates[best_index]


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    min_score: float = MIN_RELEVANCE_SCORE,
) -> dict:
    question_tokens = _tokens(question)
    relevant = [
        item
        for item in retrieved_chunks
        if item["score"] >= min_score
        and len(question_tokens & _tokens(item["text"])) >= 1
    ]
    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    answer_parts = []
    for item in relevant[:3]:
        sentence = _best_sentence(question_tokens, item["text"])
        answer_parts.append(f"- {sentence} [{item['doc_id']}, score={item['score']:.3f}]")

    return {
        "answer": "Based on the indexed RideFlow corpus:\n" + "\n".join(answer_parts),
        "sources": [
            {
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "name": item["name"],
                "score": item["score"],
                "text": item["text"],
            }
            for item in relevant[:5]
        ],
    }
