from app.generator import generate_answer
from app.prompts import REFUSAL_MESSAGE


def test_generate_answer_uses_relevant_sources() -> None:
    retrieved = [
        {
            "doc_id": "doc_price",
            "chunk_id": "doc_price_chunk_001",
            "name": "Pricing",
            "score": 0.42,
            "text": "RideFlow estimates future taxi price from tariff, distance, duration, demand, weather, and traffic.",
        }
    ]
    response = generate_answer("How does price estimation work?", retrieved, min_score=0.05)
    assert "doc_price" in response["answer"]
    assert response["sources"][0]["doc_id"] == "doc_price"


def test_generate_answer_refuses_without_relevant_context() -> None:
    retrieved = [
        {
            "doc_id": "doc_noise",
            "chunk_id": "doc_noise_chunk_001",
            "name": "Noise",
            "score": 0.0,
            "text": "This chunk is not relevant.",
        }
    ]
    response = generate_answer("Who won the 1998 FIFA World Cup?", retrieved, min_score=0.05)
    assert response["answer"] == REFUSAL_MESSAGE
    assert response["sources"] == []
