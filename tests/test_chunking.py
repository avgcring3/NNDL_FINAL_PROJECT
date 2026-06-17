from app.chunker import chunk_documents, chunk_text


def test_chunk_text_respects_max_chars() -> None:
    text = "\n\n".join([f"Paragraph {index} about RideFlow demand." for index in range(20)])
    chunks = chunk_text(text, max_chars=160, overlap=30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 160 for chunk in chunks)


def test_chunk_text_splits_long_paragraph() -> None:
    text = " ".join(["forecast"] * 180)
    chunks = chunk_text(text, max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_documents_preserves_metadata() -> None:
    documents = [
        {
            "doc_id": "doc_test",
            "name": "Test Doc",
            "source_file": "datasets.json",
            "text": "RideFlow uses demand features.\n\nIt estimates taxi prices.",
        }
    ]
    chunks = chunk_documents(documents, max_chars=140, overlap=20)
    assert chunks[0]["doc_id"] == "doc_test"
    assert chunks[0]["chunk_id"].startswith("doc_test_chunk_")
    assert chunks[0]["name"] == "Test Doc"
