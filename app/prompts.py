SYSTEM_RULES = """Answer only from retrieved context.
If the context is not relevant enough, refuse briefly and say that the index does not contain enough information.
Always return sources with doc_id and score."""

REFUSAL_MESSAGE = (
    "I do not have enough relevant context in the RideFlow index to answer this question."
)
