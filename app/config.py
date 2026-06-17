from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

RAW_DATA_PATH = RAW_DIR / "datasets.json"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

INDEX_VECTORIZER_PATH = INDEX_DIR / "vectorizer.pkl"
INDEX_MATRIX_PATH = INDEX_DIR / "matrix.npz"
INDEX_CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

CHUNK_MAX_CHARS = 700
CHUNK_OVERLAP = 120
MIN_RELEVANCE_SCORE = 0.05

DEMO_QUESTIONS = [
    "How does RideFlow estimate a future taxi trip price?",
    "Which geocoding, routing, and weather services does RideFlow use?",
    "How is taxi demand forecasting produced in RideFlow?",
    "Who won the 1998 FIFA World Cup?",
]
