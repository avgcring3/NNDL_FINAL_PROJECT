from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DataConfig:
    default_tlc_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet"
    pickup_datetime_col: str = "tpep_pickup_datetime"
    pickup_zone_col: str = "PULocationID"
    lookback_hours: int = 48
    horizon_hours: int = 1
    top_zones: int = 20


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.1
    max_zones: int = 500
    hidden_size: int = 128


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 0.001
    high_demand_weight: float = 2.0
    random_seed: int = 42
