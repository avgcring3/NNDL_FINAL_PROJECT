from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedDemand:
    x: np.ndarray
    y: np.ndarray
    zone_ids: np.ndarray
    zone_names: np.ndarray
    timestamps: np.ndarray


def load_hourly_csv(path: str) -> pd.DataFrame:
    hourly = pd.read_csv(path)
    hourly["hour"] = pd.to_datetime(hourly["hour"], errors="coerce")
    hourly = hourly.dropna(subset=["hour", "zone_id", "demand"])
    if "zone_name" not in hourly.columns:
        hourly["zone_name"] = hourly["zone_id"].astype(str)
    return hourly[["zone_id", "zone_name", "hour", "demand"]]


def make_windows(hourly: pd.DataFrame, lookback_hours: int, horizon_hours: int) -> WindowedDemand:
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    zone_ids: list[int] = []
    zone_names: list[str] = []
    timestamps: list[pd.Timestamp] = []

    for zone_id, group in hourly.groupby("zone_id"):
        group = group.sort_values("hour")
        zone_name = str(group["zone_name"].iloc[0])
        series = group.set_index("hour")["demand"].astype(float)
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
        values = series.reindex(full_index, fill_value=0.0).to_numpy(dtype=np.float32)

        max_start = len(values) - lookback_hours - horizon_hours + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            end = start + lookback_hours
            target_end = end + horizon_hours
            windows.append(values[start:end])
            targets.append(values[end:target_end])
            zone_ids.append(int(zone_id))
            zone_names.append(zone_name)
            timestamps.append(full_index[end])

    if not windows:
        raise ValueError("Not enough hourly data to create windows. Reduce lookback or provide more history.")

    return WindowedDemand(
        x=np.asarray(windows, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        zone_ids=np.asarray(zone_ids, dtype=np.int64),
        zone_names=np.asarray(zone_names),
        timestamps=np.asarray(timestamps),
    )


def time_split(windows: WindowedDemand, train_ratio: float = 0.7, val_ratio: float = 0.15):
    order = np.argsort(windows.timestamps)
    train_end = int(len(order) * train_ratio)
    val_end = int(len(order) * (train_ratio + val_ratio))
    return order[:train_end], order[train_end:val_end], order[val_end:]
