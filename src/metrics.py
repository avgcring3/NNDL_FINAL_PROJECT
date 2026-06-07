from __future__ import annotations

import numpy as np


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def smape(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-6) -> float:
    numerator = np.abs(prediction - target)
    denominator = np.abs(target) + np.abs(prediction) + epsilon
    return float(np.mean(2.0 * numerator / denominator))


def underforecast_rate(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(prediction < target))


def peak_mae(prediction: np.ndarray, target: np.ndarray, quantile: float = 0.75) -> float:
    threshold = np.quantile(target, quantile)
    mask = target >= threshold
    if not np.any(mask):
        return mae(prediction, target)
    return mae(prediction[mask], target[mask])


def metric_bundle(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return {
        "mae": mae(prediction, target),
        "rmse": rmse(prediction, target),
        "smape": smape(prediction, target),
        "underforecast_rate": underforecast_rate(prediction, target),
        "peak_mae": peak_mae(prediction, target),
    }
