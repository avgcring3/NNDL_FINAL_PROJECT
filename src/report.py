from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(predictions: pd.DataFrame, output_path: Path, limit: int = 240) -> None:
    sample = predictions.sort_values("timestamp").head(limit)
    plt.figure(figsize=(12, 5))
    plt.plot(sample["timestamp"], sample["actual"], label="Actual", linewidth=2)
    plt.plot(sample["timestamp"], sample["model_prediction"], label="Transformer", linewidth=2)
    plt.plot(sample["timestamp"], sample["baseline_day_ago"], label="Same hour yesterday", linewidth=1.5)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Pickups")
    plt.title("Taxi demand forecast sample")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140)
    plt.close()


def write_html_report(metrics_path: Path, predictions_path: Path, plot_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for model_name, values in metrics.items():
        if not isinstance(values, dict) or "mae" not in values:
            continue
        rows.append(
            "<tr>"
            f"<td>{model_name}</td>"
            f"<td>{values['mae']:.3f}</td>"
            f"<td>{values['rmse']:.3f}</td>"
            f"<td>{values['smape']:.3f}</td>"
            f"<td>{values['underforecast_rate']:.3f}</td>"
            f"<td>{values['peak_mae']:.3f}</td>"
            "</tr>"
        )

    predictions = pd.read_csv(predictions_path).head(12)
    sample_table = predictions.to_html(index=False, classes="sample")
    image_rel = plot_path.relative_to(output_path.parent).as_posix()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RideFlow NN Demo Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #5b6572; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; }}
  </style>
</head>
<body>
  <h1>RideFlow NN Demo Report</h1>
  <div class="meta">Neural taxi demand forecasting with baselines and weighted peak-demand loss.</div>
  <h2>Metrics</h2>
  <table>
    <thead>
      <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>sMAPE</th><th>Underforecast</th><th>Peak MAE</th></tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Forecast Sample</h2>
  <img src="{image_rel}" alt="Forecast chart">
  <h2>Prediction Rows</h2>
  {sample_table}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
