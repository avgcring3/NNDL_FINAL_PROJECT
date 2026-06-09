from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, REPORTS_DIR, ModelConfig, TrainConfig
from .data import load_hourly_csv, make_windows, time_split
from .demo_data import generate_hourly_demo
from .metrics import metric_bundle
from .models import make_model, weighted_mae_loss
from .report import plot_predictions, write_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RideFlow NN taxi demand forecaster.")
    parser.add_argument("--mode", choices=["demo", "csv"], default="demo")
    parser.add_argument("--input", default=None, help="Hourly Moscow demand CSV path for csv mode.")
    parser.add_argument("--model", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--future-hours", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_hourly(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "demo":
        hourly = generate_hourly_demo(city="moscow", days=args.days, seed=args.seed)
        output = PROCESSED_DATA_DIR / "demo_moscow_hourly_demand.csv"
        hourly.to_csv(output, index=False)
        return hourly, str(output.relative_to(PROJECT_ROOT))

    if args.mode == "csv":
        if not args.input:
            raise ValueError("--input is required for csv mode")
        return load_hourly_csv(args.input), args.input

    raise ValueError(f"Unknown mode: {args.mode}")


def build_loader(windows, indices, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.log1p(torch.from_numpy(windows.x[indices]))
    y = torch.log1p(torch.from_numpy(windows.y[indices]))
    zones = torch.from_numpy(windows.zone_ids[indices])
    return DataLoader(TensorDataset(x, y, zones), batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, train_config: TrainConfig, device: torch.device) -> float:
    model.train()
    losses = []
    for x_batch, y_batch, zone_batch in tqdm(loader, desc="train", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        zone_batch = zone_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_batch, zone_batch)
        loss = weighted_mae_loss(prediction, y_batch, train_config.high_demand_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate_loss(model, loader, train_config: TrainConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for x_batch, y_batch, zone_batch in tqdm(loader, desc="eval", leave=False):
        prediction = model(x_batch.to(device), zone_batch.to(device))
        loss = weighted_mae_loss(prediction, y_batch.to(device), train_config.high_demand_weight)
        losses.append(float(loss.cpu()))
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def predict(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    for x_batch, _, zone_batch in loader:
        pred_log = model(x_batch.to(device), zone_batch.to(device)).cpu().numpy()
        predictions.append(np.expm1(pred_log))
    return np.vstack(predictions)


@torch.no_grad()
def make_future_forecast(
    model,
    hourly: pd.DataFrame,
    lookback_hours: int,
    future_hours: int,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []

    for zone_id, group in hourly.groupby("zone_id"):
        group = group.sort_values("hour")
        zone_name = str(group["zone_name"].iloc[0])
        series = group.set_index("hour")["demand"].astype(float)
        full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
        history = series.reindex(full_index, fill_value=0.0).to_numpy(dtype=np.float32).tolist()
        if len(history) < lookback_hours:
            continue

        last_timestamp = full_index[-1]
        for step in range(1, future_hours + 1):
            window = np.asarray(history[-lookback_hours:], dtype=np.float32)
            x = torch.log1p(torch.from_numpy(window).unsqueeze(0)).to(device)
            zone_tensor = torch.tensor([int(zone_id)], dtype=torch.long, device=device)
            prediction_log = model(x, zone_tensor).cpu().numpy()[0, 0]
            forecast = float(max(0.0, np.expm1(prediction_log)))
            history.append(forecast)
            rows.append(
                {
                    "timestamp": last_timestamp + pd.Timedelta(hours=step),
                    "zone_id": int(zone_id),
                    "zone_name": zone_name,
                    "forecast_demand": forecast,
                    "step_hour": step,
                }
            )

    if not rows:
        raise ValueError("Could not build future forecast. Not enough history by zone.")
    return pd.DataFrame(rows)


def baseline_predictions(windows, indices, horizon: int) -> dict[str, np.ndarray]:
    x = windows.x[indices]
    last_value = np.repeat(x[:, [-1]], horizon, axis=1)
    if x.shape[1] >= 24:
        day_ago = x[:, -24 : -24 + horizon]
        if day_ago.shape[1] < horizon:
            day_ago = last_value
    else:
        day_ago = last_value
    return {"baseline_last": last_value, "baseline_day_ago": day_ago}


def save_outputs(args, windows, test_idx, prediction, future_forecast, metrics, history) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baselines = baseline_predictions(windows, test_idx, args.horizon)
    actual = windows.y[test_idx]

    metrics["baseline_last"] = metric_bundle(baselines["baseline_last"], actual)
    metrics["baseline_day_ago"] = metric_bundle(baselines["baseline_day_ago"], actual)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    frame = pd.DataFrame(
        {
            "timestamp": windows.timestamps[test_idx],
            "zone_id": windows.zone_ids[test_idx],
            "zone_name": windows.zone_names[test_idx],
            "actual": actual[:, 0],
            "model_prediction": prediction[:, 0],
            "baseline_last": baselines["baseline_last"][:, 0],
            "baseline_day_ago": baselines["baseline_day_ago"][:, 0],
        }
    ).sort_values(["timestamp", "zone_id"])
    predictions_path = ARTIFACTS_DIR / "predictions.csv"
    frame.to_csv(predictions_path, index=False)

    future_forecast.sort_values(["timestamp", "zone_name"]).to_csv(ARTIFACTS_DIR / "future_forecast.csv", index=False)
    pd.DataFrame(history).to_csv(ARTIFACTS_DIR / "training_curve.csv", index=False)
    plot_path = REPORTS_DIR / "forecast_sample.png"
    plot_predictions(frame, plot_path)
    write_html_report(metrics_path, predictions_path, plot_path, REPORTS_DIR / "demo_report.html")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_config = ModelConfig()
    train_config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, random_seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hourly, source = build_hourly(args)
    windows = make_windows(hourly, args.lookback, args.horizon)
    train_idx, val_idx, test_idx = time_split(windows)

    train_loader = build_loader(windows, train_idx, args.batch_size, shuffle=True)
    val_loader = build_loader(windows, val_idx, args.batch_size, shuffle=False)
    test_loader = build_loader(windows, test_idx, args.batch_size, shuffle=False)

    model = make_model(args.model, model_config, args.lookback, args.horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    history = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, train_config, device)
        val_loss = evaluate_loss(model, val_loader, train_config, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    prediction = predict(model, test_loader, device)
    future_forecast = make_future_forecast(model, hourly, args.lookback, args.future_hours, device)
    actual = windows.y[test_idx]
    metrics = {
        args.model: metric_bundle(prediction, actual),
        "metadata": {
            "mode": args.mode,
            "city": "moscow",
            "source": source,
            "lookback_hours": args.lookback,
            "horizon_hours": args.horizon,
            "future_hours": args.future_hours,
            "epochs": args.epochs,
            "samples": int(len(windows.x)),
            "test_samples": int(len(test_idx)),
        },
    }

    save_outputs(args, windows, test_idx, prediction, future_forecast, metrics, history)
    torch.save(model.state_dict(), ARTIFACTS_DIR / f"{args.model}_model.pt")
    print(f"saved_metrics={ARTIFACTS_DIR / 'metrics.json'}")
    print(f"saved_report={REPORTS_DIR / 'demo_report.html'}")
    print(f"saved_future_forecast={ARTIFACTS_DIR / 'future_forecast.csv'}")


if __name__ == "__main__":
    main()
