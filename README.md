# RideFlow NN

Neural-network project for forecasting Moscow taxi demand and estimating a future trip price.

## What It Does

- forecasts hourly taxi demand by Moscow zone with a Transformer model;
- finds Moscow places, streets, metro stations, and house addresses;
- builds a driving route between points A and B;
- estimates a future trip price from tariff, distance, duration, demand, weather, traffic, date, and time;
- displays every price coefficient and its source;
- includes a 14-day forecast, trained model, Moscow data, metrics, and model report.

## One-Click Start

On Windows, double-click:

```text
START_RIDEFLOW.bat
```

The launcher:

1. checks and installs missing dependencies;
2. rebuilds the model and 14-day forecast when artifacts are missing or outdated;
3. starts the Streamlit dashboard;
4. opens it in the browser.

## Manual Start

```powershell
python -m pip install -r requirements.txt
python -m src.train --mode demo --epochs 5 --days 60 --lookback 48 --future-hours 336 --model transformer
streamlit run src/app.py
```

## Project Structure

- `src/` — model, training pipeline, dashboard, geocoding, routing, weather, and price calculation;
- `artifacts/` — trained model, metrics, predictions, and current 14-day forecast;
- `data/processed/` — generated Moscow hourly demand dataset;
- `reports/` — model report and forecast chart;
- `docs/` — pitch and modeling decisions;
- `scripts/` — alternative PowerShell launch commands.

## Data And External Services

- demand data: reproducible synthetic operational-style Moscow zone data;
- geocoding: ArcGIS, Photon, and Nominatim fallback;
- routing: OSRM;
- weather: Open-Meteo with wttr.in fallback.

## Important Limitation

This is not an official Yandex Go price quote. Exact Yandex Go tariffs and dynamic coefficients require official partner/API access. RideFlow NN provides an explainable estimate and shows the contribution of every available coefficient.

