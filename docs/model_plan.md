# Model Plan

## Objective

Forecast taxi pickup demand for each zone and hour.

The output is the expected number of pickups in the next hour or next several hours.

## Current Public Data

As of June 3, 2026, NYC TLC lists 2026 trip record files through April 2026. TLC publishes monthly trip data with an operational delay, typically around two months.

Default real-data source:

```text
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-04.parquet
```

## Data Contract

Minimum required fields:

- `pickup_datetime`;
- `pickup_zone_id`;
- `pickup_zone_name`.

NYC TLC raw parquet files use:

- `tpep_pickup_datetime`;
- `PULocationID`.

The code converts raw trip rows into hourly zone demand.

## Models

### Seasonal Naive

Baselines:

- last observed demand;
- same hour yesterday.

### MLP

Flattened lag window -> dense layers -> forecast head.

### Transformer

Sequence of historical demand values -> value projection -> zone embedding -> positional encoding -> Transformer encoder -> residual forecast head.

The residual head predicts a correction over the latest known demand value. This makes training stable for short demo runs and is a common practical pattern in forecasting.

## Loss

Training uses weighted MAE on log1p demand.

High-demand target hours receive higher weight, because under-forecasting peak hours is more expensive operationally than missing quiet hours.

## Metrics

- MAE;
- RMSE;
- sMAPE;
- under-forecast rate;
- peak-hour MAE.
