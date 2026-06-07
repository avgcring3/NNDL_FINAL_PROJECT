# Moscow Data Note

## Can We Train Directly On Moscow?

Only if we obtain trip-level Moscow taxi order data.

The public-data requirement is not just "city data". For this project, we need operational taxi demand records:

- pickup timestamp;
- pickup zone or coordinates;
- enough history to train and test a forecasting model.

Open city portals often publish permits, registry information, stops, routes, or aggregated indicators. Those are useful context, but they are not enough to train a pickup-demand neural model.

## Recommended Pitch Wording

Use this wording:

"The model is trained on public operational NYC taxi trip records because they include trip-level timestamps and pickup zones. The same pipeline is designed for Moscow deployment when trip-level order logs are available from a taxi operator, aggregator, or transport partner."

## Moscow Deployment Plan

1. Receive partner export with pickup timestamp and pickup zone.
2. Map coordinates to administrative zones or operational hexagons.
3. Aggregate records into hourly demand.
4. Fine-tune or retrain the model on Moscow data.
5. Evaluate by time-split test set and operational peak-hour under-forecasting.

## Why Not Fake Moscow Data?

Synthetic Moscow demo data is acceptable for local testing and presentation flow.

It should not be claimed as the real training dataset. The defended dataset should remain NYC TLC unless real Moscow trip records are available.
