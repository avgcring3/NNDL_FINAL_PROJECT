# Project Direction

## Decision

Build a taxi demand forecasting capstone.

Do not continue with the original SubSaver pitch as the main project. SubSaver has a real business problem, but the submitted model is mostly rule-based logic plus an OpenAI API layer. The NNDL capstone needs a trained neural network and a defensible data story.

## Selected Project

RideFlow NN predicts taxi pickup demand by zone and hour.

The final repository is built around a neural time-series model and a reproducible Moscow demo dataset:

- synthetic operational-style hourly demand for Moscow zones;
- Transformer encoder for sequence forecasting;
- MLP and seasonal naive baselines;
- weighted loss for high-demand under-forecasting;
- dashboard/report for pitch demo;
- CSV and NYC TLC adapters for future real-data training.

## Moscow Scope

Moscow is a valid deployment target, but not the safest public-data training target.

The project can be done for Moscow if we have a real export from a taxi operator, aggregator, or city transport system:

- pickup timestamp;
- pickup zone or coordinates;
- optional fare, trip distance, service class, cancellation flag.

If only aggregated public statistics are available, the model cannot honestly be presented as trained on trip-level Moscow taxi demand. The repository therefore clearly labels the included Moscow data as synthetic and provides adapters for future real operational data.

## Defense Logic

For the pitch, phrase it as:

"We train and evaluate the neural forecasting pipeline on reproducible operational-style Moscow demand data. The same pipeline is ready to retrain on real trip-level order logs once a partner provides them."
