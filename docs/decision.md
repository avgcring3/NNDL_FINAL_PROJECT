# Project Direction

## Decision

Build a taxi demand forecasting capstone.

Do not continue with the original SubSaver pitch as the main project. SubSaver has a real business problem, but the submitted model is mostly rule-based logic plus an OpenAI API layer. The NNDL capstone needs a trained neural network and a defensible data story.

## Selected Project

RideFlow NN predicts taxi pickup demand by zone and hour.

The project is built around a real operational dataset and a neural time-series model:

- NYC TLC trip records for the official public dataset;
- Transformer encoder for sequence forecasting;
- MLP and seasonal naive baselines;
- weighted loss for high-demand under-forecasting;
- dashboard/report for pitch demo.

## Moscow Scope

Moscow is a valid deployment target, but not the safest public-data training target.

The project can be done for Moscow if we have a real export from a taxi operator, aggregator, or city transport system:

- pickup timestamp;
- pickup zone or coordinates;
- optional fare, trip distance, service class, cancellation flag.

If only aggregated public statistics are available, the model cannot honestly be presented as trained on trip-level Moscow taxi demand. The repository therefore includes a Moscow-style offline demo and a CSV adapter, while the defended real-data source remains NYC TLC.

## Defense Logic

For the pitch, phrase it as:

"We train and evaluate the neural forecasting model on public operational taxi trip data from NYC TLC. The same pipeline is deployment-ready for Moscow once a partner provides trip-level pickup logs."
