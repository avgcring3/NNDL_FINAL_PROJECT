# RideFlow NN Pitch Outline

## Slide 1. Problem Statement

Taxi operators lose revenue when driver supply does not match local demand. Some zones have passenger wait time and missed rides while other zones have idle drivers.

RideFlow NN predicts pickup demand by zone and hour so vehicles can be positioned before the demand appears.

## Slide 2. Data

Primary training dataset:

- NYC TLC Trip Record Data;
- latest public 2026 data currently available through April 2026 for a June 2026 pitch;
- trip-level operational records;
- pickup timestamp and pickup location ID;
- millions of real trips, published monthly with delay.

Moscow deployment:

- the same pipeline supports Moscow if a partner provides trip-level order logs;
- offline Moscow demo is used only to show the interface and workflow.

## Slide 3. Modeling Approach

Baselines:

- last observed demand;
- same hour yesterday;
- MLP on historical lag windows.

Main model:

- Transformer encoder;
- demand sequence input;
- zone embedding;
- positional encoding;
- residual forecasting head;
- weighted MAE loss for peak demand.

## Slide 4. KPI And Success

Primary metric:

- MAE/RMSE improvement versus same-hour-yesterday baseline.

Operational KPI:

- lower peak-hour under-forecasting;
- fewer idle driver hours;
- shorter passenger wait time;
- better repositioning decisions.

## Slide 5. Business Impact

RideFlow NN helps dispatch teams see where demand will appear next.

Expected impact:

- higher completed-trip rate;
- lower idle time;
- better zone coverage;
- measurable improvement over static historical averages.
