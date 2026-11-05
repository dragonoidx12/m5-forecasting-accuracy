# M5 Forecasting Accuracy - Final Report

## Objective
Forecast Walmart store-item sales for 28 days.

## Dataset
Bottom-level series: 30,490
Store count: 10
Product count: 3,049
Department count: 7
Hierarchy levels: 12
Aggregated series: 42,840

## Local Validation Period
Training data through d_1885
Prediction period: d_1886 - d_1913

## Features
- Historical sales lags
- Rolling sales averages
- Sales variability
- Calendar events
- SNAP indicators
- Product/store/category codes

## Initial RMSE Comparison
HistGradientBoosting: 2.146528
HGB Extended: 2.151290
Last 28-Day Mean: 2.218602
Seasonal Naive: 2.759459

## Hierarchical WRMSSE Comparison
               Model     RMSE  Local WRMSSE
        HGB Extended 2.151290      0.593603
HistGradientBoosting 2.146528      0.602303
  HGB 80% + Mean 20% 2.133090      0.662999
  HGB 50% + Mean 50% 2.139326      0.802691
      Seasonal Naive 2.759459      0.922781
    Last 28-Day Mean 2.218602      1.097484

## Selected Model
HGB Extended

## Local Validation WRMSSE
0.593603

## Initial HGB WRMSSE
0.602303

## Relative Improvement
Approximately 1.44%

## Final Forecasts
Validation period: d_1914 - d_1941
Evaluation period: d_1942 - d_1969

Kaggle submission rows: 60980

## Key Finding
The model with the lowest RMSE was not the
model with the lowest hierarchical WRMSSE.

## Limitations
- Historical training windows were sampled.
- Selling prices were used for WRMSSE weights,
  not as model prediction features.
- Models were selected using one validation window.
- Official Kaggle leaderboard score not measured.
- Synthetic Streamlit demo is illustrative only.
- The Streamlit app needs engineered features;
  it does not process raw sales history.
