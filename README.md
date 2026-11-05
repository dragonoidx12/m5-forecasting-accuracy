# M5 Forecasting Accuracy

## Project
Walmart sales forecasting for the next 28 days.

## Final Model
HistGradientBoosting Regressor - Extended

## Local Validation
WRMSSE: 0.593603

## Submission
60,980 series, 28 daily forecast columns.

## Streamlit
Synthetic 28-day forecast simulation and
batch prediction from engineered feature CSVs.

## Setup
pip install -r requirements.txt
streamlit run app.py

The app does not accept raw M5 daily sales files.
The Python and scikit-learn versions used for
deployment must be compatible with the saved model.

## Competition
https://www.kaggle.com/competitions/m5-forecasting-accuracy
