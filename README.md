# Predictive Modeling of Global Food Price Volatility Using Time-Series ML

An end-to-end Machine Learning mini-project focused on forecasting **country-level food price volatility** using the **WFP Global Market Monitor dataset**.

Built with a complete **time-series safe ML workflow**, recursive forecasting pipeline, and an interactive **Streamlit dashboard**.

---

## Problem Statement

Food price fluctuations directly impact food security, inflation, and humanitarian response planning.

This project predicts **future monthly food price volatility** at the **country level** using historical price change patterns and temporal ML features.

---

## Dataset

- **Source:** WFP Global Market Monitor
- **File Used:** `global-market-monitor.csv`
- **Granularity:** Country-level monthly data

---

## Project Workflow

### 1. Data Preprocessing

- metadata row handling using `skiprows=[1]`
- datetime conversion
- country-wise sorting
- missing value handling using forward/backward fill
- aggregation of monthly volatility per country

### 2. Feature Engineering

- lag features: `lag_1, lag_2, lag_3, lag_6, lag_12`
- temporal features: `month, year, quarter`
- cyclic encoding: `month_sin, month_cos`

### 3. Modeling

Models benchmarked:

- Linear Regression ( Best )
- Random Forest
- XGBoost
- LightGBM

### 4. Forecasting

Implemented **recursive 6-month multi-step forecasting**
using country-level lag updates.

### 5. Dashboard

Interactive Streamlit dashboard with:

- country selector
- historical trend
- 6-month forecast
- forecast table
- model insights

---

## Final Model Performance

| Model                 |     RMSE |        R² |
| --------------------- | -------: | --------: |
| **Linear Regression** | **5.86** | **0.407** |
| Random Forest         |     6.83 |     0.195 |
| LightGBM              |     6.90 |     0.177 |
| XGBoost               |     7.47 |     0.037 |

---

## Key Learning

A major highlight of this project was detecting and fixing **target leakage** caused by rolling-window features that included the current timestep.

This significantly improved the scientific validity of the forecasting pipeline.

---

## Streamlit Dashboard

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- VS Code Jupyter

---


---

## Author

**Sumedh Chinchmalatpure**
