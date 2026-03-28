import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Global Food Price Volatility Forecast",
    layout="wide"
)

st.title("🌍 Predictive Modeling of Global Food Price Volatility")
st.markdown("Forecasting country-level food price volatility using Time-Series ML")

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/processed_country_volatility.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/linear_regression.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, feature_cols

df = load_data()
model, feature_cols = load_model()

# remove leakage columns if still present
safe_cols = [col for col in df.columns if col in ["Date", "CountryName", "MonthlyChangeSA"] + feature_cols]
df = df[safe_cols]

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Dashboard Controls")

country_name = st.sidebar.selectbox(
    "Select Country",
    sorted(df["CountryName"].unique())
)

forecast_horizon = st.sidebar.slider(
    "Forecast Months",
    min_value=3,
    max_value=6,
    value=6
)

# -----------------------------
# FILTER COUNTRY
# -----------------------------
country_df = df[df["CountryName"] == country_name].copy()
country_df = country_df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# HISTORICAL TREND
# -----------------------------
st.subheader(f"📈 Historical Trend — {country_name}")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    country_df["Date"].tail(24),
    country_df["MonthlyChangeSA"].tail(24)
)

ax.set_title(f"Last 24 Months — {country_name}")
ax.set_xlabel("Date")
ax.set_ylabel("Monthly Change SA")
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# RECURSIVE FORECAST
# -----------------------------
st.subheader(f"🔮 {forecast_horizon}-Month Forecast")

last_row = country_df.iloc[-1].copy()

future_dates = pd.date_range(
    start=last_row["Date"] + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq="MS"
)

lag_values = {
    "lag_1": last_row["lag_1"],
    "lag_2": last_row["lag_2"],
    "lag_3": last_row["lag_3"],
    "lag_6": last_row["lag_6"],
    "lag_12": last_row["lag_12"]
}

future_predictions = []

for future_date in future_dates:
    month = future_date.month
    year = future_date.year
    quarter = future_date.quarter

    input_row = pd.DataFrame([{
        "lag_1": lag_values["lag_1"],
        "lag_2": lag_values["lag_2"],
        "lag_3": lag_values["lag_3"],
        "lag_6": lag_values["lag_6"],
        "lag_12": lag_values["lag_12"],
        "month": month,
        "year": year,
        "quarter": quarter,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12)
    }])

    pred = model.predict(input_row)[0]

    future_predictions.append({
        "Date": future_date,
        "Forecast": pred
    })

    lag_values["lag_12"] = lag_values["lag_6"]
    lag_values["lag_6"] = lag_values["lag_3"]
    lag_values["lag_3"] = lag_values["lag_2"]
    lag_values["lag_2"] = lag_values["lag_1"]
    lag_values["lag_1"] = pred

forecast_df = pd.DataFrame(future_predictions)

# -----------------------------
# FORECAST PLOT
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    country_df["Date"].tail(24),
    country_df["MonthlyChangeSA"].tail(24),
    label="Historical"
)

ax2.plot(
    forecast_df["Date"],
    forecast_df["Forecast"],
    marker="o",
    linestyle="--",
    label="Forecast"
)

ax2.set_title(f"{forecast_horizon}-Month Forecast — {country_name}")
ax2.set_xlabel("Date")
ax2.set_ylabel("Monthly Change SA")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

# -----------------------------
# FORECAST TABLE
# -----------------------------
st.subheader("📋 Forecast Values")
st.dataframe(forecast_df)

# -----------------------------
# PROJECT INSIGHTS
# -----------------------------
st.subheader("🧠 Model Insights")

st.info(
    """
    Best Model: Linear Regression  
    RMSE: 5.86  
    R²: 0.407  
    Strongest predictors: lag_1, lag_2, lag_3, month seasonality
    """
)