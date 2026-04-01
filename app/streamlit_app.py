import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Food Price Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# COUNTRY FLAG PALETTES
# ======================================
COUNTRY_THEMES = {
    "India": ("#FF9933", "#FFFFFF", "#138808", "#111111", "#115511"),
    "United States": ("#B22234", "#FFFFFF", "#3C3B6E", "#FFFFFF", "#D1D5DB"),
    "Brazil": ("#009B3A", "#FFDF00", "#002776", "#FFFFFF", "#D1FAE5"),
    "China": ("#DE2910", "#FFDE00", "#8B0000", "#FFFFFF", "#FDE68A"),
    "Japan": ("#FFFFFF", "#BC002D", "#1F2937", "#111111", "#6B7280"),
    "Germany": ("#000000", "#DD0000", "#FFCE00", "#FFFFFF", "#FDE68A"),
    "Ukraine": ("#005BBB", "#FFD500", "#005BBB", "#111111", "#1E3A8A"),
    "Pakistan": ("#01411C", "#FFFFFF", "#01411C", "#FFFFFF", "#BBF7D0"),
    "default": ("#0D1B2A", "#1B2838", "#4FC3F7", "#FFFFFF", "#B0BEC5")
}

COUNTRY_NAME_MAP = {
    "United States of America": "United States",
    "USA": "United States",
    "US": "United States",
    "Viet Nam": "Vietnam",
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "United Republic of Tanzania": "Tanzania",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Cote d'Ivoire": "Ivory Coast",
}

DEFAULT_THEME = COUNTRY_THEMES["default"]


def get_theme(country):
    normalized = COUNTRY_NAME_MAP.get(country, country)
    return COUNTRY_THEMES.get(normalized, DEFAULT_THEME)


def hex_to_rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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

country_options = sorted(df["CountryName"].unique())

if "selected_country" not in st.session_state:
    st.session_state.selected_country = country_options[0]

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:
    st.markdown("## Global Food Security Intelligence")
    country_name = st.selectbox(
        "Select Country",
        country_options,
        index=country_options.index(st.session_state.selected_country)
    )
    forecast_horizon = st.slider("Forecast Horizon (months)", 3, 6, 6)
    show_coeff = st.toggle("Show Feature Impact", value=True)

st.session_state.selected_country = country_name
p, s, acc, txt, txt2 = get_theme(country_name)

# ======================================
# DYNAMIC CSS
# ======================================
st.markdown(f"""
<style>
html, body, [class*="css"] {{ color: {txt}; }}

.stApp {{
    background: linear-gradient(135deg, {p} 0%, {s} 50%, {acc} 100%);
}}

section[data-testid="stSidebar"] {{
    background: {hex_to_rgba(p, 0.88)};
}}

.glass-card {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 1.5rem;
    backdrop-filter: blur(14px);
    margin-bottom: 1.5rem;
}}

.metric-card {{
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
}}

/* IMPORTANT: KEEP MENU & HEADER VISIBLE */
header {{ visibility: visible !important; }}
#MainMenu {{ visibility: visible !important; }}

</style>
""", unsafe_allow_html=True)

country_df = df[df["CountryName"] == country_name].sort_values("Date").reset_index(drop=True)

# ======================================
# HERO
# ======================================
st.markdown(f"""
<div class="glass-card">
    <h1 style="margin-bottom:0.5rem;">{country_name} Food Price Volatility Command Center</h1>
    <p style="opacity:0.85; line-height:1.7;">
        AI-powered forecasting dashboard for monitoring food price instability, seasonal momentum,
        and forward volatility signals. Theme adapts to the selected country's flag palette.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================
# METRICS
# ======================================
latest_val = country_df["MonthlyChangeSA"].iloc[-1]
hist_mean = country_df["MonthlyChangeSA"].mean()
hist_std = country_df["MonthlyChangeSA"].std()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><h3>Latest</h3><h2>{latest_val:+.2f}</h2></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h3>Mean</h3><h2>{hist_mean:.2f}</h2></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h3>Volatility</h3><h2>{hist_std:.2f}</h2></div>', unsafe_allow_html=True)

# ======================================
# WORLD MAP
# ======================================
map_df = df.groupby("CountryName", as_index=False).agg({"MonthlyChangeSA": "mean"})
map_df["MapCountry"] = map_df["CountryName"].replace(COUNTRY_NAME_MAP)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
fig_map = px.choropleth(
    map_df,
    locations="MapCountry",
    locationmode="country names",
    color="MonthlyChangeSA",
    hover_name="CountryName",
    color_continuous_scale="Viridis"
)
fig_map.update_layout(
    height=450,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=txt)
)
st.plotly_chart(fig_map, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# FORECAST ENGINE
# ======================================
last_row = country_df.iloc[-1].copy()
future_dates = pd.date_range(
    last_row["Date"] + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq="MS"
)

lag_values = {k: last_row[k] for k in ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12"]}
future_predictions = []

for future_date in future_dates:
    month = future_date.month
    input_row = pd.DataFrame([{
        "lag_1": lag_values["lag_1"],
        "lag_2": lag_values["lag_2"],
        "lag_3": lag_values["lag_3"],
        "lag_6": lag_values["lag_6"],
        "lag_12": lag_values["lag_12"],
        "month": month,
        "year": future_date.year,
        "quarter": future_date.quarter,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12)
    }])

    pred = model.predict(input_row)[0]
    future_predictions.append({"Date": future_date, "Forecast": pred})

    lag_values["lag_12"] = lag_values["lag_6"]
    lag_values["lag_6"] = lag_values["lag_3"]
    lag_values["lag_3"] = lag_values["lag_2"]
    lag_values["lag_2"] = lag_values["lag_1"]
    lag_values["lag_1"] = pred

forecast_df = pd.DataFrame(future_predictions)

# ======================================
# FORECAST CHART
# ======================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=country_df["Date"].tail(24),
    y=country_df["MonthlyChangeSA"].tail(24),
    mode="lines",
    name="Historical",
    line=dict(color=txt2, width=3)
))
fig.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Forecast"],
    mode="lines+markers",
    name="Forecast",
    line=dict(color=acc, width=3, dash="dash")
))
fig.update_layout(
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=txt)
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# TABLE + INSIGHTS
# ======================================
left, right = st.columns([2, 1])
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Forecast Table")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    proj_mean = forecast_df["Forecast"].mean()
    signal = "Stable" if abs(proj_mean) < hist_std * 0.5 else ("Elevated" if proj_mean > 0 else "Declining")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Market Intelligence")
    st.write(f"**Signal:** {signal}")
    st.write(f"**Projected Mean:** {proj_mean:+.3f}")
    st.write(f"**Forecast Window:** {forecast_horizon} months")
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# FEATURE IMPACT
# ======================================
if show_coeff:
    coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": model.coef_})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=True).tail(10)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig2 = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Top Feature Drivers"
    )
    fig2.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=txt)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================
# FOOTER
# ======================================
st.markdown(f"""
<div style='text-align:center; padding:2rem; opacity:0.5;'>
    Global Food Price Intelligence Platform · {country_name}
</div>
""", unsafe_allow_html=True)