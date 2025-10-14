# app/app.py
from __future__ import annotations
from pathlib import Path
import sys
import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- 시스템 경로 설정 ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.carbon_engine import add_carbon_columns
from src.data_utils import load_factory_data

# --- Streamlit 설정 ---
st.set_page_config(page_title="GreenOpt — CO₂e Intelligence Dashboard", layout="wide")

# --- 로고 로드 ---
LOGO_PATH = ROOT / "app" / "assets" / "greenopt_logo.png"
if LOGO_PATH.exists():
    logo_base64 = base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
            <img src="data:image/png;base64,{logo_base64}" width="55">
            <h1 style="color:#00ffb3;margin-bottom:0;">GreenOpt — CO₂e Intelligence Dashboard</h1>
        </div>
        <p style="color:#8a8a8a;margin-top:0;">AI-driven Carbon Footprint & Forecast System</p>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("GreenOpt — CO₂e Intelligence Dashboard")
    st.caption("AI-driven Carbon Footprint & Forecast System")

# --- Sidebar ---
st.sidebar.header("Configuration")
DATA_DIR = ROOT / "data"
DATASETS = {
    "Latest (1y)": DATA_DIR / "factory_data.csv",
    "3 Years (Full)": DATA_DIR / "factory_data_3y.csv",
}
dataset_choice = st.sidebar.selectbox("Dataset", list(DATASETS.keys()), index=1)
DATA_PATH = DATASETS[dataset_choice]

agg_level = st.sidebar.selectbox("Aggregation", ["Hourly", "Daily"], index=1)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 3, 30, 7, step=1)
show_table = st.sidebar.checkbox("Show raw data table", value=False)

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_and_prepare(path: Path) -> pd.DataFrame:
    df = load_factory_data(path)
    df = add_carbon_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df

df = load_and_prepare(DATA_PATH)
date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

# --- Filter Data ---
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

if agg_level == "Daily":
    df = (
        df.set_index("timestamp")
        .resample("D")
        .agg({"electricity_kwh": "sum", "gas_m3": "sum", "production_ton": "sum", "co2e_kg": "sum"})
        .reset_index()
    )

# --- KPIs ---
total_co2e = df["co2e_kg"].sum()
avg_co2e = df["co2e_kg"].mean()
avg_pcf = df["co2e_kg"].sum() / df["production_ton"].sum() if df["production_ton"].sum() > 0 else None

col1, col2, col3 = st.columns(3)
col1.metric("Total CO₂e", f"{total_co2e:,.0f} kg")
col2.metric("Avg CO₂e", f"{avg_co2e:,.2f} kg/{agg_level.lower()}")
col3.metric("Avg PCF", f"{avg_pcf:,.2f} kg/ton" if avg_pcf else "N/A")

# --- Plotly 시각화 ---
st.markdown("### 📈 CO₂e Emission Trends")
fig = px.line(df, x="timestamp", y="co2e_kg", title="CO₂e Over Time", template="plotly_dark")
fig.update_traces(line_color="#00FFAA")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### ⚡ Energy Consumption Breakdown")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["electricity_kwh"], name="Electricity (kWh)", line=dict(color="#32CD32")))
fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["gas_m3"], name="Gas (m³)", line=dict(color="#FFA500")))
fig2.update_layout(template="plotly_dark", title="Electricity vs Gas Consumption")
st.plotly_chart(fig2, use_container_width=True)

# --- AI 예측 (Forecast) ---
st.markdown("### 🔮 AI-based CO₂e Forecast")
try:
    # Prepare features
    X = df[["electricity_kwh", "gas_m3", "production_ton"]]
    y = df["co2e_kg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predict future
    last_row = df.iloc[-1]
    horizon_hours = forecast_horizon * (24 if agg_level == "Hourly" else 1)
    future_dates = pd.date_range(df["timestamp"].iloc[-1] + pd.Timedelta(hours=1 if agg_level == "Hourly" else 1),
                                 periods=horizon_hours, freq="H" if agg_level == "Hourly" else "D")
    future_input = np.tile(last_row[["electricity_kwh", "gas_m3", "production_ton"]].values, (len(future_dates), 1))
    forecast = model.predict(future_input)

    forecast_df = pd.DataFrame({"timestamp": future_dates, "predicted_co2e": forecast})
    fig3 = px.line(forecast_df, x="timestamp", y="predicted_co2e",
                   title=f"{forecast_horizon}-Day Forecast (AI Prediction)", template="plotly_dark")
    fig3.update_traces(line_color="#00BFFF")
    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.warning(f"Forecast unavailable: {e}")

# --- Raw Data Table ---
if show_table:
    st.markdown("### Raw Data (Sample)")
    st.dataframe(df.head(50), use_container_width=True)

# --- Download 버튼 ---
csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 Download CSV (Filtered)",
    data=csv,
    file_name="greenopt_filtered.csv",
    mime="text/csv",
)

st.markdown(
    "<p style='text-align:center;color:#666;font-size:0.8rem;margin-top:1rem;'>© 2025 GreenOpt — ESG·AI Data Intelligence Platform</p>",
    unsafe_allow_html=True,
)
