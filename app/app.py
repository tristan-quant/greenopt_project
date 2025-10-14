# app/app.py
from __future__ import annotations
import os, sys, subprocess, base64, io
from pathlib import Path

# --- 패키지 자동 설치 ---
required = ["streamlit", "plotly", "pandas", "numpy", "scikit-learn", "matplotlib"]
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"])

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- 프로젝트 경로 설정 ---
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# --- 로고(Base64 내장) ---
logo_base64 = """
iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAABHNCSVQICAgIfAhkiAAAAOJJREFUeJztwYEJgDAMBMHnR7Y4jmT2FJEq
jA0dQ5ek6vDL2z0AAAAAAAAAAMDZcwAAAI8gL9FJ8uSfAADwVq9fAAAAH/z9DQAAgD7yf4D08XyDwAAAwKf8AwAAAMDXkQAAAACf8AwA
AADAn/wDAAAAAF9EAgAAAPAPAAAAAMDXEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXw
AAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAgAAADAXwAAAMDfEAj4
AAB9sU8HDHCeNwAAAABJRU5ErkJggg==
"""

# --- 페이지 설정 ---
st.set_page_config(page_title="GreenOpt — CO₂e Intelligence Dashboard", layout="wide")

# --- 헤더 (로고 + 제목) ---
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
        <img src="data:image/png;base64,{logo_base64}" width="55">
        <h1 style="color:#00FFAA;margin-bottom:0;">GreenOpt — CO₂e Intelligence Dashboard</h1>
    </div>
    <p style="color:#999;margin-top:0;">AI-driven Carbon Footprint & Forecast System</p>
    """,
    unsafe_allow_html=True,
)

# --- 데이터 로드 ---
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = DATA_DIR / "factory_data_3y.csv"
    if not path.exists():
        # 간단한 더미 데이터 생성 (3년치)
        ts = pd.date_range("2022-10-15", "2025-10-15", freq="H")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "timestamp": ts,
            "electricity_kwh": rng.normal(500, 50, len(ts)),
            "gas_m3": rng.normal(60, 10, len(ts)),
            "production_ton": rng.normal(20, 3, len(ts)),
        })
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    df["co2e_kg"] = df["electricity_kwh"] * 0.475 + df["gas_m3"] * 2.0
    df["pcf_kg_per_ton"] = df["co2e_kg"] / df["production_ton"]
    return df

df = load_data()

# --- 사이드바 ---
st.sidebar.header("Settings")
agg_level = st.sidebar.selectbox("Aggregation", ["Hourly", "Daily"], index=1)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 3, 30, 7)
show_table = st.sidebar.checkbox("Show raw data", value=False)

date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

# --- 필터 ---
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

# --- KPI ---
col1, col2, col3 = st.columns(3)
col1.metric("Total CO₂e", f"{df['co2e_kg'].sum():,.0f} kg")
col2.metric("Avg CO₂e", f"{df['co2e_kg'].mean():,.2f} kg/{agg_level.lower()}")
avg_pcf = df["co2e_kg"].sum() / df["production_ton"].sum()
col3.metric("Avg PCF", f"{avg_pcf:,.2f} kg/ton")

# --- 그래프 ---
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

# --- AI 예측 ---
st.markdown("### 🔮 AI-based CO₂e Forecast")
try:
    X = df[["electricity_kwh", "gas_m3", "production_ton"]]
    y = df["co2e_kg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    last_row = df.iloc[-1][["electricity_kwh", "gas_m3", "production_ton"]]
    horizon = forecast_horizon * (24 if agg_level == "Hourly" else 1)
    future_dates = pd.date_range(df["timestamp"].iloc[-1] + pd.Timedelta(hours=1 if agg_level=="Hourly" else 1),
                                 periods=horizon, freq="H" if agg_level=="Hourly" else "D")
    X_future = np.tile(last_row.values, (len(future_dates), 1))
    preds = model.predict(X_future)
    pred_df = pd.DataFrame({"timestamp": future_dates, "predicted_co2e": preds})
    fig3 = px.line(pred_df, x="timestamp", y="predicted_co2e", title=f"{forecast_horizon}-Day Forecast", template="plotly_dark")
    fig3.update_traces(line_color="#00BFFF")
    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.warning(f"Forecast unavailable: {e}")

# --- Raw Data ---
if show_table:
    st.markdown("### Raw Data (Sample)")
    st.dataframe(df.head(50), use_container_width=True)

# --- 다운로드 ---
st.download_button(
    label="📥 Download CSV (Filtered)",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="greenopt_filtered.csv",
    mime="text/csv",
)

st.markdown("<p style='text-align:center;color:#777;font-size:0.8rem;'>© 2025 GreenOpt — ESG·AI Data Intelligence Platform</p>", unsafe_allow_html=True)
