from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ---------- LOGO DISPLAY ----------
logo_path = Path(__file__).resolve().parent / "assets" / "greenopt_logo.png"
if logo_path.exists():
    logo = Image.open(logo_path)
    st.image(logo, width=180)
st.markdown(
    "<h1 style='color:#00FFAA;margin-top:-15px;'>GreenOpt — Digital ESG Engine</h1>",
    unsafe_allow_html=True,
)
st.caption("AI-driven Carbon Optimization & Catena-X Integration")

# ---------- LOAD OR GENERATE DATA ----------
@st.cache_data(show_spinner=False)
def load_data():
    path = DATA_DIR / "factory_data_3y.csv"
    if not path.exists():
        ts = pd.date_range("2022-10-15", "2025-10-15", freq="D")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "timestamp": ts,
            "electricity_kwh": rng.normal(12000, 800, len(ts)).round(2),
            "gas_m3": rng.normal(1600, 200, len(ts)).round(2),
            "production_ton": rng.normal(400, 25, len(ts)).round(2),
        })
        df["co2e_kg"] = df["electricity_kwh"]*0.475 + df["gas_m3"]*2.0
        df["pcf_kg_per_ton"] = df["co2e_kg"]/df["production_ton"]
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if "co2e_kg" not in df.columns:
            df["co2e_kg"] = df["electricity_kwh"]*0.475 + df["gas_m3"]*2.0
        if "pcf_kg_per_ton" not in df.columns:
            df["pcf_kg_per_ton"] = df["co2e_kg"]/df["production_ton"]
    return df

df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Settings")
date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)
agg = st.sidebar.selectbox("Aggregation", ["Daily", "Monthly"], index=0)
show_table = st.sidebar.checkbox("Show raw data", False)

# ---------- FILTER ----------
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

if agg == "Monthly":
    df = (
        df.set_index("timestamp")
        .resample("M")
        .agg({
            "electricity_kwh": "sum",
            "gas_m3": "sum",
            "production_ton": "sum",
            "co2e_kg": "sum"
        })
        .reset_index()
    )

# ---------- KPI SUMMARY ----------
total = df["co2e_kg"].sum()
avg = df["co2e_kg"].mean()
avg_pcf = (df["co2e_kg"].sum() / df["production_ton"].sum())
col1, col2, col3 = st.columns(3)
col1.metric("Total CO₂e", f"{total:,.0f} kg")
col2.metric("Avg CO₂e", f"{avg:,.2f} kg/{agg.lower()}")
col3.metric("Avg PCF", f"{avg_pcf:,.2f} kg/ton")

# ---------- TAB LAYOUT ----------
tabs = st.tabs(["📊 Dashboard", "🤖 AI Forecast", "🧩 Optimization", "🧮 Scenario", "📤 Catena-X JSON"])

# ---------- 1. Dashboard ----------
with tabs[0]:
    st.markdown("### 📈 CO₂e Emission Trend")
    fig = px.line(df, x="timestamp", y="co2e_kg", template="plotly_dark",
                  title="CO₂e Over Time", color_discrete_sequence=["#00FFAA"])
    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#00FFAA"),
        xaxis_title="Date",
        yaxis_title="CO₂e (kg)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⚡ Energy Breakdown")
    fig2 = px.line(df, x="timestamp", y=["electricity_kwh", "gas_m3"], template="plotly_dark",
                   color_discrete_sequence=["#00CC88", "#00FFAA"], title="Energy Consumption")
    fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"))
    st.plotly_chart(fig2, use_container_width=True)

    if show_table:
        st.dataframe(df.tail(50), use_container_width=True, height=400)

# ---------- 2. AI Forecast ----------
with tabs[1]:
    st.markdown("### 🔮 14-Day Moving Average Forecast")
    horizon = st.slider("Forecast horizon (days)", 7, 60, 14)
    hist = df.set_index("timestamp")["co2e_kg"]
    ma = hist.rolling(window=7, min_periods=1).mean()
    forecast_idx = pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=horizon)
    forecast_values = np.full(len(forecast_idx), ma.iloc[-1])
    forecast = pd.Series(forecast_values, index=forecast_idx)

    fig3 = px.line(template="plotly_dark")
    fig3.add_scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#00FFAA"))
    fig3.add_scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(color="#00BFFF", dash="dot"))
    fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"))
    st.plotly_chart(fig3, use_container_width=True)

# ---------- 3. Optimization ----------
with tabs[2]:
    st.markdown("### ⚙️ Optimization — Lagrange Multiplier")
    production_target = st.number_input("Production target (tons)", 300.0, 600.0, 400.0)
    def f(x): return 0.475*x[0] + 2.0*x[1]
    def constraint(x): return production_target - (0.002*x[0] + 0.005*x[1])
    res = minimize(f, [10000, 2000], constraints={'type': 'eq', 'fun': constraint})
    opt_elec, opt_gas = res.x
    st.success(f"✅ Optimal Electricity: {opt_elec:,.0f} kWh / Gas: {opt_gas:,.0f} m³")
    st.metric("Expected Emission", f"{f(res.x):,.2f} kg CO₂e")

# ---------- 4. Scenario ----------
with tabs[3]:
    st.markdown("### 🧮 Scenario Simulation")
    eff = st.slider("Electric Efficiency Improvement (%)", -20, 20, 0)
    gas_eff = st.slider("Gas Reduction (%)", -20, 20, 0)
    df_sim = df.copy()
    df_sim["electricity_kwh"] *= (1 - eff/100)
    df_sim["gas_m3"] *= (1 - gas_eff/100)
    df_sim["co2e_kg"] = df_sim["electricity_kwh"]*0.475 + df_sim["gas_m3"]*2.0
    change = ((df_sim["co2e_kg"].sum() - df["co2e_kg"].sum())/df["co2e_kg"].sum())*100
    st.metric("Total Emission Change", f"{change:+.2f}%")
    fig4 = px.line(df_sim, x="timestamp", y="co2e_kg", template="plotly_dark",
                   title="Scenario-adjusted CO₂e", color_discrete_sequence=["#00FFAA"])
    fig4.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"))
    st.plotly_chart(fig4, use_container_width=True)

# ---------- 5. Catena-X JSON ----------
with tabs[4]:
    st.markdown("### 📤 Catena-X / Digital Product Passport Export")
    sample = {
        "product_id": "GREENOPT-SAMPLE-001",
        "timestamp": str(pd.Timestamp.now()),
        "scope1_kg": float(df["gas_m3"].sum()*2.0),
        "scope2_kg": float(df["electricity_kwh"].sum()*0.475),
        "total_kg": float(df["co2e_kg"].sum()),
        "unit": "kg CO2e",
        "format": "Catena-X JSON v1.0"
    }
    json_text = json.dumps(sample, indent=2)
    st.code(json_text, language="json")
    st.download_button("💾 Download JSON", json_text, file_name="greenopt_catenaX.json")

# ---------- FOOTER ----------
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.8rem;margin-top:1rem;'>© 2025 GreenOpt — ESG·AI Digital Engine</p>",
    unsafe_allow_html=True,
)
