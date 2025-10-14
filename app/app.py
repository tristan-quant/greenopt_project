# =====================================================
# GreenOpt — Digital ESG Engine (Full Feature, Safe Auto-Install)
# =====================================================
from __future__ import annotations

# ---------- 0) 누락 패키지 자동 설치 ----------
import sys, subprocess

def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦 Installing missing package: {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

for pkg in ["streamlit", "pandas", "numpy", "plotly", "scipy", "Pillow"]:
    _ensure(pkg)

# ---------- 1) 일반 import ----------
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- 2) 경로 설정 ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[0]
DATA_DIR = ROOT / "data"
ASSET_DIR = APP_DIR / "assets"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"

# ---------- 3) 탄소 계산 로직 ----------
EMISSION_FACTOR_ELECTRICITY = 0.475  # kg CO2e/kWh
EMISSION_FACTOR_GAS = 2.0           # kg CO2e/m3

def add_carbon_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["co2e_kg"] = (
        df["electricity_kwh"] * EMISSION_FACTOR_ELECTRICITY
        + df["gas_m3"] * EMISSION_FACTOR_GAS
    )
    def _pcf(row):
        prod = row.get("production_ton", 0.0)
        return (row["co2e_kg"] / prod) if prod and prod > 0 else np.nan
    df["pcf_kg_per_ton"] = df.apply(_pcf, axis=1)
    return df

@st.cache_data(show_spinner=False)
def load_factory_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=24*7, freq="H"),
            "electricity_kwh": np.random.uniform(80, 220, 24*7),
            "gas_m3": np.random.uniform(8, 35, 24*7),
            "production_ton": np.random.uniform(3, 16, 24*7),
            "line": np.random.choice(["A-Line", "B-Line"], 24*7),
            "product": np.random.choice(["Widget-X", "Widget-Y"], 24*7),
        })
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ---------- 4) 헤더 + 로고 ----------
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("GreenOpt — AI-driven Carbon Footprint Calculator")
    st.caption("Real-time CO₂e and Product Carbon Footprint (PCF) Dashboard")
with col2:
    # 로고 자동 탐색
    LOGO_CANDIDATES = [
        ASSET_DIR / "greenopt_logo.png",
        ASSET_DIR / "logo.png",
        ASSET_DIR / "logo_512.png",
        ROOT / "assets" / "greenopt_logo.png",
        ROOT / "assets" / "logo.png",
    ]
    logo_path = next((p for p in LOGO_CANDIDATES if p.exists()), None)
    if logo_path:
        st.image(Image.open(logo_path), use_container_width=True)
        st.caption(f"✅ Loaded logo: {logo_path.relative_to(ROOT)}")
    else:
        st.info("ℹ️ No logo found in app/assets or assets directory.")

st.divider()

# ---------- 5) 데이터 로딩 ----------
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.success("Uploaded CSV loaded.")
    else:
        df = load_factory_data(DEFAULT_CSV)
        if DEFAULT_CSV.exists():
            st.info(f"Loaded: {DEFAULT_CSV.name}")
        else:
            st.warning("Default CSV not found. Using sample data.")

# ---------- 6) 파생컬럼 ----------
df = add_carbon_columns(df)

# ---------- 7) 필터 ----------
with st.sidebar:
    st.header("Filters")
    if "line" in df.columns:
        sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique().tolist()))
    else:
        sel_lines = []
    if "product" in df.columns:
        sel_products = st.multiselect("Product", sorted(df["product"].dropna().unique().tolist()))
    else:
        sel_products = []
    if "timestamp" in df.columns:
        dmin = pd.to_datetime(df["timestamp"]).min().date()
        dmax = pd.to_datetime(df["timestamp"]).max().date()
        start, end = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    else:
        start, end = None, None

mask = pd.Series(True, index=df.index)
if sel_lines and "line" in df.columns:
    mask &= df["line"].isin(sel_lines)
if sel_products and "product" in df.columns:
    mask &= df["product"].isin(sel_products)
if start and end:
    mask &= (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

df_f = df.loc[mask].sort_values("timestamp")

# ---------- 8) KPI 카드 ----------
def show_kpi(df_in: pd.DataFrame):
    total_co2e = df_in["co2e_kg"].sum() if not df_in.empty else 0.0
    avg_pcf = df_in["pcf_kg_per_ton"].mean() if not df_in.empty else np.nan
    last = df_in.iloc[-1]["co2e_kg"] if not df_in.empty else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("Total CO₂e (kg)", f"{total_co2e:,.0f}")
    c2.metric("Avg PCF (kg/ton)", f"{avg_pcf:,.2f}" if np.isfinite(avg_pcf) else "N/A")
    c3.metric("Last Hour CO₂e (kg)", f"{last:,.1f}" if np.isfinite(last) else "N/A")

show_kpi(df_f)

# ---------- 9) 시각화 ----------
st.subheader("Emissions & PCF Trends")
if not df_f.empty:
    if _HAS_PLOTLY:
        fig1 = px.line(df_f, x="timestamp", y="co2e_kg", title="Hourly CO₂e Emissions (kg)", markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.line(df_f, x="timestamp", y="pcf_kg_per_ton", title="PCF (kg/ton)", markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Plotly not available. Showing Streamlit charts.")
        st.line_chart(df_f.set_index("timestamp")["co2e_kg"])
        st.line_chart(df_f.set_index("timestamp")["pcf_kg_per_ton"])
else:
    st.warning("No data after filters.")

# ---------- 10) 데이터 테이블 ----------
with st.expander("Show filtered data"):
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True)

# ---------- 11) 다운로드 ----------
csv_bytes = df_f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="greenopt_filtered.csv", mime="text/csv")

# ---------- 12) 최적화 데모 ----------
st.subheader("Energy Cost Optimization Demo")
if _HAS_SCIPY:
    target = st.slider("Target energy sum (units)", min_value=50, max_value=500, value=200, step=10)

    def objective(x):
        e, g = x
        return 0.15 * e + 0.08 * g

    cons = [{"type": "eq", "fun": lambda x, t=target: x[0] + x[1] - t}]
    bounds = [(0, None), (0, None)]
    x0 = [target / 2, target / 2]
    res = minimize(objective, x0, bounds=bounds, constraints=cons)
    st.write({"electricity": float(res.x[0]), "gas": float(res.x[1]), "cost": float(res.fun)})
else:
    st.info("SciPy not available. Skipping optimization demo.")
