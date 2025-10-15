# =====================================================
# GreenOpt — Digital ESG Engine (Final Integrated Edition)
# =====================================================

from __future__ import annotations
import sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------- 0) Auto-install guard ----------
def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

for pkg in [
    "pandas", "numpy", "plotly", "scipy", "streamlit", "Pillow", "scikit-learn", "reportlab"
]:
    _ensure(pkg)

import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- 1) UI/Theme ----------
st.set_page_config(page_title="GreenOpt — Carbon Intelligence Platform", layout="wide")

# dark theme style
BG = "#0B0E11"
BG2 = "#111827"
TXT = "#F9FAFB"
GREEN = "#22C55E"

def init_theme():
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{
        background-color:{BG}!important;
        color:{TXT}!important;
      }}
      .stApp {{background-color:{BG}!important;color:{TXT}!important;}}
      h1,h2,h3,h4,h5,h6,p,div,span,td,th,button,label,select,input,textarea {{
        color:{TXT}!important;
      }}
      [data-testid="stSidebar"], .stSidebar {{
        background-color:{BG2}!important;
        color:{TXT}!important;
      }}
      /* Upload box */
      [data-testid="stFileUploaderDropzone"] {{
        background:{BG2}!important;
        border:1px dashed #374151!important;
        color:{TXT}!important;
      }}
      /* Number input styling */
      .stNumberInput div[data-baseweb="input"],
      .stNumberInput input {{
        background:{BG2}!important;
        color:{TXT}!important;
        border:1px solid #374151!important;
      }}
      /* Plus/minus buttons dark & colored */
      .stNumberInput div[role="group"],
      .stNumberInput div[data-baseweb="button-group"] {{
        background:{BG2}!important;
      }}
      .stNumberInput [data-baseweb="button"] {{
        background:{BG2}!important; color:{TXT}!important;
      }}
      .stNumberInput div[role="group"] > [data-baseweb="button"]:first-child {{
        border:1px solid #EF4444!important;
      }}
      .stNumberInput div[role="group"] > [data-baseweb="button"]:last-child {{
        border:1px solid {GREEN}!important;
      }}
      .stNumberInput div[role="group"] > [data-baseweb="button"]:first-child svg,
      .stNumberInput div[role="group"] > [data-baseweb="button"]:first-child svg * {{
        fill:#EF4444!important; stroke:#EF4444!important;
      }}
      .stNumberInput div[role="group"] > [data-baseweb="button"]:last-child svg,
      .stNumberInput div[role="group"] > [data-baseweb="button"]:last-child svg * {{
        fill:{GREEN}!important; stroke:{GREEN}!important;
      }}
      .stNumberInput [data-baseweb="button"]:hover {{
        filter:brightness(1.2);
      }}
      /* Expander/Containers/Table */
      [data-testid="stExpander"] details, [data-testid="stExpander"] summary {{
        background:{BG2}!important; color:{TXT}!important;
        border:1px solid #374151!important; border-radius:10px!important;
      }}
      [data-testid="stStyledTable"] thead th {{
        background:#0F172A!important; color:{TXT}!important;
      }}
      [data-testid="stStyledTable"] tbody td {{
        background:{BG2}!important; color:{TXT}!important;
      }}
    </style>
    """, unsafe_allow_html=True)

init_theme()

# ---------- 2) Paths & sample data ----------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"

EMISSION_FACTOR_ELECTRICITY_DEFAULT = 0.475
EMISSION_FACTOR_GAS = 2.0

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        periods = 24 * 365 * 3
        idx = pd.date_range("2023-01-01", periods=periods, freq="H")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "timestamp": idx,
            "electricity_kwh": rng.uniform(80, 220, periods),
            "gas_m3": rng.uniform(8, 35, periods),
            "production_ton": rng.uniform(4, 18, periods),
            "line": rng.choice(["A-Line","B-Line","C-Line"], periods, p=[0.4,0.4,0.2]),
            "product": rng.choice(["Widget-X","Widget-Y","Widget-Z"], periods),
        })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "electricity_kwh":"sum","gas_m3":"sum","production_ton":"sum",
        "co2e_kg":"sum","pcf_kg_per_ton":"mean"
    }
    return (df.set_index("timestamp").resample(rule).agg(agg).reset_index())

# ---------- 3) Sidebar ----------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (3+ years preferred)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
else:
    df = load_data(DEFAULT_CSV)
    st.sidebar.info("Loaded default / generated sample data.")

st.sidebar.header("Scope 2 method")
scope2_method = st.sidebar.selectbox("Electricity EF method", ["Location-based","Market-based"])
if scope2_method == "Location-based":
    ef_elec_input = st.sidebar.number_input("EF (kg/kWh)", value=EMISSION_FACTOR_ELECTRICITY_DEFAULT, step=0.01)
else:
    ef_elec_input = st.sidebar.number_input("EF (market-based, kg/kWh)", value=0.0, step=0.01)

# ---------- 4) Filters ----------
st.sidebar.header("Filters")

tmin_all = df["timestamp"].min().date()
tmax_all = df["timestamp"].max().date()

range_mode = st.sidebar.radio("Range mode", ["All data","Custom"], horizontal=True, index=0)
if range_mode == "Custom":
    start_date, end_date = st.sidebar.date_input("Date range",
                    value=(tmin_all,tmax_all),min_value=tmin_all,max_value=tmax_all)
else:
    start_date, end_date = tmin_all, tmax_all

sel_lines = st.sidebar.multiselect("Line", sorted(df["line"].dropna().unique()))
sel_products = st.sidebar.multiselect("Product", sorted(df["product"].dropna().unique()))
rule = st.sidebar.selectbox("Time granularity", ["H","D","W","M"], index=1)

# ---------- 5) Filter apply ----------
if range_mode == "Custom":
    mask = (df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date)+pd.Timedelta(hours=23,minutes=59))
    df_f = df.loc[mask].copy()
else:
    df_f = df.copy()

if sel_lines: df_f = df_f[df_f["line"].isin(sel_lines)]
if sel_products: df_f = df_f[df_f["product"].isin(sel_products)]
df = df_f.sort_values("timestamp").reset_index(drop=True)

# ---------- 6) Carbon calc ----------
def add_carbon_columns(df_in):
    dfc = df_in.copy()
    dfc["co2e_kg"] = dfc["electricity_kwh"]*ef_elec_input + dfc["gas_m3"]*EMISSION_FACTOR_GAS
    dfc["pcf_kg_per_ton"] = np.where(dfc["production_ton"]>0, dfc["co2e_kg"]/dfc["production_ton"], np.nan)
    return dfc

df = add_carbon_columns(df)
df_g = resample_df(df, rule)

# ---------- 7) KPI ----------
st.title("GreenOpt — AI Carbon Intelligence Platform")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total CO₂e (kg)", f"{df_g['co2e_kg'].sum():,.0f}")
c2.metric("Avg PCF (kg/ton)", f"{df_g['pcf_kg_per_ton'].mean():,.2f}")
c3.metric(f"Last {rule} CO₂e (kg)", f"{df_g['co2e_kg'].iloc[-1]:,.1f}")
c4.metric("Periods", f"{len(df_g):,}")

# ---------- 8) Chart ----------
st.subheader("Time-series overview")
if not df_g.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"],
                             mode="lines", name="CO₂e (kg)",
                             line=dict(color=GREEN,width=2.5)))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      font_color=TXT, title_font_color=TXT)
    fig.update_xaxes(showgrid=False, range=[df_g["timestamp"].min(), df_g["timestamp"].max()])
    fig.update_yaxes(gridcolor="#1F2937")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data to plot")

# ---------- 9) Optimization ----------
st.subheader("Optimization (toy)")
with st.expander("Run optimization"):
    scenario = st.selectbox("Scenario", ["Min Cost (CO₂e cap)","Min Emissions (Production target)"])
    co2e_cap = st.number_input("CO₂e cap (kg)", value=float(df_g["co2e_kg"].mean()*1.1) if not df_g.empty else 1000.0)
    prod_target = st.number_input("Production target (ton)", value=float(df["production_ton"].mean()*24) if not df.empty else 100.0)

    price_elec, price_gas = 0.15, 0.08
    ef_elec, ef_gas = ef_elec_input, EMISSION_FACTOR_GAS

    if scenario.startswith("Min Cost"):
        obj = lambda x: price_elec*x[0] + price_gas*x[1]
        cons = [{"type":"ineq","fun":lambda x: co2e_cap - (ef_elec*x[0] + ef_gas*x[1])}]
    else:
        obj = lambda x: ef_elec*x[0] + ef_gas*x[1]
        cons = [{"type":"ineq","fun":lambda x: 0.02*x[0] + 0.05*x[1] - prod_target}]

    res = minimize(obj, [100,100], bounds=[(0,None),(0,None)], constraints=cons)
    e_opt,g_opt = res.x
    cost_opt = price_elec*e_opt + price_gas*g_opt
    co2e_opt = ef_elec*e_opt + ef_gas*g_opt

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Electricity (unit)", f"{e_opt:,.2f}")
    m2.metric("Gas (unit)", f"{g_opt:,.2f}")
    m3.metric("Total Cost", f"{cost_opt:,.2f}")
    m4.metric("CO₂e (kg)", f"{co2e_opt:,.2f}")

    st.dataframe(pd.DataFrame([{
        "electricity":round(e_opt,2),"gas":round(g_opt,2),
        "cost":round(cost_opt,2),"co2e":round(co2e_opt,2),
        "success":bool(res.success)
    }]), use_container_width=True)
