# =====================================================
# GreenOpt — Focused Stable App (Dark tabs + reliable charts)
# - Minimal deps: streamlit, plotly, pandas, numpy
# - Graphs: always reflect data, safe with 1~N points
# - Tabs: active/inactive ALL dark
# =====================================================
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- Page ----------
st.set_page_config(page_title="GreenOpt — Carbon Intelligence (Stable)", layout="wide")

# ---------- Theme (pure dark, tabs always dark) ----------
BG   = "#0b0e11"   # app background
BG2  = "#111827"   # panels / sidebar
TXT  = "#ffffff"   # text
GRID = "#1f2937"   # chart grid
BORDER = "#374151" # border
GREEN= "#22c55e"   # brand green
RED  = "#ef4444"   # red (decrease button etc.)

def apply_theme():
    st.markdown(f"""
    <style>
    html, body, .stApp, .block-container {{
        background: {BG} !important; color: {TXT} !important;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    * {{ color: {TXT}; }}

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: {BG2} !important; color: {TXT} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {TXT} !important; }}

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background: {BG2} !important; border: 1px dashed {BORDER} !important; color: {TXT} !important;
    }}
    [data-testid="stFileUploader"] button {{ background: {BG2} !important; color: {TXT} !important; border:1px solid {BORDER} !important; }}

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stDateInput input, select, textarea {{
        background: {BG2} !important; color: {TXT} !important; border:1px solid {BORDER} !important; border-radius:10px !important;
    }}
    div[data-baseweb="select"] > div {{ background:{BG2}!important; color:{TXT}!important; border:1px solid {BORDER}!important; }}

    /* Tabs: FORCE DARK ALWAYS (active/inactive) */
    .stTabs [role="tablist"] {{ border-bottom: 1px solid {BORDER} !important; }}
    .stTabs [role="tab"] {{
        background:{BG2}!important; color:{TXT}!important; border:1px solid {BORDER}!important;
        margin-right:6px!important; border-top-left-radius:10px!important; border-top-right-radius:10px!important;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background:#0f172a!important; border-color:{GREEN}!important; color:{TXT}!important;
    }}
    .stTabs [role="tab"]:hover {{ background: rgba(34,197,94,.12)!important; }}
    .stTabs div[role="tabpanel"] {{
        background:{BG2}!important; border:1px solid {BORDER}!important; border-top:none!important;
        border-bottom-left-radius:12px!important; border-bottom-right-radius:12px!important; padding:12px 10px!important;
    }}

    /* Number +/- : dark + red/green */
    .stNumberInput div[data-baseweb="input"], .stNumberInput input {{
        background:{BG2}!important; color:{TXT}!important; border:1px solid {BORDER}!important; border-radius:10px!important;
    }}
    .stNumberInput button[aria-label="Decrease value"] {{ border:1px solid {RED}!important; }}
    .stNumberInput button[aria-label="Decrease value"] svg * {{ fill:{RED}!important; stroke:{RED}!important; }}
    .stNumberInput button[aria-label="Increase value"] {{ border:1px solid {GREEN}!important; }}
    .stNumberInput button[aria-label="Increase value"] svg * {{ fill:{GREEN}!important; stroke:{GREEN}!important; }}

    /* Tables / Metrics */
    [data-testid="stStyledTable"] thead th {{ background:#0f172a!important; color:{TXT}!important; }}
    [data-testid="stTable"] th, [data-testid="stTable"] td {{ color:{TXT}!important; background:{BG2}!important; border-color:{BORDER}!important; }}

    /* Plotly toolbar */
    .modebar {{ filter: invert(1) !important; }}
    </style>
    """, unsafe_allow_html=True)

def style_fig(fig: go.Figure, x_range=None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT), title_font=dict(color=TXT),
        legend_font=dict(color=TXT)
    )
    fig.update_xaxes(color=TXT, gridcolor=GRID, zerolinecolor=BORDER)
    fig.update_yaxes(color=TXT, gridcolor=GRID, zerolinecolor=BORDER)
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    return fig

apply_theme()

# ---------- Data helpers ----------
DATA_DIR = Path("data")
DEFAULT_CSV = DATA_DIR / "factory_data.csv"

REQUIRED = {"timestamp","electricity_kwh","gas_m3","production_ton"}

def load_or_sample(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        # generate 3y hourly sample
        periods = 24*365*3
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
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)

def add_carbon(df: pd.DataFrame, ef_elec: float, ef_gas: float=2.0) -> pd.DataFrame:
    out = df.copy()
    out["co2e_kg"] = out["electricity_kwh"]*ef_elec + out["gas_m3"]*ef_gas
    out["pcf_kg_per_ton"] = np.where(out["production_ton"]>0, out["co2e_kg"]/out["production_ton"], np.nan)
    return out

def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"electricity_kwh":"sum","gas_m3":"sum","production_ton":"sum","co2e_kg":"sum","pcf_kg_per_ton":"mean"}
    return df.set_index("timestamp").resample(rule).agg(agg).reset_index()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV (3+ years preferred)", type=["csv"])
    if up:
        df = pd.read_csv(up)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        st.success("CSV loaded.")
    else:
        df = load_or_sample(DEFAULT_CSV)
        st.info("Loaded default / generated 3-year sample.")

    miss = REQUIRED - set(df.columns)
    if miss:
        st.error("Missing columns: " + ", ".join(sorted(miss)))
        st.stop()

    if "line" not in df.columns: df["line"] = "All-Line"
    if "product" not in df.columns: df["product"] = "All-Product"

    st.header("Scope 2")
    scope2 = st.selectbox("Method", ["Location-based","Market-based"], index=0)
    ef = st.number_input("EF (kg/kWh)" if scope2=="Location-based" else "EF (market-based, kg/kWh)",
                         value=0.475 if scope2=="Location-based" else 0.0, step=0.01)

    st.header("Filters")
    tmin, tmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    range_mode = st.radio("Range", ["All data","Custom"], horizontal=True, key="range_mode")
    if range_mode == "Custom":
        sd, ed = st.date_input("Date range", value=(tmin, tmax), min_value=tmin, max_value=tmax)
    else:
        sd, ed = tmin, tmax

    sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique().tolist()))
    sel_prod  = st.multiselect("Product", sorted(df["product"].dropna().unique().tolist()))
    rule = st.selectbox("Time granularity", ["H","D","W","M"], index=3)  # default: Monthly

# ---------- Apply filters safely (never lose data accidentally) ----------
if range_mode == "Custom":
    sdt = pd.to_datetime(sd)
    edt = pd.to_datetime(ed) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_f = df[(df["timestamp"]>=sdt) & (df["timestamp"]<=edt)].copy()
else:
    df_f = df.copy()

if sel_lines: df_f = df_f[df_f["line"].isin(sel_lines)]
if sel_prod:  df_f = df_f[df_f["product"].isin(sel_prod)]
df_f = df_f.sort_values("timestamp").reset_index(drop=True)

# 너무 좁힌 경우 자동 전체로 롤백 (선택 범위가 전체의 30% 미만)
full_span = (df["timestamp"].max() - df["timestamp"].min()).days + 1
sel_span  = (df_f["timestamp"].max() - df_f["timestamp"].min()).days + 1 if not df_f.empty else 0
if sel_span == 0 or sel_span < max(1, int(full_span*0.30)):
    df_f = df.copy()

# ---------- Carbon + Resample ----------
df_c = add_carbon(df_f, ef)
df_g = resample(df_c, rule)

# ---------- Header ----------
st.title("GreenOpt — Carbon Intelligence")
st.caption("Dark tabs that stay dark • Charts that always show data")

# ---------- Tabs (항상 다크) ----------
tab1, tab2 = st.tabs(["Overview", "Data"])

with tab1:
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total CO₂e (kg)", f"{df_g['co2e_kg'].sum():,.0f}")
    c2.metric("Avg PCF (kg/ton)", f"{df_g['pcf_kg_per_ton'].mean():,.2f}")
    last = df_g["co2e_kg"].iloc[-1] if not df_g.empty else np.nan
    c3.metric(f"Last {rule} CO₂e (kg)", f"{last:,.1f}" if pd.notna(last) else "N/A")
    c4.metric("Periods", f"{len(df_g):,}")

    # Main chart — robust: points 1~N, full-range x-axis, y-padding
    st.subheader("Time-series overview")
    if not df_g.empty:
        x = pd.to_datetime(df_g["timestamp"])
        y = pd.to_numeric(df_g["co2e_kg"], errors="coerce").astype(float)
        mode = "lines+markers" if len(y) >= 2 else "markers"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode=mode, name="CO₂e (kg)",
            line=dict(color=GREEN, width=2.6),
            marker=dict(size=6, color=GREEN)
        ))

        # X축: 항상 원본 df 전체 기간으로(3년 보장). 단 min==max면 생략
        xmin = pd.to_datetime(df["timestamp"].min())
        xmax = pd.to_datetime(df["timestamp"].max())
        if pd.notna(xmin) and pd.notna(xmax) and xmin < xmax:
            fig = style_fig(fig, x_range=[xmin, xmax])
        else:
            fig = style_fig(fig)

        # y축: 변화가 거의 없어도 보이도록 패딩
        yv = y[~np.isnan(y)]
        if len(yv) > 0:
            y_min, y_max = float(np.min(yv)), float(np.max(yv))
            span = y_max - y_min
            pad = max(1.0, (abs(y_min)+1e-9)*0.01) if span <= 0 else max(span*0.06, 1e-6)
            fig.update_yaxes(range=[y_min - pad, y_max + pad])

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data in selected range.")

with tab2:
    st.subheader("Resampled data")
    st.dataframe(df_g, use_container_width=True)
    # 간단 다운로드 (Streamlit 내부 기본 스타일 → 위 CSS가 다크로 감쌈)
    st.download_button(
        "⬇️ Download CSV",
        data=df_g.to_csv(index=False).encode("utf-8"),
        file_name="greenopt_resampled.csv",
        mime="text/csv"
    )

# 마지막에 다시 적용(우선순위 보장)
apply_theme()
