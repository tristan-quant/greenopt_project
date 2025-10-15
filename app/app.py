# =====================================================
# GreenOpt — Digital ESG Engine (Dark + Green, Minimal Stable)
# Forecast • Anomaly • Scope2 • CBAM • PDF • Partner Hub
# =====================================================
from __future__ import annotations

import sys, subprocess
def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)
        except Exception:
            pass

for pkg in [
    "streamlit", "pandas", "numpy", "plotly", "scipy", "Pillow",
    "scikit-learn", "statsmodels", "reportlab"
]:
    _ensure(pkg)

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

_HAS_PLOTLY = False
_HAS_STATSMODELS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    sm = None

from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- Page config & theme ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")

GREEN = "#22C55E"
BG    = "#0E1117"
BG2   = "#111827"
TXT   = "#F3F4F6"

def init_theme():
    if _HAS_PLOTLY:
        import plotly.io as pio
        base = {
            "layout": {
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "font": {"color": TXT},
                "xaxis": {"gridcolor": "#374151", "zerolinecolor": "#374151"},
                "yaxis": {"gridcolor": "#374151", "zerolinecolor": "#374151"},
                "colorway": [GREEN, "#10B981", "#34D399", "#6EE7B7"],
            }
        }
        pio.templates["greenopt_dark"] = go.layout.Template(base)
        pio.templates.default = "greenopt_dark"
    st.markdown(f"""
    <style>
      .stApp {{ background:{BG}; color:{TXT}; }}
      [data-testid="stSidebar"], .stSidebar {{ background:{BG2} !important; color:{TXT} !important; }}
      [data-testid="stSidebar"] * {{ color:{TXT} !important; }}
      .block {{ background:{BG2}; border-radius:16px; padding:16px; }}
      .stMarkdown, .stText, .stCaption {{ color:{TXT}; }}
      a {{ color:{GREEN}; }}
      .modebar-group * {{ filter: invert(88%) !important; }}
    </style>
    """, unsafe_allow_html=True)

init_theme()

# paths
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()
ROOT = APP_DIR.parents[0]
DATA_DIR = ROOT / "data"
ASSET_DIR = APP_DIR / "assets"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"

# Emission factors
EMISSION_FACTOR_ELECTRICITY_DEFAULT = 0.475
EMISSION_FACTOR_GAS = 2.0

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        periods = 24*365*3
        idx = pd.date_range("2023-01-01", periods=periods, freq="H")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "timestamp": idx,
            "electricity_kwh": rng.uniform(80, 220, periods),
            "gas_m3": rng.uniform(8, 35, periods),
            "production_ton": rng.uniform(4, 18, periods),
            "line": rng.choice(["A-Line","B-Line","C-Line"], periods),
            "product": rng.choice(["Widget-X","Widget-Y","Widget-Z"], periods),
        })
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "electricity_kwh":"sum",
        "gas_m3":"sum",
        "production_ton":"sum",
        "co2e_kg":"sum",
        "pcf_kg_per_ton":"mean"
    }
    return (df.set_index("timestamp")
              .resample(rule)
              .agg(agg)
              .reset_index())

def _capabilities_banner():
    st.caption(f"Capabilities — plotly: {_HAS_PLOTLY} | statsmodels: {_HAS_STATSMODELS}")

# Header
h1, h2 = st.columns([0.12, 0.88])
with h1:
    logo_candidates = [
        ASSET_DIR / "greenopt_logo.png",
        ASSET_DIR / "logo.png",
        ROOT / "assets" / "greenopt_logo.png",
        ROOT / "assets" / "logo.png",
    ]
    logo = next((p for p in logo_candidates if p.exists()), None)
    if logo:
        st.image(Image.open(logo))
    else:
        st.caption("No logo found.")
with h2:
    st.title("GreenOpt — AI Carbon Intelligence Platform")
    st.caption("Forecast • Optimization • Anomaly • Analytics")
    _capabilities_banner()
st.divider()

# Sidebar
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (3+ years)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        st.success("Loaded uploaded CSV.")
    else:
        df = load_data(DEFAULT_CSV)
        st.info("Loaded default or generated sample.")

    # 필수 컬럼 체크
    required = {"timestamp","electricity_kwh","gas_m3","production_ton"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    st.header("Scope 2 method")
    scope2_method = st.selectbox("Electricity EF method", ["Location-based", "Market-based"], index=0)
    if scope2_method == "Location-based":
        ef_elec_input = st.number_input("EF (kg/kWh)", value=EMISSION_FACTOR_ELECTRICITY_DEFAULT, step=0.01)
    else:
        ef_elec_input = st.number_input("EF (market-based kg/kWh)", value=0.0, step=0.01)

    st.header("Filters")
    tmin, tmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    start_date, end_date = st.date_input("Date range", value=(tmin,tmax), min_value=tmin, max_value=tmax)
    sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique()) if "line" in df.columns else [])
    sel_products = st.multiselect("Product", sorted(df["product"].dropna().unique()) if "product" in df.columns else [])
    rule = st.selectbox("Time granularity", ["H","D","W","M"], index=1)

with st.sidebar:
    st.header("External Features")
    aux = st.file_uploader("Upload AUX CSV", type=["csv"])
    if aux:
        df_aux = pd.read_csv(aux)
        df_aux["timestamp"] = pd.to_datetime(df_aux["timestamp"]).dt.tz_localize(None)
        df = df.merge(df_aux, on="timestamp", how="left")
        st.success("AUX merged.")
    else:
        t_days = (df["timestamp"] - df["timestamp"].min()).dt.days.values
        if "temperature_c" not in df.columns:
            df["temperature_c"] = 18 + 7*np.sin(2*np.pi*(t_days/365)) + np.random.normal(0,1.5,len(df))
        if "utilization_pct" not in df.columns:
            base = 70 + 20*np.sin(2*np.pi*(t_days/30)) + np.random.normal(0,5,len(df))
            df["utilization_pct"] = np.clip(base, 20, 100)

# Apply filters
mask = (df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
if sel_lines and "line" in df.columns: mask &= df["line"].isin(sel_lines)
if sel_products and "product" in df.columns: mask &= df["product"].isin(sel_products)
df = df.loc[mask].copy()

def add_carbon_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    dfc = df_in.copy()
    dfc["co2e_kg"] = dfc["electricity_kwh"] * ef_elec_input + dfc["gas_m3"] * EMISSION_FACTOR_GAS
    dfc["pcf_kg_per_ton"] = np.where(dfc["production_ton"]>0, dfc["co2e_kg"]/dfc["production_ton"], np.nan)
    return dfc

df = add_carbon_columns(df)
df_g = resample_df(df, rule)

def show_kpis(dfi: pd.DataFrame):
    total = dfi["co2e_kg"].sum()
    avg_pcf = dfi["pcf_kg_per_ton"].mean()
    last = dfi.iloc[-1]["co2e_kg"] if not dfi.empty else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total CO₂e (kg)", f"{total:,.0f}")
    c2.metric("Avg PCF (kg/ton)", f"{avg_pcf:,.2f}" if np.isfinite(avg_pcf) else "N/A")
    c3.metric(f"Last {rule} CO₂e (kg)", f"{last:,.1f}" if np.isfinite(last) else "N/A")
    c4.metric("Periods", f"{len(dfi):,}")

show_kpis(df_g)

st.subheader("Time-series overview")
if not df_g.empty:
    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"], mode="lines",
                                 name="CO₂e", line=dict(color=GREEN, width=2.5)))
        fig.update_layout(title="CO₂e (resampled)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots(facecolor=BG2)
        ax.set_facecolor(BG2)
        ax.plot(df_g["timestamp"], df_g["co2e_kg"], color=GREEN)
        st.pyplot(fig, use_container_width=True)
else:
    st.warning("No data in selected range")

st.subheader("Anomaly Detection")
with st.expander("Detect anomalies"):
    if len(df_g) >= 30:
        X = df_g[["co2e_kg"]].fillna(method="ffill")
        iso = IsolationForest(contamination=0.02, random_state=42)
        labels = iso.fit_predict(X)
        df_g["anomaly"] = (labels == -1)
        if _HAS_PLOTLY:
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"], mode="lines",
                                       name="CO₂e", line=dict(color=GREEN, width=2.2)))
            anom = df_g[df_g["anomaly"]]
            fig_a.add_trace(go.Scatter(x=anom["timestamp"], y=anom["co2e_kg"], mode="markers",
                                       name="Anomaly", marker=dict(size=8, symbol="x", color="#FCA5A5")))
            fig_a.update_layout(title="Anomaly detection")
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.write(df_g[df_g["anomaly"]].head())
    else:
        st.info("Need at least 30 periods")

# Forecasting (최소 형태)
st.subheader("Forecasting")
with st.expander("Train & forecast"):
    horizon = st.slider("Forecast horizon (periods)", 1, 30, 7)
    dff = df_g.copy()
    dff["lag1"] = dff["co2e_kg"].shift(1)
    dff = dff.dropna().reset_index(drop=True)
    if len(dff) > horizon + 1:
        train = dff.iloc[:-horizon]
        test = dff.iloc[-horizon:]
        y_train, y_test = train["co2e_kg"], test["co2e_kg"]
        X_train, X_test = train[["lag1"]], test[["lag1"]]
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X_train, y_train)
        pred = gbr.predict(X_test)
        mae = mean_absolute_error(y_test, pred)

        colA, colB = st.columns(2)
        colA.metric("MAE", f"{mae:,.2f}")
        colB.metric("Last pred", f"{pred[-1]:,.2f}")
        if _HAS_PLOTLY:
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=train["timestamp"], y=train["co2e_kg"], name="Train"))
            fig_f.add_trace(go.Scatter(x=test["timestamp"], y=y_test, name="Actual"))
            fig_f.add_trace(go.Scatter(x=test["timestamp"], y=pred, name="Forecast",
                                       line=dict(color=GREEN, width=2.2)))
            fig_f.update_layout(title="Forecast")
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.write("Forecast done")
    else:
        st.info("Not enough data to forecast")

# Carbon pricing
st.subheader("Carbon Cost")
with st.expander("Apply pricing"):
    price_t = st.number_input("Carbon price per tCO₂e", value=50.0)
    if "co2e_kg" in st.session_state or not df_g.empty:
        df_cost = pd.DataFrame({
            "timestamp": df_g["timestamp"],
            "cost_local": df_g["co2e_kg"] / 1000 * price_t
        })
        if _HAS_PLOTLY:
            fig_c = px.line(df_cost, x="timestamp", y="cost_local", title="Carbon Cost")
            for tr in fig_c.data:
                tr.line.color = GREEN
                tr.line.width = 2.2
            st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(df_cost.tail(10))
    else:
        st.info("No data to price")

# Partner Hub
st.subheader("Partner Hub")
tab_bench, tab_inv, tab_trust = st.tabs(["Benchmark", "Invite", "Trust"])
with tab_bench:
    st.markdown("Benchmark upload CSV (pcf_kg_per_ton)")
    f = st.file_uploader("Partner CSV", type=["csv"], key="pb")
    if f:
        dfp = pd.read_csv(f)
        if "pcf_kg_per_ton" not in dfp.columns:
            st.error("Need column pcf_kg_per_ton")
        else:
            ours = df_g["pcf_kg_per_ton"].mean()
            peers = dfp["pcf_kg_per_ton"].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Our avg", f"{ours:,.2f}")
            c2.metric("Peer avg", f"{peers:,.2f}")
            c3.metric("Gap", f"{peers-ours:,.2f}")
with tab_inv:
    import uuid, json
    invite_code = st.text_input("Invite code", value=str(uuid.uuid4()))
    if not df_g.empty:
        sample = df_g.tail(5)[["timestamp","co2e_kg","pcf_kg_per_ton"]].copy()
        sample["timestamp"] = pd.to_datetime(sample["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        for col in ["co2e_kg","pcf_kg_per_ton"]:
            sample[col] = pd.to_numeric(sample[col], errors="coerce").astype(float)
        pack = {
            "title": "Partner Pack",
            "kpi": {
                "total_co2e_kg": float(df_g["co2e_kg"].sum()),
                "avg_pcf": float(df_g["pcf_kg_per_ton"].mean())
            },
            "sample": sample.to_dict(orient="records"),
            "invite_code": invite_code
        }
        pack_bytes = json.dumps(pack, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        st.download_button("Download Pack", data=pack_bytes, file_name="pack.json", mime="application/json")
with tab_trust:
    import hashlib
    payload = pd.DataFrame({
        "timestamp": df_g["timestamp"].astype(str),
        "co2e_kg": df_g["co2e_kg"].round(6)
    }).to_csv(index=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    st.code(f"SHA256 = {digest}")

# PDF export
st.subheader("Export Report")
def build_pdf(df_summary: pd.DataFrame, kpis: dict, note: str) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "GreenOpt Report")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Scope2: {scope2_method} | EF_elec: {ef_elec_input}")
    y -= 15
    c.drawString(40, y, f"Period: {start_date} ~ {end_date}")
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "KPIs")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in kpis.items():
        c.drawString(50, y, f"- {k}: {v}")
        y -= 14
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

if not df_g.empty:
    kpis = {
        "Total CO₂e (kg)": f"{df_g['co2e_kg'].sum():,.0f}",
        "Avg PCF (kg/ton)": f"{df_g['pcf_kg_per_ton'].mean():,.2f}"
    }
    pdf_bytes = build_pdf(df_g, kpis, "Report note")
    st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
