# =====================================================
# GreenOpt — Digital ESG Engine (FINAL • All-in-One • Ultra-Dark)
# Single-file app.py (self-contained)
# =====================================================
from __future__ import annotations

from pathlib import Path
from io import BytesIO
import json, uuid, hashlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

# PDF (optional; installed via requirements)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- Page ----------
st.set_page_config(page_title="GreenOpt — Carbon Intelligence Platform", layout="wide")

# ---------- Theme (Dark + White Text) ----------
BG   = "#0b0e11"   # app background
BG2  = "#111827"   # panels / sidebar
TXT  = "#ffffff"   # text white
GRID = "#1f2937"   # chart grid
BORDER = "#374151" # subtle borders
GREEN= "#22c55e"   # brand green
RED  = "#ef4444"   # red for decrease buttons

def apply_theme():
    st.markdown(f"""
    <style>
    html, body, .stApp, .block-container {{
        background: {BG} !important; color: {TXT} !important;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background: {BG2} !important; color: {TXT} !important;
    }}

    a, a:link, a:visited {{ color: {GREEN} !important; text-decoration: none !important; }}
    a:hover {{ text-decoration: underline !important; }}

    .stTextInput input, .stNumberInput input, .stDateInput input, select, textarea {{
        background: {BG2} !important; color: {TXT} !important;
        border: 1px solid {BORDER} !important; border-radius: 10px !important;
    }}
    div[data-baseweb="select"] > div {{
        background: {BG2} !important; color: {TXT} !important; border:1px solid {BORDER} !important;
    }}

    .stButton button, [data-baseweb="button"] {{
        background: #1e293b !important; color: {TXT} !important;
        border: 1px solid {BORDER} !important; border-radius: 10px !important;
    }}
    .stButton button:hover, [data-baseweb="button"]:hover {{
        background: {GREEN} !important; color: {TXT} !important;
    }}

    [data-testid="stDownloadButton"] > button,
    button[kind="secondary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-secondaryFormSubmit"] {{
        background: {BG2} !important; color: {TXT} !important;
        border: 1px solid {BORDER} !important; border-radius: 10px !important;
    }}
    [data-testid="stDownloadButton"] > button:hover,
    button[kind="secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-secondaryFormSubmit"]:hover {{
        background: {GREEN} !important; color: {TXT} !important;
        border-color: {GREEN} !important;
    }}

    /* NumberInput +/- */
    .stNumberInput div[data-baseweb="input"], .stNumberInput input {{
        background:{BG2} !important; color:{TXT} !important;
        border:1px solid {BORDER} !important; border-radius:10px !important;
    }}
    .stNumberInput button[aria-label="Decrease value"] {{ border:1px solid {RED} !important; }}
    .stNumberInput button[aria-label="Increase value"] {{ border:1px solid {GREEN} !important; }}

    /* Tabs */
    .stTabs [role="tablist"] {{ border-bottom: 1px solid {BORDER} !important; }}
    .stTabs [role="tab"] {{
        background:{BG2} !important; color:{TXT} !important;
        border:1px solid {BORDER} !important; border-bottom:none !important;
        margin-right:6px !important; border-top-left-radius:10px !important; border-top-right-radius:10px !important;
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background:#0f172a !important; border-color:{GREEN} !important; color:{TXT} !important;
    }}
    .stTabs div[role="tabpanel"] {{
        background:{BG2} !important; border:1px solid {BORDER} !important; border-top:none !important;
        border-bottom-left-radius:12px !important; border-bottom-right-radius:12px !important;
        color:{TXT} !important; padding:12px 10px !important;
    }}

    /* Tables / Metrics */
    [data-testid="stStyledTable"] thead th {{ background:#0f172a !important; color:{TXT} !important; }}
    [data-testid="stStyledTable"] tbody td {{ background:{BG2} !important; color:{TXT} !important; border-color:{GRID} !important; }}
    [data-testid="stTable"] th, [data-testid="stTable"] td {{ color:{TXT} !important; background:{BG2} !important; border-color:{BORDER} !important; }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color:{TXT} !important; }}

    .modebar {{ filter: invert(1) !important; }}
    </style>
    """, unsafe_allow_html=True)

def style_fig(fig: go.Figure, x_range=None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT),
        title_font=dict(color=TXT),
        legend_font=dict(color=TXT),
    )
    fig.update_xaxes(color=TXT, gridcolor=GRID, zerolinecolor=BORDER)
    fig.update_yaxes(color=TXT, gridcolor=GRID, zerolinecolor=BORDER)
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    return fig

apply_theme()

# ---------- Paths ----------
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()
ROOT = APP_DIR.parent
DATA_DIR = ROOT / "data"
ASSET_DIR = APP_DIR / "assets"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"

# ---------- Defaults ----------
EMISSION_FACTOR_ELECTRICITY_DEFAULT = 0.475  # kg/kWh
EMISSION_FACTOR_GAS = 2.0                    # kg/m3

# ---------- Utilities ----------
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
            "line": rng.choice(["A-Line","B-Line","C-Line"], periods, p=[0.4,0.4,0.2]),
            "product": rng.choice(["Widget-X","Widget-Y","Widget-Z"], periods),
        })
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)

def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"electricity_kwh":"sum","gas_m3":"sum","production_ton":"sum","co2e_kg":"sum","pcf_kg_per_ton":"mean"}
    return df.set_index("timestamp").resample(rule).agg(agg).reset_index()

def add_carbon_columns(df_in: pd.DataFrame, ef_elec: float) -> pd.DataFrame:
    out = df_in.copy()
    out["co2e_kg"] = out["electricity_kwh"]*float(ef_elec) + out["gas_m3"]*EMISSION_FACTOR_GAS
    out["pcf_kg_per_ton"] = np.where(out["production_ton"]>0, out["co2e_kg"]/out["production_ton"], np.nan)
    return out

# ---------- Header with logo ----------
lc, rc = st.columns([0.14, 0.86])
with lc:
    logo = None
    for p in [
        ASSET_DIR / "greenopt_logo.png",
        ASSET_DIR / "logo.png",
        ROOT / "assets" / "greenopt_logo.png",
        ROOT / "assets" / "logo.png",
    ]:
        if p.exists():
            logo = p; break
    if logo:
        st.image(Image.open(logo))
    else:
        st.caption("Tip: place logo at app/assets/greenopt_logo.png")
with rc:
    st.title("GreenOpt — AI Carbon Intelligence Platform")
    st.caption("Forecast • Optimization • Anomaly • Digital ESG")

st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV (3+ years preferred)", type=["csv"])
    if up:
        df = pd.read_csv(up)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        st.success("CSV loaded.")
    else:
        df = load_data(DEFAULT_CSV)
        st.info("Loaded default / generated sample data.")

    # Required / optional columns
    base_required = {"timestamp","electricity_kwh","gas_m3","production_ton"}
    missing_base = base_required - set(df.columns)
    if missing_base:
        st.error(f"Missing required columns: {', '.join(sorted(missing_base))}")
        st.stop()

    optional_warnings = []
    if "line" not in df.columns:
        df["line"] = "All-Line"; optional_warnings.append("line")
    if "product" not in df.columns:
        df["product"] = "All-Product"; optional_warnings.append("product")
    if optional_warnings:
        st.info("Optional columns not found → using single bucket: " + ", ".join(optional_warnings))

    st.header("Scope 2 method")
    scope2 = st.selectbox("Electricity EF method", ["Location-based","Market-based"], index=0)
    ef_elec_input = st.number_input(
        "EF (kg/kWh)" if scope2=="Location-based" else "EF (market-based, kg/kWh)",
        value=float(EMISSION_FACTOR_ELECTRICITY_DEFAULT if scope2=="Location-based" else 0.0),
        step=0.01
    )

    st.header("Filters")
    tmin_all = df["timestamp"].min().date()
    tmax_all = df["timestamp"].max().date()

    if "range_mode_key" not in st.session_state:
        st.session_state["range_mode_key"] = "All data"
    if st.session_state["range_mode_key"] not in ("All data","Custom"):
        st.session_state["range_mode_key"] = "All data"

    range_mode = st.radio("Range mode", ["All data","Custom"], horizontal=True, key="range_mode_key")

    if range_mode == "Custom":
        start_date, end_date = st.date_input(
            "Date range", value=(tmin_all, tmax_all),
            min_value=tmin_all, max_value=tmax_all,
            key="custom_date_range_key"
        )
        if st.button("Reset to full range", use_container_width=True):
            st.session_state["range_mode_key"] = "All data"
            st.session_state.pop("custom_date_range_key", None)
            st.experimental_rerun()
    else:
        start_date, end_date = tmin_all, tmax_all
        st.session_state.pop("custom_date_range_key", None)

    line_opts = sorted(pd.Series(df["line"]).dropna().unique().tolist())
    sel_lines = st.multiselect("Line", line_opts) if line_opts else []
    product_opts = sorted(pd.Series(df["product"]).dropna().unique().tolist())
    sel_products = st.multiselect("Product", product_opts) if product_opts else []

    rule = st.selectbox("Time granularity", ["H","D","W","M"], index=3)

    st.header("External Features")
    aux = st.file_uploader("AUX CSV (timestamp, temperature_c, utilization_pct ...)", type=["csv"], key="aux")
    if aux:
        aux_df = pd.read_csv(aux)
        aux_df["timestamp"] = pd.to_datetime(aux_df["timestamp"]).dt.tz_localize(None)
        df = df.merge(aux_df, on="timestamp", how="left")
        st.success("AUX merged.")
    else:
        t_days = (df["timestamp"] - df["timestamp"].min()).dt.days.values
        if "temperature_c" not in df.columns:
            df["temperature_c"] = 18 + 7*np.sin(2*np.pi*(t_days/365)) + np.random.normal(0, 1.5, len(df))
        if "utilization_pct" not in df.columns:
            base = 70 + 20*np.sin(2*np.pi*(t_days/30)) + np.random.normal(0, 5, len(df))
            df["utilization_pct"] = np.clip(base, 20, 100)

# ---------- Apply filters ----------
if range_mode == "Custom":
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_f = df[(df["timestamp"]>=start_dt) & (df["timestamp"]<=end_dt)].copy()
else:
    df_f = df.copy()

if sel_lines:
    df_f = df_f[df_f["line"].isin(sel_lines)]
if sel_products:
    df_f = df_f[df_f["product"].isin(sel_products)]
df_f = df_f.sort_values("timestamp").reset_index(drop=True)

# Too short? fallback to full
full_span = (df["timestamp"].max() - df["timestamp"].min()).days + 1
sel_span  = (df_f["timestamp"].max() - df_f["timestamp"].min()).days + 1 if not df_f.empty else 0
if sel_span == 0 or sel_span < max(1, int(full_span * 0.30)):
    df_f = df.copy()

# ---------- Carbon + Resample ----------
df_c = add_carbon_columns(df_f, ef_elec_input)
df_g = resample_df(df_c, rule)

# ---------- KPIs ----------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total CO₂e (kg)", f"{df_g['co2e_kg'].sum():,.0f}")
c2.metric("Avg PCF (kg/ton)", f"{df_g['pcf_kg_per_ton'].mean():,.2f}")
last_val = df_g["co2e_kg"].iloc[-1] if not df_g.empty else 0.0
c3.metric(f"Last {rule} CO₂e (kg)", f"{last_val:,.1f}")
c4.metric("Periods", f"{len(df_g):,}")

# ---------- Main Chart ----------
st.subheader("Time-series overview")
if not df_g.empty:
    y = df_g["co2e_kg"].astype(float)
    x = pd.to_datetime(df_g["timestamp"])
    trace_mode = "lines+markers" if len(df_g) >= 2 else "markers"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode=trace_mode, name="CO₂e (kg)",
        line=dict(color=GREEN, width=2.4),
        marker=dict(size=6, color=GREEN)
    ))
    xmin = pd.to_datetime(df["timestamp"].min())
    xmax = pd.to_datetime(df["timestamp"].max())
    if pd.notna(xmin) and pd.notna(xmax) and xmin < xmax:
        fig = style_fig(fig, x_range=[xmin, xmax])
    else:
        fig = style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data in selected range.")

# ---------- STL (optional) ----------
def section_stl(df_g: pd.DataFrame):
    st.subheader("Seasonal-Trend Decomposition (STL)")
    with st.expander("Show STL"):
        try:
            import statsmodels.api as sm
        except Exception:
            st.info("`statsmodels` 미설치 — requirements.txt에 `statsmodels==0.14.3` 추가 시 활성화됩니다.")
            return
        try:
            s = df_g.set_index("timestamp")["co2e_kg"]
            step = df_g["timestamp"].diff().mode()[0]
            s = s.asfreq(step, method="pad")
            stl = sm.tsa.STL(s, robust=True).fit()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stl.trend.index,   y=stl.trend.values,   name="Trend",    line=dict(color=GREEN)))
            fig.add_trace(go.Scatter(x=stl.seasonal.index,y=stl.seasonal.values,name="Seasonal"))
            fig.add_trace(go.Scatter(x=stl.resid.index,   y=stl.resid.values,   name="Residual"))
            st.plotly_chart(style_fig(fig), use_container_width=True)
        except Exception as e:
            st.info(f"STL skipped: {e}")

# ---------- Anomaly ----------
def section_anomaly(df_g: pd.DataFrame):
    st.subheader("Anomaly Detection")
    with st.expander("Detect anomalies"):
        if len(df_g) < 30:
            st.info("Need at least 30 periods."); return
        X = df_g[["co2e_kg"]].fillna(method="ffill")
        iso = IsolationForest(contamination=0.02, random_state=42)
        labels = iso.fit_predict(X)
        v = df_g.copy(); v["anomaly"] = (labels == -1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v["timestamp"], y=v["co2e_kg"], mode="lines",
                                 name="CO₂e", line=dict(color=GREEN, width=2.0)))
        aa = v[v["anomaly"]]
        fig.add_trace(go.Scatter(x=aa["timestamp"], y=aa["co2e_kg"], mode="markers",
                                 name="Anomaly", marker=dict(size=8, symbol="x", color="#FCA5A5")))
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ---------- Forecast ----------
def section_forecast(df_g: pd.DataFrame):
    st.subheader("Forecasting")
    with st.expander("Train & forecast"):
        if len(df_g) < 20:
            st.info("Need more data to forecast."); return
        horizon = st.slider("Forecast horizon (periods)", 7 if rule=="D" else 24, 60, 14)
        dff = df_g[["timestamp","co2e_kg"]].copy()
        dff["lag1"] = dff["co2e_kg"].shift(1)
        dff = dff.dropna().reset_index(drop=True)
        if len(dff) <= horizon + 1:
            st.info("Not enough data after feature engineering."); return
        tr, te = dff.iloc[:-horizon], dff.iloc[-horizon:]
        y_tr, y_te = tr["co2e_kg"], te["co2e_kg"]
        X_tr, X_te = tr[["lag1"]], te[["lag1"]]
        gbr = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
        pred = gbr.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        a,b = st.columns(2)
        a.metric("MAE", f"{mae:,.2f}")
        b.metric("Last pred", f"{pred[-1]:,.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr["timestamp"], y=tr["co2e_kg"], name="Train"))
        fig.add_trace(go.Scatter(x=te["timestamp"], y=y_te, name="Actual"))
        fig.add_trace(go.Scatter(x=te["timestamp"], y=pred, name="Forecast", line=dict(color=GREEN, width=2.2)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.session_state["pred_series"] = pd.Series(pred, index=te["timestamp"])

# ---------- Optimization ----------
def section_optimization(df_g: pd.DataFrame, df_c: pd.DataFrame):
    st.subheader("Optimization (toy)")
    with st.expander("Run optimization"):
        scenario = st.selectbox("Scenario", ["Min Cost (CO₂e cap)","Min Emissions (Production target)"])
        co2e_cap = st.number_input("CO₂e cap (kg)", value=float(df_g["co2e_kg"].quantile(0.75)) if not df_g.empty else 1000.0, step=50.0)
        prod_tgt  = st.number_input("Production target (ton)", value=float(df_c["production_ton"].mean()*24) if not df_c.empty else 100.0, step=10.0)
        price_e, price_g = 0.15, 0.08
        ef_e, ef_g = ef_elec_input, EMISSION_FACTOR_GAS

        if scenario.startswith("Min Cost"):
            def obj(x): e,g=x; return price_e*e + price_g*g
            cons = [{"type":"ineq","fun":lambda x: co2e_cap - (ef_e*x[0] + ef_g*x[1])}]
            x0 = [co2e_cap/max(ef_e,1e-9)*0.5, co2e_cap/ef_g*0.5]
        else:
            def obj(x): e,g=x; return ef_e*e + ef_g*g
            alpha, beta = 0.02, 0.05
            cons = [{"type":"ineq","fun":lambda x: (alpha*x[0] + beta*x[1]) - prod_tgt}]
            x0 = [prod_tgt/alpha*0.5, prod_tgt/beta*0.5]

        res = minimize(obj, x0, bounds=[(0,None),(0,None)], constraints=cons)
        e_opt, g_opt = float(res.x[0]), float(res.x[1])
        cost_opt = price_e*e_opt + price_g*g_opt
        co2e_opt = ef_e*e_opt + ef_g*g_opt
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Electricity (unit)", f"{e_opt:,.2f}")
        m2.metric("Gas (unit)", f"{g_opt:,.2f}")
        m3.metric("Total Cost", f"{cost_opt:,.2f}")
        m4.metric("CO₂e (kg)", f"{co2e_opt:,.2f}")
        st.dataframe(pd.DataFrame([{
            "electricity":round(e_opt,2),"gas":round(g_opt,2),
            "cost":round(cost_opt,2),"co2e":round(co2e_opt,2),"success":bool(res.success)
        }]), use_container_width=True)

# ---------- Carbon Pricing ----------
def section_carbon_pricing():
    st.subheader("Carbon Pricing")
    with st.expander("Apply price"):
        price_per_t = st.number_input("Carbon price (per tCO₂e)", value=85.0, step=1.0)
        df_cost = pd.DataFrame({
            "timestamp": df_g["timestamp"],
            "cost_local": (df_g["co2e_kg"]/1000.0) * price_per_t
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cost["timestamp"], y=df_cost["cost_local"], mode="lines",
                                 name="Carbon Cost", line=dict(color=GREEN, width=2.1)))
        st.plotly_chart(style_fig(fig), use_container_width=True)
        st.dataframe(df_cost.tail(12), use_container_width=True)

# ---------- Partner Hub ----------
def section_partner_hub():
    st.subheader("Partner Hub — Benchmark • Invite • Trust")
    tab_b, tab_i, tab_t = st.tabs(["Benchmark","Invite","Trust"])

    with tab_b:
        st.caption("Upload partner benchmark CSV (timestamp, product, line, pcf_kg_per_ton)")
        f = st.file_uploader("Partner CSV", type=["csv"], key="pb")
        if f:
            pdf_ = pd.read_csv(f)
            if "pcf_kg_per_ton" not in pdf_.columns:
                st.error("Missing 'pcf_kg_per_ton'")
            else:
                ours = df_g["pcf_kg_per_ton"].mean()
                peers = pdf_["pcf_kg_per_ton"].mean()
                a,b,c = st.columns(3)
                a.metric("Our Avg PCF", f"{ours:,.2f}")
                b.metric("Peer Avg PCF", f"{peers:,.2f}")
                c.metric("Gap (peer-us)", f"{peers-ours:,.2f}")

    with tab_i:
        code = st.text_input("Invite code", value=str(uuid.uuid4()))
        if not df_g.empty:
            sample = df_g.tail(8)[["timestamp","co2e_kg","pcf_kg_per_ton"]].copy()
            sample["timestamp"] = pd.to_datetime(sample["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
            for col in ["co2e_kg","pcf_kg_per_ton"]:
                sample[col] = pd.to_numeric(sample[col], errors="coerce").astype(float)
            pack = {
                "title":"GreenOpt Partner Brief",
                "kpi":{
                    "total_co2e_kg": float(df_g["co2e_kg"].sum()),
                    "avg_pcf": float(df_g["pcf_kg_per_ton"].mean()),
                },
                "sample": sample.to_dict(orient="records"),
                "invite_code": code,
            }
            pack_bytes = json.dumps(pack, ensure_ascii=False, indent=2, default=str).encode("utf-8")
            st.download_button("⬇️ Download Partner Pack (JSON)", data=pack_bytes,
                               file_name="greenopt_partner_pack.json", mime="application/json")
        else:
            st.info("Need data to build a partner pack.")

    with tab_t:
        payload = pd.DataFrame({
            "timestamp": df_g["timestamp"].astype(str),
            "co2e_kg": df_g["co2e_kg"].round(6)
        }).to_csv(index=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        st.text_input("SHA256(data_slice)", value=digest, disabled=True)
        st.caption(f"Scope2: {scope2} • EF_electricity(kg/kWh): {ef_elec_input} • EF_gas(kg/m³): {EMISSION_FACTOR_GAS}")

# ---------- PDF Export ----------
def section_pdf():
    st.subheader("Export KPI / Report (PDF)")

    def build_pdf(df_summary: pd.DataFrame, kpis: dict) -> bytes:
        buf = BytesIO(); c = canvas.Canvas(buf, pagesize=A4); w,h = A4; y = h-50
        c.setFont("Helvetica-Bold", 16); c.drawString(40,y, "GreenOpt — Carbon Intelligence Report"); y-=25
        c.setFont("Helvetica", 10); c.drawString(40,y, f"Scope2: {scope2} | EF_elec: {ef_elec_input} kg/kWh"); y-=15
        c.drawString(40,y, f"Period: {str(df_summary['timestamp'].min().date())} ~ {str(df_summary['timestamp'].max().date())}"); y-=25
        c.setFont("Helvetica-Bold", 12); c.drawString(40,y, "KPIs"); y-=18
        c.setFont("Helvetica", 10)
        for k,v in kpis.items(): c.drawString(50,y, f"- {k}: {v}"); y-=14
        c.showPage(); c.save(); buf.seek(0); return buf.read()

    if not df_g.empty:
        kpis = {
            "Total CO₂e (kg)": f"{df_g['co2e_kg'].sum():,.0f}",
            "Avg PCF (kg/ton)": f"{df_g['pcf_kg_per_ton'].mean():,.2f}",
            f"Periods ({rule})": f"{len(df_g):,}",
        }
        pdf = build_pdf(df_g, kpis)
        st.download_button("📄 Download KPI Report (PDF)", data=pdf, file_name="greenopt_report.pdf", mime="application/pdf")
    else:
        st.info("No data to export.")

# ---------- Render sections ----------
section_stl(df_g)
section_anomaly(df_g)
section_forecast(df_g)
section_optimization(df_g, df_c)
section_carbon_pricing()
section_partner_hub()
section_pdf()

# CSS priority (apply again at bottom)
apply_theme()
