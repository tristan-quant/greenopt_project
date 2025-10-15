# =====================================================
# GreenOpt — Digital ESG Engine (Dark+Green, Cloud-Stable)
# Forecast • Anomaly • Scope2 • CBAM • PDF • Partner Hub
# =====================================================
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

# (옵션) Plotly / statsmodels
_HAS_PLOTLY = False
_HAS_STATSMODELS = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    px = None; go = None
try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    sm = None

# ---------- Page & Theme ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")

GREEN = "#22C55E"
BG    = "#0E1117"   # 앱 전체 배경
BG2   = "#111827"   # 카드/사이드
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

    # 강제 CSS (사이드바 포함)
    st.markdown(f"""
    <style>
      .stApp {{ background:{BG}; color:{TXT}; }}
      [data-testid="stHeader"] {{ background: transparent; }}

      /* 사이드바 전체 다크 */
      [data-testid="stSidebar"], .stSidebar, .css-1d391kg, .css-1lcbmhc {{
        background:{BG2} !important; color:{TXT} !important;
      }}
      [data-testid="stSidebar"] * {{ color:{TXT} !important; }}
      div[data-baseweb="select"] > div {{ background:{BG2}; color:{TXT}; }}
      .stTextInput input, .stNumberInput input, .stDateInput input {{ background:{BG2}; color:{TXT}; }}

      .block {{ background:{BG2}; border-radius:16px; padding:16px; }}
      .stDataFrame, .stMarkdown, .stText, .stCaption {{ color:{TXT}; }}
      a {{ color:{GREEN}; }}

      /* Plotly 모드바 아이콘 */
      .modebar-group * {{ filter: invert(88%) !important; }}
    </style>
    """, unsafe_allow_html=True)

def style_fig(fig, title=None, line_color=GREEN, line_width=2.2):
    fig.update_layout(
        template="greenopt_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT),
        xaxis=dict(gridcolor="#374151", zerolinecolor="#374151"),
        yaxis=dict(gridcolor="#374151", zerolinecolor="#374151"),
        title=title or fig.layout.title.text if fig.layout.title.text else None,
    )
    for tr in fig.data:
        if hasattr(tr, "line"):
            tr.line.color = getattr(tr.line, "color", None) or line_color
            tr.line.width = getattr(tr.line, "width", None) or line_width
    return fig

init_theme()

# ---------- Paths ----------
try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()
ROOT = APP_DIR.parents[0]
DATA_DIR = ROOT / "data"
ASSET_DIR = APP_DIR / "assets"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"

# ---------- Constants ----------
EMISSION_FACTOR_ELECTRICITY_DEFAULT = 0.475  # kg CO2e/kWh (예시)
EMISSION_FACTOR_GAS = 2.0                    # kg CO2e/m3

# ---------- Data helpers ----------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        # 3년치 시간단위 샘플 생성
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
    return (df.set_index("timestamp").resample(rule).agg(agg).reset_index())

# ---------- Header ----------
c1, c2 = st.columns([0.12, 0.88])
with c1:
    logo = None
    for p in [ASSET_DIR/"greenopt_logo.png", ASSET_DIR/"logo.png", ROOT/"assets"/"greenopt_logo.png", ROOT/"assets"/"logo.png"]:
        if p.exists(): logo = p; break
    if logo:
        st.image(Image.open(logo))
    else:
        st.caption("No logo found (optional).")
with c2:
    st.title("GreenOpt — AI Carbon Intelligence Platform")
    st.caption("Forecast • Optimization • Anomaly • Digital ESG")

st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (3+ years preferred)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        st.success("CSV loaded.")
    else:
        df = load_data(DEFAULT_CSV)
        st.info("Loaded default / generated sample data.")

    required = {"timestamp","electricity_kwh","gas_m3","production_ton"}
    miss = required - set(df.columns)
    if miss:
        st.error(f"Missing required columns: {', '.join(sorted(miss))}")
        st.stop()

    st.header("Scope 2 method")
    scope2_method = st.selectbox("Electricity EF method", ["Location-based","Market-based"], index=0)
    if scope2_method == "Location-based":
        ef_elec_input = st.number_input("EF (kg/kWh)", value=float(EMISSION_FACTOR_ELECTRICITY_DEFAULT), step=0.01)
    else:
        ef_elec_input = st.number_input("EF (market-based kg/kWh)", value=0.0, step=0.01)

    st.header("Filters")
    tmin, tmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    start_date, end_date = st.date_input("Date range", value=(tmin, tmax), min_value=tmin, max_value=tmax)
    sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique()) if "line" in df.columns else [])
    sel_products = st.multiselect("Product", sorted(df["product"].dropna().unique()) if "product" in df.columns else [])
    rule = st.selectbox("Time granularity", ["H","D","W","M"], index=1)

with st.sidebar:
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
mask = (df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
if sel_lines and "line" in df.columns:      mask &= df["line"].isin(sel_lines)
if sel_products and "product" in df.columns: mask &= df["product"].isin(sel_products)
df = df.loc[mask].copy()

# ---------- Carbon columns ----------
def add_carbon_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    out["co2e_kg"] = out["electricity_kwh"]*ef_elec_input + out["gas_m3"]*EMISSION_FACTOR_GAS
    out["pcf_kg_per_ton"] = np.where(out["production_ton"]>0, out["co2e_kg"]/out["production_ton"], np.nan)
    return out

df = add_carbon_columns(df)
df_g = resample_df(df, rule)

# ---------- KPIs ----------
def show_kpis(dfi: pd.DataFrame):
    total = dfi["co2e_kg"].sum()
    avg_pcf = dfi["pcf_kg_per_ton"].mean()
    last = dfi.iloc[-1]["co2e_kg"] if not dfi.empty else np.nan
    a,b,c,d = st.columns(4)
    a.metric("Total CO₂e (kg)", f"{total:,.0f}")
    b.metric("Avg PCF (kg/ton)", f"{avg_pcf:,.2f}" if np.isfinite(avg_pcf) else "N/A")
    c.metric(f"Last {rule} CO₂e (kg)", f"{last:,.1f}" if np.isfinite(last) else "N/A")
    d.metric("Periods", f"{len(dfi):,}")
show_kpis(df_g)

# ---------- Overview ----------
st.subheader("Time-series overview")
if not df_g.empty:
    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"], mode="lines",
                                 name="CO₂e (kg)", line=dict(color=GREEN, width=2.5)))
        st.plotly_chart(style_fig(fig, title="CO₂e (resampled)"), use_container_width=True)
    else:
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots(facecolor=BG2)
        ax.set_facecolor(BG2)
        ax.plot(df_g["timestamp"], df_g["co2e_kg"], color=GREEN)
        st.pyplot(fig, use_container_width=True)
else:
    st.warning("No data in selected range.")

# ---------- STL ----------
st.subheader("Seasonal-Trend Decomposition (STL)")
with st.expander("Show STL"):
    if not _HAS_STATSMODELS:
        st.info("statsmodels 미설치로 STL 생략.")
    else:
        try:
            s = df_g.set_index("timestamp")["co2e_kg"]
            try:
                step = df_g["timestamp"].diff().mode()[0]
                s = s.asfreq(step, method="pad")
            except Exception:
                pass
            stl = sm.tsa.STL(s, robust=True).fit()
            if _HAS_PLOTLY:
                comp = go.Figure()
                comp.add_trace(go.Scatter(x=stl.trend.index, y=stl.trend.values, name="Trend", line=dict(color=GREEN)))
                comp.add_trace(go.Scatter(x=stl.seasonal.index, y=stl.seasonal.values, name="Seasonal"))
                comp.add_trace(go.Scatter(x=stl.resid.index, y=stl.resid.values, name="Residual"))
                st.plotly_chart(style_fig(comp, title="STL Components"), use_container_width=True)
            else:
                st.write("Trend(head):", stl.trend.head())
        except Exception as e:
            st.info(f"STL skipped: {e}")

# ---------- Anomaly ----------
st.subheader("Anomaly Detection")
with st.expander("Detect anomalies"):
    if len(df_g) >= 30:
        X = df_g[["co2e_kg"]].fillna(method="ffill")
        iso = IsolationForest(contamination=0.02, random_state=42)
        lab = iso.fit_predict(X)
        df_g["anomaly"] = (lab == -1)
        if _HAS_PLOTLY:
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"], mode="lines",
                                       name="CO₂e", line=dict(color=GREEN, width=2.2)))
            an = df_g[df_g["anomaly"]]
            fig_a.add_trace(go.Scatter(x=an["timestamp"], y=an["co2e_kg"], mode="markers",
                                       name="Anomaly", marker=dict(size=8, symbol="x", color="#FCA5A5")))
            st.plotly_chart(style_fig(fig_a, title="Anomaly detection"), use_container_width=True)
        else:
            st.write(df_g[df_g["anomaly"]].head())
    else:
        st.info("Need at least 30 periods.")

# ---------- Forecast (경량) ----------
st.subheader("Forecasting")
with st.expander("Train & forecast"):
    horizon = st.slider("Forecast horizon (periods)", 7 if rule=="D" else 24, 60, 14)
    dff = df_g.copy()
    dff["lag1"] = dff["co2e_kg"].shift(1)
    dff = dff.dropna().reset_index(drop=True)
    if len(dff) > horizon + 1:
        tr, te = dff.iloc[:-horizon], dff.iloc[-horizon:]
        y_tr, y_te = tr["co2e_kg"], te["co2e_kg"]
        X_tr, X_te = tr[["lag1"]], te[["lag1"]]
        gbr = GradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
        pred = gbr.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        a,b = st.columns(2)
        a.metric("MAE", f"{mae:,.2f}")
        b.metric("Last pred", f"{pred[-1]:,.2f}")
        if _HAS_PLOTLY:
            f = go.Figure()
            f.add_trace(go.Scatter(x=tr["timestamp"], y=tr["co2e_kg"], name="Train"))
            f.add_trace(go.Scatter(x=te["timestamp"], y=y_te, name="Actual"))
            f.add_trace(go.Scatter(x=te["timestamp"], y=pred, name="Forecast",
                                   line=dict(color=GREEN, width=2.2)))
            st.plotly_chart(style_fig(f, title="Forecast"), use_container_width=True)
        st.session_state["pred_series"] = pd.Series(pred, index=te["timestamp"])
    else:
        st.info("Not enough data to forecast.")

# ---------- Optimization (간단) ----------
st.subheader("Optimization (toy)")
with st.expander("Run optimization"):
    scenario = st.selectbox("Scenario", ["Min Cost (CO₂e cap)","Min Emissions (Production target)"])
    co2e_cap = st.number_input("CO₂e cap (kg)", value=float(df_g["co2e_kg"].quantile(0.75)) if not df_g.empty else 1000.0, step=50.0)
    prod_tgt = st.number_input("Production target (ton)", value=float(df["production_ton"].mean()*24) if not df.empty else 100.0, step=10.0)
    price_e, price_g = 0.15, 0.08
    ef_e, ef_g = ef_elec_input, EMISSION_FACTOR_GAS

    if scenario.startswith("Min Cost"):
        def obj(x): e,g=x; return price_e*e + price_g*g
        cons = [{"type":"ineq", "fun": lambda x: co2e_cap - (ef_e*x[0]+ef_g*x[1])}]
        bnds = [(0,None),(0,None)]
        x0 = [co2e_cap/ef_e*0.5 if ef_e>0 else 0.0, co2e_cap/ef_g*0.5]
    else:
        def obj(x): e,g=x; return ef_e*e + ef_g*g
        alpha, beta = 0.02, 0.05
        cons = [{"type":"ineq", "fun": lambda x: (alpha*x[0]+beta*x[1]) - prod_tgt}]
        bnds = [(0,None),(0,None)]
        x0 = [prod_tgt/alpha*0.5, prod_tgt/beta*0.5]

    res = minimize(obj, x0, bounds=bnds, constraints=cons)
    e_opt, g_opt = float(res.x[0]), float(res.x[1])
    cost_opt = price_e*e_opt + price_g*g_opt
    co2e_opt = ef_e*e_opt + ef_g*g_opt
    st.write({"electricity": round(e_opt,2), "gas": round(g_opt,2),
              "cost": round(cost_opt,2), "co2e": round(co2e_opt,2), "success": bool(res.success)})

# ---------- Carbon Pricing ----------
st.subheader("Carbon Pricing")
with st.expander("Apply price"):
    price_per_t = st.number_input("Carbon price (per tCO₂e)", value=85.0, step=1.0)
    df_cost = pd.DataFrame({
        "timestamp": df_g["timestamp"],
        "cost_local": (df_g["co2e_kg"]/1000.0) * price_per_t
    })
    if _HAS_PLOTLY:
        fc = px.line(df_cost, x="timestamp", y="cost_local", title="Carbon Cost")
        for tr in fc.data:
            tr.line.color = GREEN; tr.line.width = 2.2
        st.plotly_chart(style_fig(fc), use_container_width=True)
    st.dataframe(df_cost.tail(12), use_container_width=True)

# ---------- Partner Hub ----------
st.subheader("Partner Hub — Benchmark • Invite • Trust")
tab_b, tab_i, tab_t = st.tabs(["Benchmark","Invite","Trust"])

with tab_b:
    st.caption("Upload partner benchmark CSV (timestamp, product, line, pcf_kg_per_ton)")
    f = st.file_uploader("Partner CSV", type=["csv"], key="pb")
    if f:
        pdf = pd.read_csv(f)
        if "pcf_kg_per_ton" not in pdf.columns:
            st.error("Missing 'pcf_kg_per_ton'")
        else:
            ours = df_g["pcf_kg_per_ton"].mean()
            peers = pdf["pcf_kg_per_ton"].mean()
            a,b,c = st.columns(3)
            a.metric("Our Avg PCF", f"{ours:,.2f}")
            b.metric("Peer Avg PCF", f"{peers:,.2f}")
            c.metric("Gap (peer-us)", f"{peers-ours:,.2f}")

with tab_i:
    import uuid, json
    code = st.text_input("Invite code", value=str(uuid.uuid4()))
    if not df_g.empty:
        sample = df_g.tail(8)[["timestamp","co2e_kg","pcf_kg_per_ton"]].copy()
        sample["timestamp"] = pd.to_datetime(sample["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        for col in ["co2e_kg","pcf_kg_per_ton"]:
            sample[col] = pd.to_numeric(sample[col], errors="coerce").astype(float)
        pack = {
            "title":"GreenOpt Partner Brief",
            "kpi":{"total_co2e_kg": float(df_g["co2e_kg"].sum()),
                   "avg_pcf": float(df_g["pcf_kg_per_ton"].mean())},
            "sample": sample.to_dict(orient="records"),
            "invite_code": code
        }
        pack_bytes = json.dumps(pack, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        st.download_button("⬇️ Download Partner Pack (JSON)", data=pack_bytes,
                           file_name="greenopt_partner_pack.json", mime="application/json")
    else:
        st.info("Need data to build a partner pack.")

with tab_t:
    import hashlib
    payload = pd.DataFrame({
        "timestamp": df_g["timestamp"].astype(str),
        "co2e_kg": df_g["co2e_kg"].round(6)
    }).to_csv(index=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    st.code(f"SHA256(data_slice) = {digest}")
    st.caption(f"Scope2: {scope2_method} | EF_electricity(kg/kWh): {ef_elec_input} | EF_gas(kg/m³): {EMISSION_FACTOR_GAS}")

# ---------- Export PDF ----------
st.subheader("Export KPI / Report (PDF)")
def build_pdf(df_summary: pd.DataFrame, kpis: dict, note: str = "") -> bytes:
    buf = BytesIO(); c = canvas.Canvas(buf, pagesize=A4); w,h = A4; y = h-50
    c.setFont("Helvetica-Bold", 16); c.drawString(40,y, "GreenOpt — Carbon Intelligence Report"); y-=25
    c.setFont("Helvetica", 10); c.drawString(40,y, f"Scope2: {scope2_method} | EF_elec: {ef_elec_input} kg/kWh"); y-=15
    c.drawString(40,y, f"Period: {str(df_summary['timestamp'].min().date())} ~ {str(df_summary['timestamp'].max().date())}"); y-=25
    c.setFont("Helvetica-Bold", 12); c.drawString(40,y, "KPIs"); y-=18
    c.setFont("Helvetica", 10)
    for k,v in kpis.items():
        c.drawString(50,y, f"- {k}: {v}"); y-=14
    c.showPage(); c.save(); buf.seek(0); return buf.read()

if not df_g.empty:
    kpis = {
        "Total CO₂e (kg)": f"{df_g['co2e_kg'].sum():,.0f}",
        "Avg PCF (kg/ton)": f"{df_g['pcf_kg_per_ton'].mean():,.2f}",
        f"Periods ({rule})": f"{len(df_g):,}",
    }
    pdf = build_pdf(df_g, kpis, "")
    st.download_button("📄 Download KPI Report (PDF)", data=pdf, file_name="greenopt_report.pdf", mime="application/pdf")
else:
    st.info("No data to export.")
