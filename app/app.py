# ✅ 1. 이 줄은 반드시 맨 첫 줄이어야 합니다.
from __future__ import annotations

# ✅ 2. 그 아래부터 설치 가드 블록 추가
import sys, subprocess

def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

for pkg in ["plotly", "scipy", "Pillow"]:
    _ensure(pkg)

# ✅ 3. 이후 일반 import 들
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
from PIL import Image

# ---------- 페이지 설정 ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")

# 여기에 나머지 코드 계속...


# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ----------------- LOGO -----------------
logo_path = Path(__file__).resolve().parent / "assets" / "greenopt_logo.png"
if logo_path.exists():
    st.image(Image.open(logo_path), width=170)
st.markdown(
    "<h1 style='color:#00FFAA;margin-top:-10px;'>GreenOpt — Digital ESG Engine</h1>",
    unsafe_allow_html=True,
)
st.caption("AI-driven ESG forecasting, optimization, hotspot analytics & Catena-X/DPP export")

# ----------------- LOAD / SEED DATA -----------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
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
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])

    # 계산 컬럼 (Scope2 전력, Scope1 연료 기준의 간단 계수)
    if "co2e_kg" not in df.columns:
        df["co2e_kg"] = df["electricity_kwh"] * 0.475 + df["gas_m3"] * 2.0
    if "pcf_kg_per_ton" not in df.columns:
        df["pcf_kg_per_ton"] = df["co2e_kg"] / df["production_ton"].replace(0, np.nan)

    # 서비스 관점 분석을 위해 프로세스(공정) 라벨(가상) 생성
    # 실제 현장 데이터 연동 시 process 컬럼(예: line_id, process_name) 그대로 사용
    processes = np.array(["Furnace", "Molding", "Assembly", "QC"])
    df["process"] = processes[(df.index % len(processes))]
    # 공정별 가중치(가스가 많은 Furnace에서 CO2e 가중)
    df.loc[df["process"] == "Furnace", "co2e_kg"] *= 1.10
    df.loc[df["process"] == "Assembly", "co2e_kg"] *= 0.95
    return df.sort_values("timestamp").reset_index(drop=True)

df = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Settings")
date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)
agg = st.sidebar.selectbox("Aggregation", ["Daily", "Monthly"], index=0)
show_table = st.sidebar.checkbox("Show raw data", False)

# 필터
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

# 집계
plot_df = df.copy()
if agg == "Monthly":
    plot_df = (
        plot_df.set_index("timestamp")
        .resample("M")
        .agg({
            "electricity_kwh": "sum",
            "gas_m3": "sum",
            "production_ton": "sum",
            "co2e_kg": "sum"
        })
        .reset_index()
    )

# ----------------- KPI -----------------
total = plot_df["co2e_kg"].sum()
avg = plot_df["co2e_kg"].mean()
avg_pcf = (plot_df["co2e_kg"].sum() / plot_df["production_ton"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total CO₂e", f"{total:,.0f} kg")
c2.metric("Avg CO₂e", f"{avg:,.2f} kg/{agg.lower()}")
c3.metric("Avg PCF", f"{avg_pcf:,.2f} kg/ton")
c4.metric("Period", f"{plot_df['timestamp'].min().date()} → {plot_df['timestamp'].max().date()}")

# ----------------- TABS -----------------
tabs = st.tabs([
    "📊 Dashboard",
    "🔥 Hotspots",
    "🤖 AI Forecast",
    "🧩 Optimization",
    "🧮 Scenario",
    "📤 DPP / Catena-X API"
])

# ========== 1) DASHBOARD ==========
with tabs[0]:
    st.markdown("### CO₂e Trend")
    fig = px.line(plot_df, x="timestamp", y="co2e_kg", template="plotly_dark",
                  color_discrete_sequence=["#00FFAA"], title=f"CO₂e Over Time ({agg})")
    fig.update_layout(hovermode="x unified", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#00FFAA"), xaxis_title="Date", yaxis_title="CO₂e (kg)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Energy Breakdown")
    fig2 = px.line(plot_df, x="timestamp", y=["electricity_kwh", "gas_m3"], template="plotly_dark",
                   color_discrete_sequence=["#00CC88", "#00FFAA"], title=f"Electricity & Gas ({agg})")
    fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"))
    st.plotly_chart(fig2, use_container_width=True)

    if show_table:
        st.dataframe(plot_df.tail(50).round(2), use_container_width=True, height=420)

# ========== 2) HOTSPOTS (공정·시간대) ==========
with tabs[1]:
    st.markdown("### Hotspot — Process Pareto (Top Emitting Processes)")
    proc = df.groupby("process", as_index=False)["co2e_kg"].sum().sort_values("co2e_kg", ascending=False)
    figp = px.bar(proc, x="process", y="co2e_kg", template="plotly_dark",
                  color_discrete_sequence=["#00FFAA"], text_auto=".2s",
                  title="Process CO₂e (Pareto)")
    figp.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"))
    st.plotly_chart(figp, use_container_width=True)

    st.markdown("### Hotspot — Calendar Heatmap (Day of Week × Month)")
    df_cal = df.copy()
    df_cal["dow"] = df_cal["timestamp"].dt.day_name()
    df_cal["month"] = df_cal["timestamp"].dt.to_period("M").astype(str)
    pivot = df_cal.pivot_table(index="dow", columns="month", values="co2e_kg", aggfunc="mean")
    pivot = pivot.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    figh = px.imshow(pivot, color_continuous_scale=["#0e1117","#093","#0f9","#0fa","#0fb"],
                     template="plotly_dark", aspect="auto")
    figh.update_layout(title="Average CO₂e by Day-of-Week × Month",
                       plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font=dict(color="#00FFAA"))
    st.plotly_chart(figh, use_container_width=True)

# ========== 3) AI FORECAST (간단 MA) ==========
with tabs[2]:
    st.markdown("### 🔮 Moving-Average Forecast (operations-friendly)")
    horizon = st.slider("Forecast horizon (days)", 7, 90, 21)
    hist = plot_df.set_index("timestamp")["co2e_kg"]
    ma = hist.rolling(window=7, min_periods=1).mean()
    idx = pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=horizon)
    forecast = pd.Series(np.full(len(idx), ma.iloc[-1]), index=idx)
    fig3 = px.line(template="plotly_dark")
    fig3.add_scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#00FFAA"))
    fig3.add_scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(color="#00BFFF", dash="dot"))
    fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"),
                       title=f"{horizon}-Day Forecast")
    st.plotly_chart(fig3, use_container_width=True)

# ========== 4) OPTIMIZATION (라그랑주) ==========
with tabs[3]:
    st.markdown("### ⚙️ Lagrange Optimization — 생산량 제약 하 CO₂e 최소화")
    prod_target = st.number_input("Production target (tons)", 300.0, 600.0, 400.0)
    def f(x): return 0.475*x[0] + 2.0*x[1]  # 전력·가스 배출 계수
    def constraint(x): return prod_target - (0.002*x[0] + 0.005*x[1])  # 단순 생산함수
    res = minimize(f, [10000, 2000], constraints={'type':'eq','fun':constraint})
    opt_e, opt_g = res.x
    st.success(f"Optimal: Electricity {opt_e:,.0f} kWh / Gas {opt_g:,.0f} m³")
    st.metric("Expected Emission", f"{f(res.x):,.2f} kg CO₂e")

# ========== 5) SCENARIO ==========
with tabs[4]:
    st.markdown("### 🧮 Scenario Simulator — 효율 변화 반영")
    eff = st.slider("Electric efficiency improvement (%)", -20, 20, 0)
    gas_eff = st.slider("Gas reduction (%)", -20, 20, 0)
    sim = plot_df.copy()
    sim["electricity_kwh"] *= (1 - eff/100)
    sim["gas_m3"] *= (1 - gas_eff/100)
    sim["co2e_kg"] = sim["electricity_kwh"]*0.475 + sim["gas_m3"]*2.0
    change = (sim["co2e_kg"].sum() - plot_df["co2e_kg"].sum()) / plot_df["co2e_kg"].sum() * 100
    st.metric("Total Emission Change", f"{change:+.2f}%")
    fig4 = px.line(sim, x="timestamp", y="co2e_kg", template="plotly_dark",
                   color_discrete_sequence=["#00FFAA"], title="Scenario-adjusted CO₂e")
    fig4.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#00FFAA"))
    st.plotly_chart(fig4, use_container_width=True)

# ========== 6) DPP / Catena-X API (모의) ==========
with tabs[5]:
    st.markdown("### 📤 Digital Product Passport / Catena-X JSON & REST (Mock)")
    payload = {
        "product_id": "GREENOPT-SAMPLE-001",
        "period": {
            "start": str(plot_df["timestamp"].min().date()),
            "end": str(plot_df["timestamp"].max().date())
        },
        "scope1_kg": float(plot_df["gas_m3"].sum() * 2.0),
        "scope2_kg": float(plot_df["electricity_kwh"].sum() * 0.475),
        "total_kg": float(plot_df["co2e_kg"].sum()),
        "unit": "kg CO2e",
        "format": "Catena-X JSON v1.0",
    }
    js = json.dumps(payload, indent=2)
    st.code(js, language="json")
    st.download_button("💾 Download DPP JSON", js, file_name="greenopt_dpp.json")

    st.markdown("#### REST API 예시 (모의 엔드포인트)")
    st.code(
        "POST /api/v1/dpp\n"
        "Content-Type: application/json\n\n" + js,
        language="bash"
    )

# ----------------- FOOTER -----------------
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.8rem;margin-top:1rem;'>© 2025 GreenOpt — ESG·AI Digital Engine</p>",
    unsafe_allow_html=True,
)
