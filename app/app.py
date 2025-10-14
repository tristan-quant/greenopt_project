# =====================================================
# GreenOpt — Digital ESG Engine (Full Feature, Safe Auto-Install)
# =====================================================
from __future__ import annotations

# ---------- 0) 누락 패키지 자동 설치 가드 ----------
import sys, subprocess

def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦 Installing: {pkg} ...")
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
from typing import Optional

# Plotly가 없어도 동작하도록 폴백 처리
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

# ---------- 2) 페이지/경로 설정 ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"
LOGO_CANDIDATES = [
    ASSET_DIR / "logo_512.png",
    ASSET_DIR / "logo.png",
    ASSET_DIR / "brand" / "logo_512.png",
]

# ---------- 3) 외부 모듈 시도 → 실패 시 로컬 구현 ----------
def _import_with_fallback():
    carbon_engine = None
    data_utils = None
    try:
        from src.carbon_engine import add_carbon_columns as _add
        carbon_engine = _add
    except Exception:
        carbon_engine = None
    try:
        from src.data_utils import load_factory_data as _load
        data_utils = _load
    except Exception:
        data_utils = None
    return carbon_engine, data_utils

_add_carbon_columns_ext, _load_factory_data_ext = _import_with_fallback()

# 로컬 기본 상수/함수 (외부 모듈 대체)
EMISSION_FACTOR_ELECTRICITY = 0.475  # kg CO2e/kWh
EMISSION_FACTOR_GAS        = 2.0     # kg CO2e/m3

def add_carbon_columns_local(df: pd.DataFrame) -> pd.DataFrame:
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

def add_carbon_columns(df: pd.DataFrame) -> pd.DataFrame:
    if _add_carbon_columns_ext is not None:
        try:
            return _add_carbon_columns_ext(df)
        except Exception:
            pass
    return add_carbon_columns_local(df)

@st.cache_data(show_spinner=False)
def load_factory_data(path: Path) -> pd.DataFrame:
    if _load_factory_data_ext is not None:
        try:
            return _load_factory_data_ext(path)
        except Exception:
            pass
    # 로컬 로더
    if path.exists():
        df = pd.read_csv(path)
    else:
        # 샘플 데이터 생성 (앱이 꺼지지 않도록)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=24*14, freq="H"),  # 2주
            "electricity_kwh": np.random.uniform(80, 220, 24*14),
            "gas_m3": np.random.uniform(8, 35, 24*14),
            "production_ton": np.random.uniform(3, 16, 24*14),
            "line": np.random.choice(["A-Line", "B-Line"], 24*14),
            "product": np.random.choice(["Widget-X", "Widget-Y"], 24*14),
        })
    # timestamp 형 변환 보정
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ---------- 4) 헤더/로고 ----------
# ---------- 로고 자동탐색 + 디버그 ----------
import os
from pathlib import Path
from PIL import Image

def find_logo_paths(root: Path) -> list[Path]:
    candidates = []
    # 1) 명시적 후보 위치
    explicit = [
        root / "assets" / "logo_512.png",
        root / "assets" / "logo.png",
        root / "assets" / "brand" / "logo_512.png",
        root / "assets" / "brand" / "logo.png",
        # 혹시 app/assets 에 넣었을 가능성까지 탐색
        root / "app" / "assets" / "logo_512.png",
        root / "app" / "assets" / "logo.png",
    ]
    for p in explicit:
        if p.exists():
            candidates.append(p)

    # 2) repo 전체에서 logo*.(png|jpg) 스캔 (너무 넓으면 필요 시 주석)
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in root.rglob(f"logo*{ext[1:]}"):  # e.g., logo*.png
            if p.is_file() and p not in candidates:
                candidates.append(p)

    return candidates

def show_logo_debug(root: Path):
    assets_dir = root / "assets"
    st.caption(f"🔎 Logo search base: {root}")
    st.caption(f"🔎 assets dir exists: {assets_dir.exists()} ({assets_dir})")
    if assets_dir.exists():
        try:
            st.caption("📂 assets contents (top-level): " + ", ".join(sorted(os.listdir(assets_dir))[:20]))
        except Exception as e:
            st.caption(f"⚠️ assets list error: {e}")

logo_candidates = find_logo_paths(ROOT)
if logo_candidates:
    logo_path = logo_candidates[0]
    try:
        st.image(Image.open(logo_path), caption="", use_container_width=True)
        st.caption(f"✅ Loaded logo: {logo_path.relative_to(ROOT)}")
    except Exception as e:
        st.warning(f"⚠️ Failed to load logo: {logo_path} ({e})")
        show_logo_debug(ROOT)
else:
    st.info("ℹ️ No logo found. Looking in /assets by default.")
    show_logo_debug(ROOT)


# ---------- 5) 데이터 로딩 및 업로드 ----------
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload CSV (timestamp, electricity_kwh, gas_m3, production_ton, ...)", type=["csv"])
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
            st.warning("Default CSV not found. Using generated sample data.")

# ---------- 6) 파생 컬럼 (CO₂e, PCF) ----------
df = add_carbon_columns(df)

# ---------- 7) 사이드바 필터 ----------
with st.sidebar:
    st.header("Filters")
    # 라인/제품 필터(있을 경우만)
    if "line" in df.columns:
        sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique().tolist()), default=None)
    else:
        sel_lines = []
    if "product" in df.columns:
        sel_products = st.multiselect("Product", sorted(df["product"].dropna().unique().tolist()), default=None)
    else:
        sel_products = []

    # 날짜 범위
    date_min = pd.to_datetime(df["timestamp"].min()).date() if "timestamp" in df.columns else None
    date_max = pd.to_datetime(df["timestamp"].max()).date() if "timestamp" in df.columns else None
    if date_min and date_max:
        start, end = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    else:
        start, end = None, None

# 필터 적용
mask = pd.Series(True, index=df.index)
if sel_lines:
    if "line" in df.columns:
        mask &= df["line"].isin(sel_lines)
if sel_products:
    if "product" in df.columns:
        mask &= df["product"].isin(sel_products)
if start and end:
    mask &= (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

df_f = df.loc[mask].sort_values("timestamp")

# ---------- 8) KPI 카드 ----------
def _kpi_block(df_in: pd.DataFrame):
    total_co2e = float(df_in["co2e_kg"].sum()) if not df_in.empty else 0.0
    avg_pcf = float(df_in["pcf_kg_per_ton"].mean()) if not df_in.empty else np.nan
    latest = df_in.iloc[-1] if not df_in.empty else None
    latest_co2e = float(latest["co2e_kg"]) if latest is not None else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Total CO₂e (kg)", f"{total_co2e:,.0f}")
    c2.metric("Avg PCF (kg/ton)", f"{avg_pcf:,.2f}" if np.isfinite(avg_pcf) else "N/A")
    c3.metric("Last Hour CO₂e (kg)", f"{latest_co2e:,.1f}" if np.isfinite(latest_co2e) else "N/A")

_kpi_block(df_f)

# ---------- 9) 시각화 ----------
st.subheader("Emissions & PCF")
if not df_f.empty:
    if _HAS_PLOTLY:
        fig1 = px.line(df_f, x="timestamp", y="co2e_kg", title="Hourly CO₂e Emissions (kg)", markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.line(df_f, x="timestamp", y="pcf_kg_per_ton", title="PCF (kg per ton)", markers=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Plotly not available. Using Streamlit native charts.")
        st.line_chart(df_f.set_index("timestamp")["co2e_kg"], use_container_width=True)
        st.line_chart(df_f.set_index("timestamp")["pcf_kg_per_ton"], use_container_width=True)
else:
    st.warning("No data after filters.")

# ---------- 10) 원본/가공 데이터 보기 ----------
with st.expander("Show filtered data table"):
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True)

# ---------- 11) CSV 다운로드 ----------
csv_bytes = df_f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data (CSV)", data=csv_bytes, file_name="greenopt_filtered.csv", mime="text/csv")

# ---------- 12) 간단 최적화 예시 ----------
st.subheader("Energy Cost Optimization (Toy Example)")
if _HAS_SCIPY:
    st.caption("Minimize cost with electricity+gas = target (default 200). This is a toy demo.")
    target = st.slider("Target total energy (arbitrary units)", min_value=50, max_value=500, value=200, step=10)

    def objective(x):
        elec, gas = x
        return 0.15 * elec + 0.08 * gas

    cons = [{"type": "eq", "fun": lambda x, t=target: x[0] + x[1] - t}]
    bounds = [(0, None), (0, None)]
    x0 = [target/2, target/2]
    res = minimize(objective, x0, bounds=bounds, constraints=cons)
    st.write({"electricity": float(res.x[0]), "gas": float(res.x[1]), "cost": float(res.fun)})
else:
    st.info("SciPy not available. Skipping optimization demo.")

# ---------- 13) 부가 옵션 ----------
with st.sidebar:
    st.header("Options")
    show_raw = st.checkbox("Show raw columns", value=False)
    if show_raw:
        st.write("Raw dataframe head:")
        st.dataframe(df.head(), use_container_width=True)

    st.caption("Emission factors (editable here are placeholders):")
    ef_elec = st.number_input("EF Electricity (kg/kWh)", value=float(EMISSION_FACTOR_ELECTRICITY), step=0.01)
    ef_gas  = st.number_input("EF Gas (kg/m3)", value=float(EMISSION_FACTOR_GAS), step=0.01)
    if ef_elec != EMISSION_FACTOR_ELECTRICITY or ef_gas != EMISSION_FACTOR_GAS:
        # 재계산 (로컬만 반영)
        df_tmp = df.copy()
        df_tmp["co2e_kg"] = df_tmp["electricity_kwh"] * ef_elec + df_tmp["gas_m3"] * ef_gas
        df_tmp["pcf_kg_per_ton"] = df_tmp["co2e_kg"] / df_tmp["production_ton"]
        st.success("Emission factors updated locally. Use main filters above to view effects.")
