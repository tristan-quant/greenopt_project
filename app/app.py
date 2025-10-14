# =====================================================
# GreenOpt — Digital ESG Engine
# Safe Auto-Install Version (2025-10)
# =====================================================

# ⚠️ 반드시 맨 위에 있어야 합니다.
from __future__ import annotations

# ---------- 1. 필수 패키지 자동 설치 ----------
import sys, subprocess

def _ensure(pkg: str):
    """지정된 패키지가 없을 경우 자동 설치"""
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦 Installing missing package: {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

# 필요한 주요 패키지들
for pkg in ["streamlit", "pandas", "numpy", "plotly", "scipy", "Pillow"]:
    _ensure(pkg)

# ---------- 2. 일반 import ----------
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize
from PIL import Image
from pathlib import Path

# ---------- 3. 페이지 설정 ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")
ROOT = Path(__file__).resolve().parents[1]

# ---------- 4. 타이틀 ----------
st.title("GreenOpt — AI-driven Carbon Footprint Calculator")
st.caption("Demo dashboard for hourly CO₂e and product carbon footprint (PCF)")

# ---------- 5. 데이터 로딩 예시 ----------
DATA_PATH = ROOT / "data" / "factory_data.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"Data loaded successfully: {DATA_PATH.name}")
except FileNotFoundError:
    st.warning(f"⚠️ 데이터 파일이 없습니다: {DATA_PATH}")
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=24, freq="H"),
        "electricity_kwh": np.random.uniform(100, 200, 24),
        "gas_m3": np.random.uniform(10, 30, 24),
        "production_ton": np.random.uniform(5, 15, 24),
    })

# ---------- 6. 탄소 배출량 계산 ----------
EMISSION_FACTOR_ELECTRICITY = 0.475  # kg CO2e/kWh
EMISSION_FACTOR_GAS = 2.0            # kg CO2e/m3

df["co2e_kg"] = (
    df["electricity_kwh"] * EMISSION_FACTOR_ELECTRICITY
    + df["gas_m3"] * EMISSION_FACTOR_GAS
)
df["pcf_kg_per_ton"] = df["co2e_kg"] / df["production_ton"]

# ---------- 7. 시각화 ----------
fig = px.line(
    df,
    x="timestamp",
    y="co2e_kg",
    title="Hourly CO₂e Emissions (kg)",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# ---------- 8. 최적화 예시 ----------
st.subheader("Optimization Example")

def objective(x):
    # 단순 비용함수 예시: (전기*비용 + 가스*비용)
    electricity, gas = x
    return 0.15 * electricity + 0.08 * gas

constraints = [{"type": "eq", "fun": lambda x: x[0] + x[1] - 200}]
x0 = [100, 100]
res = minimize(objective, x0, constraints=constraints)

st.write("Optimal energy mix:", res.x)
st.write("Total cost:", res.fun)
