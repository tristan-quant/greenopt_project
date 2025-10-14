# app/app.py
from __future__ import annotations
from pathlib import Path
import base64
import io
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# 설정: 페이지
st.set_page_config(page_title="GreenOpt — CO₂e Intelligence Dashboard", layout="wide")

# 프로젝트 루트
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 로고(Base64) — 파일 없이 동작
# 필요하면 나중에 실제 파일로 교체 가능
LOGO_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAYAAACe+Y9XAAAABHNCSVQICAgIfAhkiAAAAbZJREFUeJztm8FtwkAQhT+V
rM9oBNbW2pHkqQxaKQJQwKcn5m8WmJw2S1yA9O3qP8Fq0r4W4k7w8w0b3iE1XlV3o3Wb7wqQpU0l1Q5wFJH8pGq2sK6c8Z
Yx0NwVqfZQv3V8rWwVv7x7g7Gg1nQbq0Qb6mE5Ewq2tX8l0oK6u3oQJxk2hJ0k7bV0lq2p8mC3H1Rk0x3Y1w+2M0i3HkB8
q6cVQWwU2vQw3rQivc5I0QJYw5kJfWc8cGQJzQkJwFQ5cQzFJv0o3wZyY0n0w1cEJrXVwZb9b2yHq3Jq2Cw3m0Cz3c5mS
vQ7kqgZ9KzS9WkK7c9H2wGJv3m4J1z0wHfQxvK5m3oCwqK4G0sKqkJ8GQhB2rQ5j8iGqf6q2Yw1m9T3bqNwq8ZbU4v8w8u
J3t9Qq3P8t3d8Q3q4n0B2h7wYlq6c8cTjJmGkq8P8p8gYxgcmcQAAAB8Y0lEQVR4nO2bW2wUVRzHP6r9w1C4lR6kQe0iI
yZQq1wYQk0cFZ9J0VQmJ+o1w7H3yq8nQ2+JfQzP2yqgWlG7S4K5mQ2SYQ2mYyVqg2mQ3k3qQm0i2v8b7lS9b3t2s2s9m+f
Zzqf3s1gYVb+qF3c+7e+77z3ru9gJZrVfZpX4k2q1Z9W8i5m2sYx2xwKJco2l3hQw1+gUQW1Gd8G3gXlq6r2TQ8Jr7rH1b
6BbgT2oQ8k6oU2k2M3k8a4C2gQk9qk9q6yqRk4rQ8A2oS8k4yG3wS8tJ5qQ2o6g/1y5x5uG8JYB2V3l8m2KX4xgqY1s8b9
p5JgqzYbq7T0zYfHfGmP3Ykqkq6mT0Cw2wq4uH9wqk3QGJzH4H0b2g8pZ9BqkC1mG8rQv3zQ7o3y2Qm1q4wqkH2m8gZqf
kL2U3E0y0vQn2m8oZqfUL2U3E0y0vQn2m8oZqfUL2U3E0y0vQn2m8oY8Ew9Gm8r1Qkq6oYjL0mL3b6y1iWlYp6m0Uo4bJ
…(생략 가능: 실제 배포엔 이대로 두세요)…
"""

# ─────────────────────────────────────────────────────────
# 유틸: 로고 렌더
def render_header():
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <img src="data:image/png;base64,{LOGO_BASE64.strip()}" width="54">
            <div>
                <h1 style="color:#00FFAA;margin:0;">GreenOpt — CO₂e Intelligence Dashboard</h1>
                <p style="color:#999;margin:0;">AI-driven Carbon Footprint & Forecast System</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# 데이터 로드/생성
@st.cache_data(show_spinner=False)
def load_or_seed() -> pd.DataFrame:
    target = DATA_DIR / "factory_data_3y.csv"
    if not target.exists():
        # 3년치 시뮬레이션 데이터 시드 (시간당)
        ts = pd.date_range("2022-10-15 00:00:00", "2025-10-15 00:00:00", freq="H", inclusive="left")
        rng = np.random.default_rng(42)

        hours = ts.hour
        months = ts.month
        elec = (
            450
            + 50*np.sin((hours-8)/24*2*np.pi)
            + 30*np.cos((hours)/24*2*np.pi)
            + 60*np.where((months>=6)&(months<=8), 1.2, 1.0)
            + rng.normal(0, 15, len(ts))
        )
        gas = (
            55 + 8*np.cos((hours-6)/24*2*np.pi)
            + 25*np.where((months<=3)|(months>=11), 1.2, 0.8)
            + rng.normal(0, 3.5, len(ts))
        )
        weekday = pd.Series(ts).dt.weekday.values
        prod = (
            (20 + 3*np.sin((hours-9)/24*2*np.pi))
            * np.where(weekday>=5, 0.65, 1.0)
            + rng.normal(0, 1.2, len(ts))
        )
        df = pd.DataFrame({
            "timestamp": ts,
            "electricity_kwh": np.clip(elec, 280, None).round(2),
            "gas_m3": np.clip(gas, 10, None).round(2),
            "production_ton": np.clip(prod, 0, None).round(2),
        })
        # 간단한 배출량 계산(엔진 내장)
        df["co2e_kg"] = df["electricity_kwh"]*0.475 + df["gas_m3"]*2.0
        df["pcf_kg_per_ton"] = df["co2e_kg"] / df["production_ton"].replace(0, np.nan)

        target.write_text(df.to_csv(index=False), encoding="utf-8")
        return df
    else:
        df = pd.read_csv(target, parse_dates=["timestamp"])
        # 안전장치: 필요한 파생컬럼 없으면 생성
        if "co2e_kg" not in df.columns:
            df["co2e_kg"] = df["electricity_kwh"]*0.475 + df["gas_m3"]*2.0
        if "pcf_kg_per_ton" not in df.columns:
            df["pcf_kg_per_ton"] = df["co2e_kg"] / df["production_ton"].replace(0, np.nan)
        return df

# ─────────────────────────────────────────────────────────
# 간단 예측(외부 라이브러리 없이): 이동평균 + 계절성 보정
def naive_forecast(series: pd.Series, periods: int, freq: str, seasonal_periods: int | None = None) -> pd.Series:
    """단순 이동평균/계절성 평균 기반 예측. statsmodels/prophet 없이 구동."""
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)

    # 계절성 평균 사용(있으면)
    if seasonal_periods and len(series) >= seasonal_periods:
        base = series.tail(seasonal_periods).mean()
    else:
        base = series.tail(min(168, len(series))).mean()  # 최대 1주(시간 기준) 이동평균

    last_ts = series.index[-1]
    if freq == "H":
        idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=periods, freq="H")
    else:
        idx = pd.date_range(last_ts + pd.Timedelta(days=1), periods=periods, freq="D")

    return pd.Series(np.full(len(idx), base), index=idx)

# ─────────────────────────────────────────────────────────
# 메인
def main():
    render_header()

    df = load_or_seed().sort_values("timestamp").copy()

    # ─ Sidebar
    st.sidebar.header("Settings")
    agg = st.sidebar.selectbox("Aggregation", ["Hourly", "Daily"], index=1)
    horizon_days = st.sidebar.slider("Forecast horizon (days)", 3, 30, 7)
    show_table = st.sidebar.checkbox("Show raw data", value=False)

    date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
    date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

    # 필터
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

    # 집계
    plot_df = df.copy()
    if agg == "Daily":
        plot_df = (
            plot_df.set_index("timestamp")
            .resample("D")
            .agg({"co2e_kg": "sum", "electricity_kwh": "sum", "gas_m3": "sum", "production_ton": "sum"})
            .reset_index()
        )

    # ─ KPI
    total_co2e = plot_df["co2e_kg"].sum()
    avg_co2e = plot_df["co2e_kg"].mean()
    total_prod = plot_df["production_ton"].sum()
    avg_pcf = (plot_df["co2e_kg"].sum() / total_prod) if total_prod and not np.isnan(total_prod) else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total CO₂e", f"{total_co2e:,.0f} kg")
    k2.metric("Avg CO₂e", f"{avg_co2e:,.2f} kg/{agg.lower()}")
    k3.metric("Avg PCF", "N/A" if np.isnan(avg_pcf) else f"{avg_pcf:,.2f} kg/ton")
    k4.metric("Period", f"{plot_df['timestamp'].min().date()} → {plot_df['timestamp'].max().date()}")

    # ─ 그래프 1: CO₂e 추이
    st.subheader(f"Time series — CO₂e (kg) [{agg}]")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(plot_df["timestamp"], plot_df["co2e_kg"])
    ax1.set_xlabel("timestamp")
    ax1.set_ylabel("kg CO₂e")
    ax1.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig1)

    # ─ 그래프 2: 전력/가스 추이
    st.subheader("Electricity vs Gas")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(plot_df["timestamp"], plot_df["electricity_kwh"], label="Electricity (kWh)")
    ax2.plot(plot_df["timestamp"], plot_df["gas_m3"], label="Gas (m³)")
    ax2.set_xlabel("timestamp")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

    # ─ 예측: 이동평균 기반
    st.subheader("Forecast — CO₂e (naive)")
    freq = "H" if agg == "Hourly" else "D"
    periods = horizon_days * (24 if freq == "H" else 1)
    hist = plot_df.set_index("timestamp")["co2e_kg"]
    seasonal_periods = 24 if freq == "H" else 7  # 일/주기준
    fcst = naive_forecast(hist, periods=periods, freq=freq, seasonal_periods=seasonal_periods)

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(hist.index, hist.values, label="History")
    ax3.plot(fcst.index, fcst.values, label=f"Forecast (+{horizon_days}d)")
    ax3.set_xlabel("timestamp")
    ax3.set_ylabel("kg CO₂e")
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.legend()
    st.pyplot(fig3)

    # ─ 데이터 테이블 & 다운로드
    st.subheader("Sample rows")
    st.dataframe(plot_df.head(50), use_container_width=True)

    st.download_button(
        "📥 Download filtered CSV",
        data=plot_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="greenopt_filtered.csv",
        mime="text/csv",
    )

    st.markdown(
        "<p style='text-align:center;color:#666;font-size:0.8rem;margin-top:1rem;'>© 2025 GreenOpt — ESG·AI Data Intelligence Platform</p>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

