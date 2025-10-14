from __future__ import annotations
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.carbon_engine import add_carbon_columns
from src.data_utils import load_factory_data

st.set_page_config(page_title="GreenOpt — CO₂e Dashboard", layout="wide")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "factory_data.csv"

st.title("GreenOpt — AI-driven Carbon Footprint Calculator")
st.caption("Demo dashboard for hourly CO₂e and product carbon footprint (PCF)")

# ---------- Helpers ----------
REQUIRED = {"timestamp", "electricity_kwh", "gas_m3"}

@st.cache_data(show_spinner=False)
def load_and_prepare(path: Path) -> pd.DataFrame:
    df = load_factory_data(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    df = add_carbon_columns(df)
    # 핵심 컬럼 결측 제거(필요시 Imputer로 대체 가능)
    df = df.dropna(subset=["timestamp", "electricity_kwh", "gas_m3", "co2e_kg"]).copy()
    # 정렬 보장
    df = df.sort_values("timestamp")
    return df

def guard_empty(df: pd.DataFrame, msg: str) -> bool:
    if df.empty:
        st.info(msg)
        return True
    return False

# ---------- Load ----------
try:
    df = load_and_prepare(DATA_PATH)
except Exception as e:
    st.error(f"데이터 로드/전처리 중 오류: {e}")
    st.stop()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

date_min = df["timestamp"].min().date()
date_max = df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range", [date_min, date_max], min_value=date_min, max_value=date_max
)

# 집계 단위 선택
agg_level = st.sidebar.selectbox("Aggregation", ["Hourly", "Daily"], index=0)

# 필터 적용
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    # 끝일 포함을 위해 +1day 후 미만(<) 필터
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
else:
    st.warning("날짜 범위를 올바르게 선택해 주세요.")

if guard_empty(df, "선택된 조건에 맞는 데이터가 없습니다. 날짜 범위를 조정해 보세요."):
    st.stop()

# ---------- Aggregation ----------
plot_df = df.copy()
if agg_level == "Daily":
    plot_df = (
        plot_df.set_index("timestamp")
        .resample("D")
        .agg({"co2e_kg": "sum", "electricity_kwh": "sum", "gas_m3": "sum", "production_ton": "sum"})
        .reset_index()
    )

# ---------- KPIs ----------
col1, col2, col3 = st.columns(3)
col1.metric("Total CO₂e (kg)", f"{plot_df['co2e_kg'].sum():,.0f}")
col2.metric("Avg CO₂e per hour/day (kg)", f"{plot_df['co2e_kg'].mean():,.2f}")
if "production_ton" in df.columns and df["production_ton"].notna().any():
    # 집계 레벨에 맞춰 pcf 평균(단순평균) 표기
    pcf_series = df["pcf_kg_per_ton"].dropna()
    col3.metric("Avg PCF (kg/ton)", f"{pcf_series.mean():,.2f}")
else:
    col3.metric("Avg PCF (kg/ton)", "N/A")

# ---------- Plot ----------
st.subheader(f"Time series — CO₂e (kg) [{agg_level}]")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(plot_df["timestamp"], plot_df["co2e_kg"])
ax.set_xlabel("timestamp")
ax.set_ylabel("kg CO₂e")
ax.set_title(f"{agg_level} CO₂e")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# ---------- Table & Download ----------
st.subheader("Sample rows")
st.dataframe(df.head(20), use_container_width=True)

st.download_button(
    "Download filtered CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="greenopt_filtered.csv",
    mime="text/csv",
)
