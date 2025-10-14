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

df = load_factory_data(DATA_PATH)
df = add_carbon_columns(df)

# Sidebar filters
st.sidebar.header("Filters")
date_min = df["timestamp"].min().date()
date_max = df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

st.subheader("Time series — CO₂e (kg)")
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["timestamp"], df["co2e_kg"])
ax.set_xlabel("timestamp")
ax.set_ylabel("kg CO₂e")
ax.set_title("Hourly CO₂e")
st.pyplot(fig)

st.subheader("Sample rows")
st.dataframe(df.head(20))