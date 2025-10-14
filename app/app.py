# =====================================================
# GreenOpt — Digital ESG Engine (Advanced AI & Optimization)
# =====================================================
from __future__ import annotations

# ---------- 0) Auto-install guard ----------
import sys, subprocess

def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦 Installing: {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

# Core + ML/TS libs
for pkg in ["streamlit", "pandas", "numpy", "plotly", "scipy", "Pillow", "scikit-learn", "statsmodels"]:
    _ensure(pkg)

# ---------- 1) Imports ----------
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# plotting
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# optimization
from scipy.optimize import minimize

# ML
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

# ---------- 2) Paths & page ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[0]
DATA_DIR = ROOT / "data"
ASSET_DIR = APP_DIR / "assets"
DEFAULT_CSV = DATA_DIR / "factory_data.csv"      # 3년치가 들어 있다고 가정

# ---------- 3) Emission factors & helpers ----------
EMISSION_FACTOR_ELECTRICITY = 0.475  # kg CO2e/kWh
EMISSION_FACTOR_GAS = 2.0           # kg CO2e/m3

def add_carbon_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["co2e_kg"] = df["electricity_kwh"] * EMISSION_FACTOR_ELECTRICITY + df["gas_m3"] * EMISSION_FACTOR_GAS
    df["pcf_kg_per_ton"] = np.where(df["production_ton"]>0, df["co2e_kg"]/df["production_ton"], np.nan)
    return df

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        # fallback 샘플: 3년치(시간단위)
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # 시간해상도 변경: H/D/W/M
    agg = {
        "electricity_kwh":"sum",
        "gas_m3":"sum",
        "production_ton":"sum",
        "co2e_kg":"sum",
        "pcf_kg_per_ton":"mean"
    }
    return (df
            .set_index("timestamp")
            .resample(rule)
            .agg(agg)
            .reset_index())

# ---------- 4) Header (logo left-top) ----------
header_cols = st.columns([0.12, 0.88])
with header_cols[0]:
    logo_candidates = [
        ASSET_DIR / "greenopt_logo.png",
        ASSET_DIR / "logo.png",
        ROOT / "assets" / "greenopt_logo.png",
        ROOT / "assets" / "logo.png",
    ]
    logo = next((p for p in logo_candidates if p.exists()), None)
    if logo:
        st.image(Image.open(logo))
with header_cols[1]:
    st.title("GreenOpt — AI Carbon Intelligence Platform")
    st.caption("Forecasting • Optimization • Anomaly detection • 3-year time-series analytics")

st.divider()

# ---------- 5) Sidebar: data & controls ----------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (3+ years preferred)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.success("Uploaded CSV loaded.")
    else:
        df = load_data(DEFAULT_CSV)
        st.info(f"Loaded: {DEFAULT_CSV.name}" if DEFAULT_CSV.exists() else "Using generated 3-year sample")

    st.header("Filters")
    # 전체 3년 범위를 기본값으로
    tmin, tmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    start_date, end_date = st.date_input("Date range", value=(tmin, tmax), min_value=tmin, max_value=tmax)
    sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique().tolist()) if "line" in df.columns else [])
    sel_products = st.multiselect("Product", sorted(df["product"].dropna().unique().tolist()) if "product" in df.columns else [])
    rule = st.selectbox("Time granularity", ["H","D","W","M"], index=1)  # 기본 일간

# apply filters
mask = (df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)-pd.Timedelta(seconds=1))
if sel_lines and "line" in df.columns: mask &= df["line"].isin(sel_lines)
if sel_products and "product" in df.columns: mask &= df["product"].isin(sel_products)

df = add_carbon_columns(df.loc[mask].copy())
df_g = resample_df(df, rule)

# ---------- 6) KPI ----------
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

# ---------- 7) Charts ----------
st.subheader("Time-series overview")
if not df_g.empty:
    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"], mode="lines+markers", name="CO₂e (kg)"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_g.set_index("timestamp")["co2e_kg"])
else:
    st.warning("No data in selected range")

# ---------- 8) Decomposition (seasonality/trend) ----------
st.subheader("Seasonal-Trend Decomposition (STL)")
with st.expander("Show STL decomposition"):
    try:
        # STL은 균등간격 필요: 결측 보간
        s = df_g.set_index("timestamp")["co2e_kg"].asfreq(df_g["timestamp"].diff().mode()[0], method="pad")
    except Exception:
        s = df_g.set_index("timestamp")["co2e_kg"]
    try:
        stl = sm.tsa.STL(s, robust=True).fit()
        if _HAS_PLOTLY:
            comp_fig = go.Figure()
            comp_fig.add_trace(go.Scatter(x=stl.trend.index, y=stl.trend.values, name="Trend"))
            comp_fig.add_trace(go.Scatter(x=stl.seasonal.index, y=stl.seasonal.values, name="Seasonal"))
            comp_fig.add_trace(go.Scatter(x=stl.resid.index, y=stl.resid.values, name="Residual"))
            comp_fig.update_layout(title="STL Components")
            st.plotly_chart(comp_fig, use_container_width=True)
        else:
            st.write("Trend head:", stl.trend.head())
            st.write("Seasonal head:", stl.seasonal.head())
            st.write("Residual head:", stl.resid.head())
    except Exception as e:
        st.info(f"STL decomposition skipped: {e}")

# ---------- 9) Anomaly detection ----------
st.subheader("Anomaly Detection")
with st.expander("Detect anomalies (IsolationForest)"):
    if len(df_g) >= 30:
        X = df_g[["co2e_kg"]].fillna(method="ffill")
        iso = IsolationForest(contamination=0.02, random_state=42)
        labels = iso.fit_predict(X)
        df_g["anomaly"] = (labels == -1)
        if _HAS_PLOTLY:
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=df_g["timestamp"], y=df_g["co2e_kg"], mode="lines", name="CO₂e"))
            anom = df_g[df_g["anomaly"]]
            fig_a.add_trace(go.Scatter(x=anom["timestamp"], y=anom["co2e_kg"], mode="markers", name="Anomaly", marker=dict(size=8, symbol="x")))
            fig_a.update_layout(title="Anomaly detection")
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.write(df_g[df_g["anomaly"]].head())
    else:
        st.info("Need at least 30 periods for anomaly detection.")

# ---------- 10) Forecasting (ML & ARIMA) ----------
st.subheader("Forecasting")
with st.expander("Train models & forecast"):
    horizon = st.slider(f"Forecast horizon ({rule})", 7 if rule=="D" else 24, 180, 30)  # 기본 30 기간 예측
    # Feature engineering for ML
    dff = df_g.copy()
    dff["hour"] = dff["timestamp"].dt.hour
    dff["dow"] = dff["timestamp"].dt.dayofweek
    dff["month"] = dff["timestamp"].dt.month
    for lag in [1,2,3,6,12]:
        dff[f"lag_{lag}"] = dff["co2e_kg"].shift(lag)
    dff["roll_7"] = dff["co2e_kg"].rolling(7).mean()
    dff = dff.dropna().reset_index(drop=True)

    if len(dff) > horizon + 24:
        # Train/Test split by time
        train = dff.iloc[:-horizon].copy()
        test = dff.iloc[-horizon:].copy()

        features = [c for c in dff.columns if c not in ["timestamp","co2e_kg","anomaly"]]
        X_train, y_train = train[features], train["co2e_kg"]
        X_test, y_test = test[features], test["co2e_kg"]

        # 10-1) Gradient Boosting Regressor
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X_train, y_train)
        pred_ml = gbr.predict(X_test)
        mae_ml = mean_absolute_error(y_test, pred_ml)
        mape_ml = mean_absolute_percentage_error(y_test, pred_ml)

        # 10-2) ARIMA (auto order small search)
        try:
            # 간단 탐색: (p,d,q) within small grid
            best_aic, best_order, best_model = 1e18, None, None
            y_series = train.set_index("timestamp")["co2e_kg"]
            for p in [0,1,2]:
                for d in [0,1]:
                    for q in [0,1,2]:
                        try:
                            model = sm.tsa.ARIMA(y_series, order=(p,d,q)).fit()
                            if model.aic < best_aic:
                                best_aic, best_order, best_model = model.aic, (p,d,q), model
                        except:
                            pass
            if best_model is not None:
                fc = best_model.forecast(steps=horizon)
                pred_arima = fc.values
                mae_arima = mean_absolute_error(y_test.values, pred_arima)
                mape_arima = mean_absolute_percentage_error(y_test.values, pred_arima)
            else:
                pred_arima = None
                mae_arima = mape_arima = np.nan
        except Exception:
            pred_arima = None
            mae_arima = mape_arima = np.nan

        c1, c2 = st.columns(2)
        with c1:
            st.write("GBR — MAE:", round(mae_ml,2), "MAPE:", f"{mape_ml*100:.2f}%")
        with c2:
            st.write("ARIMA — MAE:", round(mae_arima,2) if np.isfinite(mae_arima) else "N/A",
                     "MAPE:", f"{mape_arima*100:.2f}%" if np.isfinite(mape_arima) else "N/A")

        # Plot
        if _HAS_PLOTLY:
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=train["timestamp"], y=train["co2e_kg"], name="Train"))
            fig_f.add_trace(go.Scatter(x=test["timestamp"], y=y_test, name="Actual"))
            fig_f.add_trace(go.Scatter(x=test["timestamp"], y=pred_ml, name="GBR Forecast"))
            if pred_arima is not None:
                fig_f.add_trace(go.Scatter(x=test["timestamp"], y=pred_arima, name="ARIMA Forecast"))
            fig_f.update_layout(title="Forecast comparison")
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.write("Forecast head (GBR):", pred_ml[:5])
    else:
        st.info("Not enough data after feature engineering for the chosen horizon.")

# ---------- 11) Optimization with Lagrange (emission or cost) ----------
st.subheader("Optimization (Lagrangian / constrained)")
with st.expander("Run optimization"):
    st.markdown("""
**두 가지 시나리오**
1) **비용 최소화**: 비용을 최소로 하되, 예상 CO₂e가 목표(한도) 이하  
2) **배출 최소화**: CO₂e 최소로 하되, 생산량(ton)을 목표 이상  
라그랑주 승수법의 아이디어를 사용한 제약최적화로 SciPy `minimize`를 활용합니다.
""")
    scenario = st.selectbox("Scenario", ["Minimize Cost (CO₂e cap)","Minimize Emissions (Production target)"])
    co2e_cap = st.number_input("CO₂e cap (kg)", value=float(df_g["co2e_kg"].quantile(0.75)) if not df_g.empty else 1_000.0, step=100.0)
    prod_target = st.number_input("Production target (ton)", value=float(df["production_ton"].mean()*24) if not df.empty else 100.0, step=10.0)

    # decision vars: electricity, gas (집계 기간 평균 개념)
    # 단가/배출계수
    price_elec, price_gas = 0.15, 0.08
    ef_elec, ef_gas = EMISSION_FACTOR_ELECTRICITY, EMISSION_FACTOR_GAS

    if scenario == "Minimize Cost (CO₂e cap)":
        def obj(x):  # 비용
            e, g = x
            return price_elec*e + price_gas*g
        cons = [{"type":"ineq", "fun": lambda x: co2e_cap - (ef_elec*x[0] + ef_gas*x[1])}]
        bounds = [(0, None),(0,None)]
        x0 = [co2e_cap/ef_elec*0.5, co2e_cap/ef_gas*0.5]
    else:
        def obj(x):  # 배출
            e, g = x
            return ef_elec*e + ef_gas*g
        # 간단 생산 함수: ton ~= alpha*e + beta*g (시演)
        alpha, beta = 0.02, 0.05
        cons = [{"type":"ineq", "fun": lambda x: (alpha*x[0] + beta*x[1]) - prod_target}]
        bounds = [(0, None),(0,None)]
        x0 = [prod_target/alpha*0.5, prod_target/beta*0.5]

    res = minimize(obj, x0, bounds=bounds, constraints=cons)
    e_opt, g_opt = float(res.x[0]), float(res.x[1])
    cost_opt = price_elec*e_opt + price_gas*g_opt
    co2e_opt = ef_elec*e_opt + ef_gas*g_opt
    st.write({
        "electricity": round(e_opt,2),
        "gas": round(g_opt,2),
        "cost": round(cost_opt,2),
        "co2e": round(co2e_opt,2),
        "success": bool(res.success)
    })
    st.caption("라그랑주 승수법 관점: 제약식을 만족하는 영역에서 목적함수 기울기와 제약의 기울기가 정비례(KKT)에 가까운 해를 찾도록 수치적으로 탐색")

# ---------- 12) Data table & download ----------
with st.expander("Data (resampled)"):
    st.dataframe(df_g, use_container_width=True)
csv_bytes = df_g.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download (resampled) CSV", data=csv_bytes, file_name="greenopt_resampled.csv", mime="text/csv")
