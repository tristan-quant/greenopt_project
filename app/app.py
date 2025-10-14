# =====================================================
# GreenOpt — Digital ESG Engine
# =====================================================
from __future__ import annotations

# ---------- 0) Auto-install guard ----------
import sys, subprocess

def _ensure(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

for pkg in [
    "streamlit","pandas","numpy","plotly","scipy","Pillow",
    "scikit-learn","statsmodels","xgboost","catboost","reportlab"
]:
    _ensure(pkg)

# ---------- 1) Imports ----------
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st   # ✅ 이제부터 st 사용 가능
from PIL import Image

# ---------- 2) Optional flags ----------
_HAS_PLOTLY = False
_HAS_STATSMODELS = False
_HAS_XGBOOST = False
_HAS_CATBOOST = False

# Plotly (optional)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None

# statsmodels (optional)
try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    sm = None

# xgboost (optional)
try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None

# catboost (optional)
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None

# 나머지 import
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- 3) Page config ----------
st.set_page_config(page_title="GreenOpt — Digital ESG Engine", layout="wide")

# ✅ 이제 caption을 여기에 써야 정상작동
st.caption(f"statsmodels available: {_HAS_STATSMODELS}")
st.caption(f"plotly available: {_HAS_PLOTLY}")
st.caption(f"xgboost available: {_HAS_XGBOOST}")
st.caption(f"catboost available: {_HAS_CATBOOST}")

# ---------- 3) Emission factors & helpers ----------
EMISSION_FACTOR_ELECTRICITY_DEFAULT = 0.475  # kg CO2e/kWh (location-based 예시)
EMISSION_FACTOR_GAS = 2.0                    # kg CO2e/m3

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        # fallback: 3년치 시간단위 샘플 생성
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
    # 시간 해상도 변경: H/D/W/M
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

# ---------- 5) Sidebar: data & baseline controls ----------
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

    st.header("Scope 2 method")
    scope2_method = st.selectbox("Electricity EF method", ["Location-based", "Market-based"], index=0)
    if scope2_method == "Location-based":
        ef_elec_input = st.number_input("EF (location-based, kg/kWh)",
                                        value=float(EMISSION_FACTOR_ELECTRICITY_DEFAULT), step=0.01)
    else:
        # 시장기반: 구매전력 보증서/RECs 등 반영(예시 기본값 0.0)
        ef_elec_input = st.number_input("EF (market-based, kg/kWh)", value=0.000, step=0.01)

    st.header("Filters")
    tmin, tmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    start_date, end_date = st.date_input("Date range", value=(tmin, tmax), min_value=tmin, max_value=tmax)
    sel_lines = st.multiselect("Line", sorted(df["line"].dropna().unique().tolist()) if "line" in df.columns else [])
    sel_products = st.multiselect("Product", sorted(df["product"].dropna().unique().tolist()) if "product" in df.columns else [])
    rule = st.selectbox("Time granularity", ["H","D","W","M"], index=1)  # 기본 일간

# ---------- 6) Multivariate features (temperature/utilization) ----------
with st.sidebar:
    st.header("External Features")
    aux = st.file_uploader("Upload AUX CSV (timestamp, temperature_c, utilization_pct ...)", type=["csv"], key="aux")
    if aux:
        df_aux = pd.read_csv(aux)
        df_aux["timestamp"] = pd.to_datetime(df_aux["timestamp"])
        df = df.merge(df_aux, on="timestamp", how="left")
        st.success("AUX merged: " + ", ".join([c for c in df_aux.columns if c != "timestamp"]))
    else:
        # AUX 미제공 시 예시 피처 생성 (동작 보장)
        if "temperature_c" not in df.columns:
            t = (df["timestamp"] - df["timestamp"].min()).dt.days.values
            df["temperature_c"] = 18 + 7*np.sin(2*np.pi*(t/365)) + np.random.normal(0, 1.5, len(df))
        if "utilization_pct" not in df.columns:
            base = 70 + 20*np.sin(2*np.pi*(t/30)) + np.random.normal(0, 5, len(df))
            df["utilization_pct"] = np.clip(base, 20, 100)

# ---------- 7) Apply filters ----------
mask = (df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)-pd.Timedelta(seconds=1))
if sel_lines and "line" in df.columns:       mask &= df["line"].isin(sel_lines)
if sel_products and "product" in df.columns:  mask &= df["product"].isin(sel_products)
df = df.loc[mask].copy()

# ---------- 8) Carbon columns with chosen Scope2 EF ----------
def add_carbon_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    dfc = df_in.copy()
    ef_elec = ef_elec_input
    ef_gas  = EMISSION_FACTOR_GAS
    dfc["co2e_kg"] = dfc["electricity_kwh"] * ef_elec + dfc["gas_m3"] * ef_gas
    dfc["pcf_kg_per_ton"] = np.where(dfc["production_ton"]>0, dfc["co2e_kg"]/dfc["production_ton"], np.nan)
    return dfc

df = add_carbon_columns(df)
df_g = resample_df(df, rule)

# ---------- 9) KPI ----------
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

# ---------- 10) Overview chart ----------
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

# ---------- 11) STL decomposition ----------
st.subheader("Seasonal-Trend Decomposition (STL)")
with st.expander("Show STL decomposition"):
    if not _HAS_STATSMODELS:
        st.info("statsmodels가 설치되어 있지 않아 STL 분석을 건너뜁니다. requirements.txt에 statsmodels를 추가하세요.")
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
                comp_fig = go.Figure()
                comp_fig.add_trace(go.Scatter(x=stl.trend.index, y=stl.trend.values, name="Trend"))
                comp_fig.add_trace(go.Scatter(x=stl.seasonal.index, y=stl.seasonal.values, name="Seasonal"))
                comp_fig.add_trace(go.Scatter(x=stl.resid.index, y=stl.resid.values, name="Residual"))
                comp_fig.update_layout(title="STL Components")
                st.plotly_chart(comp_fig, use_container_width=True)
            else:
                st.write("Trend(head):", stl.trend.head())
                st.write("Seasonal(head):", stl.seasonal.head())
                st.write("Residual(head):", stl.resid.head())
        except Exception as e:
            st.info(f"STL decomposition skipped: {e}")

# ---------- 12) Anomaly detection ----------
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
            fig_a.add_trace(go.Scatter(x=anom["timestamp"], y=anom["co2e_kg"], mode="markers", name="Anomaly",
                                       marker=dict(size=8, symbol="x")))
            fig_a.update_layout(title="Anomaly detection")
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.write(df_g[df_g["anomaly"]].head())
    else:
        st.info("Need at least 30 periods for anomaly detection.")

# ---------- 13) Forecasting (GBR, ARIMA, + AutoML: XGBoost/CatBoost) ----------
st.subheader("Forecasting")
with st.expander("Train models & forecast (with AutoML)"):
    horizon = st.slider(f"Forecast horizon ({rule})", 7 if rule=="D" else 24, 180, 30)

    # Feature engineering
    dff = df_g.copy()
    dff["hour"] = dff["timestamp"].dt.hour
    dff["dow"] = dff["timestamp"].dt.dayofweek
    dff["month"] = dff["timestamp"].dt.month

    for lag in [1,2,3,6,12]:
        dff[f"lag_{lag}"] = dff["co2e_kg"].shift(lag)
    dff["roll_7"] = dff["co2e_kg"].rolling(7).mean()

    for col in ["temperature_c", "utilization_pct"]:
        if col in dff.columns:
            for lag in [1,2,3]:
                dff[f"{col}_lag{lag}"] = dff[col].shift(lag)
            dff[f"{col}_roll7"] = dff[col].rolling(7).mean()

    dff = dff.dropna().reset_index(drop=True)
    if len(dff) <= horizon + 24:
        st.info("Not enough data after feature engineering for the chosen horizon.")
    else:
        train = dff.iloc[:-horizon].copy()
        test  = dff.iloc[-horizon:].copy()
        y_train, y_test = train["co2e_kg"], test["co2e_kg"]
        features = [c for c in dff.columns if c not in ["timestamp","co2e_kg","anomaly"]]
        X_train, X_test = train[features], test[features]

        # Baseline: GBR
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X_train, y_train)
        pred_gbr = gbr.predict(X_test)
        mae_gbr  = mean_absolute_error(y_test, pred_gbr)

        # ARIMA small grid (guarded)
        pred_arima, mae_arima, best_order = None, np.inf, None
        if _HAS_STATSMODELS:
            try:
                y_series = train.set_index("timestamp")["co2e_kg"]
                best_aic, best_model, best_order = 1e18, None, None
                for p in [0,1,2]:
                    for d in [0,1]:
                        for q in [0,1,2]:
                            try:
                                m = sm.tsa.ARIMA(y_series, order=(p,d,q)).fit()
                                if m.aic < best_aic:
                                    best_aic, best_model, best_order = m.aic, m, (p,d,q)
                            except:
                                pass
                if best_model is not None:
                    pred_arima = best_model.forecast(steps=horizon).values
                    mae_arima  = mean_absolute_error(y_test.values, pred_arima)
            except Exception:
                pred_arima, mae_arima, best_order = None, np.inf, None
        else:
            st.info("ARIMA skipped (statsmodels not available).")

        # AutoML: XGBoost (if available)
        best_name, best_model_obj, best_pred, best_mae = "GBR", gbr, pred_gbr, mae_gbr
        if _HAS_XGBOOST:
            try:
                xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)
                param_xgb = {
                    "n_estimators": [300, 400, 600],
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0],
                }
                tscv = TimeSeriesSplit(n_splits=3)
                rs_xgb = RandomizedSearchCV(xgb, param_distributions=param_xgb, n_iter=8, cv=tscv,
                                            scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
                rs_xgb.fit(X_train, y_train)
                pred_xgb = rs_xgb.best_estimator_.predict(X_test)
                mae_xgb  = mean_absolute_error(y_test, pred_xgb)
                if mae_xgb < best_mae:
                    best_name, best_model_obj, best_pred, best_mae = "XGBoost", rs_xgb.best_estimator_, pred_xgb, mae_xgb
            except Exception:
                pass
        else:
            st.caption("XGBoost not available — skipped.")

        # AutoML: CatBoost (if available)
        if _HAS_CATBOOST:
            try:
                cbr = CatBoostRegressor(
                    iterations=500, depth=6, learning_rate=0.05, loss_function="MAE",
                    verbose=False, random_state=42
                )
                cbr.fit(X_train, y_train)
                pred_cbr = cbr.predict(X_test)
                mae_cbr  = mean_absolute_error(y_test, pred_cbr)
                if mae_cbr < best_mae:
                    best_name, best_model_obj, best_pred, best_mae = "CatBoost", cbr, pred_cbr, mae_cbr
            except Exception:
                pass
        else:
            st.caption("CatBoost not available — skipped.")

        colA, colB, colC = st.columns(3)
        colA.metric("Best model", best_name)
        colB.metric("Best MAE", f"{best_mae:,.2f}")
        colC.metric("GBR MAE",  f"{mae_gbr:,.2f}")
        if pred_arima is not None and np.isfinite(mae_arima):
            st.caption(f"ARIMA({best_order}) MAE: {mae_arima:,.2f}")

        if _HAS_PLOTLY:
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=train["timestamp"], y=train["co2e_kg"], name="Train"))
            fig_f.add_trace(go.Scatter(x=test["timestamp"],  y=y_test, name="Actual"))
            fig_f.add_trace(go.Scatter(x=test["timestamp"],  y=best_pred, name=f"{best_name} Forecast"))
            if pred_arima is not None:
                fig_f.add_trace(go.Scatter(x=test["timestamp"], y=pred_arima, name="ARIMA Forecast"))
            fig_f.update_layout(title="Forecast comparison (Best vs ARIMA)")
            st.plotly_chart(fig_f, use_container_width=True)

        # Save to session for scenario & report
        st.session_state["best_model_name"] = best_name
        st.session_state["best_pred_series"] = pd.Series(best_pred, index=test["timestamp"])
        st.session_state["y_actual_series"]  = pd.Series(y_test.values, index=test["timestamp"])

# ---------- 14) Optimization (Lagrangian / constrained) ----------
st.subheader("Optimization (Lagrangian / constrained)")
with st.expander("Run optimization"):
    st.markdown("""
**Two scenarios**
1) **Minimize Cost** subject to CO₂e cap  
2) **Minimize Emissions** subject to production target  
SciPy `minimize` is used (KKT-style numeric search).
""")
    scenario = st.selectbox("Scenario", ["Minimize Cost (CO₂e cap)","Minimize Emissions (Production target)"])
    co2e_cap = st.number_input("CO₂e cap (kg)", value=float(df_g["co2e_kg"].quantile(0.75)) if not df_g.empty else 1_000.0, step=100.0)
    prod_target = st.number_input("Production target (ton)", value=float(df["production_ton"].mean()*24) if not df.empty else 100.0, step=10.0)

    # decision variables: electricity, gas (aggregated period units)
    price_elec, price_gas = 0.15, 0.08
    ef_elec, ef_gas = ef_elec_input, EMISSION_FACTOR_GAS

    if scenario == "Minimize Cost (CO₂e cap)":
        def obj(x):
            e, g = x
            return price_elec*e + price_gas*g
        cons = [{"type":"ineq", "fun": lambda x: co2e_cap - (ef_elec*x[0] + ef_gas*x[1])}]
        bounds = [(0, None),(0,None)]
        x0 = [co2e_cap/ef_elec*0.5 if ef_elec>0 else 0.0, co2e_cap/ef_gas*0.5]
    else:
        def obj(x):
            e, g = x
            return ef_elec*e + ef_gas*g
        # simple production function (toy)
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
    st.caption("Lagrangian view: at optimum, objective gradient ∥ constraint gradient (KKT-like).")

# ---------- 15) Carbon price scenarios (ETS/CBAM) ----------
st.subheader("Carbon Pricing Scenarios")
with st.expander("Apply ETS/CBAM to forecasts"):
    price_per_ton = st.number_input("Carbon price (per tCO₂e, e.g., €)", value=85.0, step=1.0)
    fx_rate       = st.number_input("FX rate (local per €)", value=1450.0, step=10.0)
    cbam_markup   = st.slider("CBAM surcharge (%)", min_value=0, max_value=50, value=0, step=1)

    if "best_pred_series" in st.session_state:
        pred = st.session_state["best_pred_series"].copy()  # kg
        pred_ton = pred / 1000.0
        base_cost_eur = pred_ton * price_per_ton
        cbam_cost_eur = base_cost_eur * (1 + cbam_markup/100)
        base_cost_local = base_cost_eur * fx_rate
        cbam_cost_local = cbam_cost_eur * fx_rate

        df_cost = pd.DataFrame({
            "timestamp": pred.index,
            "co2e_kg_forecast": pred.values,
            "cost_base_eur": base_cost_eur.values,
            "cost_cbam_eur": cbam_cost_eur.values,
            "cost_base_local": base_cost_local.values,
            "cost_cbam_local": cbam_cost_local.values,
        })

        if _HAS_PLOTLY:
            fig_c = px.line(df_cost, x="timestamp", y=["cost_base_local","cost_cbam_local"],
                            title="Carbon cost (local currency)")
            st.plotly_chart(fig_c, use_container_width=True)

        st.dataframe(df_cost.tail(12), use_container_width=True)
        st.session_state["df_cost"] = df_cost
    else:
        st.info("Run Forecasting first to generate prediction series.")

# ---------- 16) Data table & CSV ----------
with st.expander("Data (resampled)"):
    st.dataframe(df_g, use_container_width=True)
csv_bytes = df_g.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download (resampled) CSV", data=csv_bytes, file_name="greenopt_resampled.csv", mime="text/csv")

# ---------- 17) KPI / Report PDF ----------
st.subheader("Export KPI / Report to PDF")

def build_pdf(df_summary: pd.DataFrame, kpis: dict, note: str = "") -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "GreenOpt — Carbon Intelligence Report")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Scope 2 method: {scope2_method} | EF_elec: {ef_elec_input} kg/kWh")
    y -= 15
    c.drawString(40, y, f"Period: {str(start_date)} ~ {str(end_date)} (rule: {rule})")
    y -= 25

    # KPIs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "KPIs")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in kpis.items():
        c.drawString(50, y, f"- {k}: {v}")
        y -= 14

    # Summary (tail)
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Recent Summary (tail)")
    y -= 18
    c.setFont("Helvetica", 9)
    tail = df_summary.tail(10).copy()
    tail_cols = list(tail.columns)[:5]
    c.drawString(50, y, " | ".join(tail_cols))
    y -= 12
    for _, row in tail.iterrows():
        line = " | ".join([str(row[col])[:20] for col in tail_cols])
        c.drawString(50, y, line)
        y -= 12
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 9)

    # Notes
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Notes")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in note.split("\n"):
        c.drawString(50, y, line[:100])
        y -= 12
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

if not df_g.empty:
    kpis = {
        "Total CO₂e (kg)": f"{df_g['co2e_kg'].sum():,.0f}",
        f"Avg PCF (kg/ton)": f"{df_g['pcf_kg_per_ton'].mean():,.2f}",
        f"Periods ({rule})": f"{len(df_g):,}"
    }
    note = "This report includes KPIs, Scope 2 method, and recent summary.\nForecast & pricing results are scenario-based and indicative."
    pdf_bytes = build_pdf(df_g, kpis, note)
    st.download_button("📄 Download KPI Report (PDF)", data=pdf_bytes, file_name="greenopt_report.pdf", mime="application/pdf")
else:
    st.info("No data to export.")
