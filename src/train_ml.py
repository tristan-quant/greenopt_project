from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from carbon_engine import add_carbon_columns

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "factory_data.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "reports" / "co2_model.pkl"
METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "co2_metrics.json"

REQUIRED_COLS = {"timestamp", "electricity_kwh", "gas_m3"}

def _cyclical_hour(ts: pd.Series) -> pd.DataFrame:
    """0~23시를 순환형 특성으로 인코딩"""
    hour = ts.dt.hour.astype(float)
    rad = 2 * np.pi * (hour / 24.0)
    return pd.DataFrame({
        "hour_sin": np.sin(rad),
        "hour_cos": np.cos(rad),
    }, index=ts.index)

def _temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """시간순 정렬 후 마지막 test_ratio 비율을 테스트로 사용"""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    cut = int(n * (1 - test_ratio))
    return df_sorted.iloc[:cut], df_sorted.iloc[cut:]

def main():
    # 1) 로드 + 컬럼 검증
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2) 배출량 컬럼 생성
    df = add_carbon_columns(df)

    # 3) 필수 타깃/피처 결측 제거
    base_cols = ["electricity_kwh", "gas_m3", "timestamp", "co2e_kg"]
    df = df.dropna(subset=base_cols).copy()

    # 4) 시간 특성 인코딩(순환형)
    hour_feats = _cyclical_hour(df["timestamp"])

    # 5) 입력 X / 타깃 y 구성
    X = pd.concat([
        df[["electricity_kwh", "gas_m3"]].astype(float),
        hour_feats
    ], axis=1)
    y = df["co2e_kg"].astype(float)

    # 6) 시간 기반 분할(누수 방지)
    data = pd.concat([X, y, df[["timestamp"]]], axis=1)
    train_df, test_df = _temporal_split(data, test_ratio=0.2)

    X_train = train_df[X.columns]
    y_train = train_df["co2e_kg"]
    X_test  = test_df[X.columns]
    y_test  = test_df["co2e_kg"]

    # 7) 스케일링 + 선형회귀 (간단 파이프라인 수동 구현)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # 8) 평가
    y_pred = model.predict(X_test_s)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # 9) 아티팩트 저장
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "features": list(X.columns)}, MODEL_PATH)

    metrics = {"r2": r2, "mae": mae, "rmse": rmse, "n_train": len(X_train), "n_test": len(X_test)}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Metrics -> R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"Saved metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()

    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
