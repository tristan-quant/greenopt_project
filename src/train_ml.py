from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from carbon_engine import add_carbon_columns

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "factory_data.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "reports" / "co2_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = add_carbon_columns(df)

    X = pd.DataFrame({
        "electricity_kwh": df["electricity_kwh"],
        "gas_m3": df["gas_m3"],
        "hour": df["timestamp"].dt.hour,
    })
    y = df["co2e_kg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()