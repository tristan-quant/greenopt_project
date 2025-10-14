from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_factory_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])

def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)