from __future__ import annotations
import pandas as pd

EMISSION_FACTOR_ELECTRICITY = 0.475  # kg CO2e / kWh (placeholder)
EMISSION_FACTOR_GAS        = 2.0     # kg CO2e / m3  (placeholder)

def calc_co2e(electricity_kwh: float, gas_m3: float) -> float:
    return electricity_kwh * EMISSION_FACTOR_ELECTRICITY + gas_m3 * EMISSION_FACTOR_GAS

def add_carbon_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["co2e_kg"] = (
        df["electricity_kwh"]*EMISSION_FACTOR_ELECTRICITY
        + df["gas_m3"]*EMISSION_FACTOR_GAS
    )
    def _pcf(row):
        prod = row.get("production_ton", 0.0)
        return (row["co2e_kg"] / prod) if prod and prod > 0 else None
    df["pcf_kg_per_ton"] = df.apply(_pcf, axis=1)
    return df