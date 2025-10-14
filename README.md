# GreenOpt — AI-driven Carbon Footprint Calculator

## What this is
A minimal, production-style starter to calculate CO₂e from electricity/gas usage,
estimate product carbon footprint (PCF), and visualize time-series emissions.

## Project layout
```
greenopt/
  ├─ src/
  │   ├─ carbon_engine.py        # CO₂e & PCF core logic
  │   ├─ train_ml.py             # simple ML baseline (LinearRegression)
  │   └─ data_utils.py           # I/O helpers
  ├─ data/
  │   └─ factory_data.csv        # sample hourly dataset (7 days)
  ├─ app/
  │   └─ app.py                  # Streamlit dashboard
  ├─ reports/                    # figures / exports
  ├─ notebooks/                  # EDA/experiments
  ├─ environment.yml
  └─ requirements.txt
```

## Quickstart
```bash
# (option A) conda
conda env create -f environment.yml
conda activate greenopt

# (option B) pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run app/app.py
```

## Notes
- Emission factors are placeholders (demo only). Replace with jurisdiction/country-specific factors.
- Sample dataset is randomly generated for prototyping.