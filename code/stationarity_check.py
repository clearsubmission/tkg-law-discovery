"""
Checks whether the audit_signal_*.csv time series are stationary.
If susc_fixed / susc_active / n_events are trending/non-stationary,
the earlier R^2 values from linear regression on raw levels are at
risk of spurious regression (Granger & Newbold 1974) and need to be
redone on DIFFERENCED series instead.
"""

import pandas as pd
import numpy as np
import json
from statsmodels.tsa.stattools import adfuller

DATA_DIR = "./data"
OUT_DIR = f"{DATA_DIR}/law_discovery"

DATASETS = ["icews14", "icews18", "gdelt", "wiki", "yago"]
COLS = ["n_events", "susc_fixed", "susc_active"]


def adf_test(series):
    series = series.dropna().values
    if len(series) < 10 or np.std(series) < 1e-10:
        return {"adf_stat": None, "p_value": None, "stationary_at_5pct": None, "note": "insufficient variation"}
    result = adfuller(series, autolag="AIC")
    return {
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "stationary_at_5pct": bool(result[1] < 0.05),
        "n_obs": len(series),
    }


def main():
    all_results = {}
    for name in DATASETS:
        path = f"{DATA_DIR}/audit_signal_{name}.csv"
        df = pd.read_csv(path)
        all_results[name] = {}
        print(f"\n=== {name} ===")
        for col in COLS:
            res = adf_test(df[col])
            all_results[name][col] = res
            stat_str = "STATIONARY" if res.get("stationary_at_5pct") else "NON-STATIONARY (risk of spurious regression)"
            print(f"  {col:15s}  ADF p={res.get('p_value')}  -> {stat_str}")

    with open(f"{OUT_DIR}/stationarity_check.json", "w") as f:
        json.dump(all_results, f, indent=2)

    n_nonstationary = sum(
        1 for ds in all_results.values() for col in ds.values()
        if col.get("stationary_at_5pct") is False
    )
    total = sum(1 for ds in all_results.values() for col in ds.values() if col.get("stationary_at_5pct") is not None)
    print(f"\n{n_nonstationary}/{total} series are non-stationary at 5% level.")
    print(f"Saved to {OUT_DIR}/stationarity_check.json")


if __name__ == "__main__":
    main()
