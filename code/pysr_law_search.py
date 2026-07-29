"""
Real symbolic regression backend (PySR/SymbolicRegression.jl) searching
for closed-form laws relating susc_fixed / susc_active to n_events, on
the DIFFERENCED series (stationarity-corrected).
"""

import os
import json
import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

DATA_DIR = "./data"
OUT_DIR = f"{DATA_DIR}/law_discovery"

DATASET_TYPES = {
    "icews14": "event_stream",
    "icews18": "event_stream",
    "gdelt":   "event_stream",
    "wiki":    "interval_snapshot",
    "yago":    "interval_snapshot",
}


def load_and_difference(name):
    path = f"{DATA_DIR}/audit_signal_{name}.csv"
    raw = pd.read_csv(path)
    diff = raw[["n_events", "susc_fixed", "susc_active"]].diff().dropna().reset_index(drop=True)
    return diff


def run_pysr_for_dataset(name, diff_df):
    X = diff_df[["susc_fixed", "susc_active"]].values.astype(np.float32)
    y = diff_df["n_events"].values.astype(np.float32)

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log_abs(x) = log(abs(x) + one(x) * 1f-8)", "square"],
        extra_sympy_mappings={"log_abs": lambda x: sympy.log(sympy.Abs(x) + 1e-8)},
        model_selection="best",
        maxsize=15,
        populations=20,
        procs=4,
        random_state=42,
        deterministic=False,
        parallelism="multiprocessing",
        temp_equation_file=True,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y, variable_names=["susc_fixed", "susc_active"])

    best_eq = str(model.sympy())
    best_row = model.equations_.iloc[-1]

    return {
        "dataset": name,
        "dataset_type": DATASET_TYPES[name],
        "best_equation": best_eq,
        "loss": float(best_row["loss"]),
        "complexity": int(best_row["complexity"]),
        "all_equations": model.equations_[["complexity", "loss", "equation"]].to_dict(orient="records"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for name in DATASET_TYPES:
        print(f"\n=== Running PySR on {name} (differenced) ===")
        diff_df = load_and_difference(name)
        res = run_pysr_for_dataset(name, diff_df)
        results[name] = res
        print(f"  Best equation: {res['best_equation']}")
        print(f"  Loss: {res['loss']:.4f}  Complexity: {res['complexity']}")

    with open(f"{OUT_DIR}/pysr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved all PySR results to {OUT_DIR}/pysr_results.json")
    print("\nSummary across datasets:")
    for name, res in results.items():
        print(f"  {name:10s} ({res['dataset_type']:18s})  {res['best_equation']}")


if __name__ == "__main__":
    main()
