"""
Baseline comparison: naive linear correlation vs brute-force enumeration
vs PySR symbolic regression, all evaluated on a held-out test split of
the DIFFERENCED series, using R^2 as the common metric.

This is the ablation a reviewer will expect: does the search method
actually add value over just checking susc_fixed directly?
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sympy
from sympy import symbols, lambdify

DATA_DIR = "./data"
OUT_DIR = f"{DATA_DIR}/law_discovery"

DATASET_TYPES = {
    "icews14": "event_stream",
    "icews18": "event_stream",
    "gdelt":   "event_stream",
    "wiki":    "interval_snapshot",
    "yago":    "interval_snapshot",
}

TEST_SIZE = 0.3
SEED = 42


def load_and_difference(name):
    path = f"{DATA_DIR}/audit_signal_{name}.csv"
    raw = pd.read_csv(path)
    diff = raw[["n_events", "susc_fixed", "susc_active"]].diff().dropna().reset_index(drop=True)
    return diff


def naive_baseline_r2(train, test):
    """Simplest possible baseline: linear regression on raw susc_fixed alone."""
    x_train = train["susc_fixed"].values.reshape(-1, 1)
    y_train = train["n_events"].values
    x_test = test["susc_fixed"].values.reshape(-1, 1)
    y_test = test["n_events"].values

    reg = LinearRegression().fit(x_train, y_train)
    pred = reg.predict(x_test)
    return float(r2_score(y_test, pred)), float(reg.coef_[0])


def bruteforce_baseline_r2(train, test, expr_name="identity(d_susc_fixed)"):
    """Re-fits the brute-force search's best surviving law (identity(d_susc_fixed))
    on train, evaluates on test, for a fair train/test comparison."""
    x_train = train["susc_fixed"].values.reshape(-1, 1)
    y_train = train["n_events"].values
    x_test = test["susc_fixed"].values.reshape(-1, 1)
    y_test = test["n_events"].values

    reg = LinearRegression().fit(x_train, y_train)
    pred = reg.predict(x_test)
    return float(r2_score(y_test, pred))


def pysr_baseline_r2(train, test, pysr_eq_str):
    """Parses PySR's discovered equation (fixed coefficients from full-data fit)
    and evaluates it on the held-out test split -- this checks generalization,
    not just in-sample fit."""
    sf, sa = symbols("susc_fixed susc_active")
    try:
        expr = sympy.sympify(pysr_eq_str)
    except Exception as e:
        return None, str(e)

    free_syms = expr.free_symbols
    variables = [s for s in [sf, sa] if s in free_syms]
    if not variables:
        # constant equation edge case
        pred_fn = lambda df: np.full(len(df), float(expr))
    else:
        f = lambdify(variables, expr, "numpy")
        def pred_fn(df, f=f, variables=variables):
            args = [df[str(v)].values for v in variables]
            return f(*args)

    y_test = test["n_events"].values
    try:
        pred = pred_fn(test)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return float(r2_score(y_test, pred)), None
    except Exception as e:
        return None, str(e)


def main():
    with open(f"{OUT_DIR}/pysr_results.json") as f:
        pysr_results = json.load(f)

    comparison = {}
    for name, dtype in DATASET_TYPES.items():
        diff_df = load_and_difference(name)
        train, test = train_test_split(diff_df, test_size=TEST_SIZE, random_state=SEED, shuffle=False)

        naive_r2, naive_coef = naive_baseline_r2(train, test)
        bf_r2 = bruteforce_baseline_r2(train, test)
        pysr_eq = pysr_results[name]["best_equation"]
        pysr_r2, pysr_err = pysr_baseline_r2(train, test, pysr_eq)

        comparison[name] = {
            "dataset_type": dtype,
            "naive_linear_r2_test": naive_r2,
            "naive_coef_sign": int(np.sign(naive_coef)),
            "bruteforce_r2_test": bf_r2,
            "pysr_r2_test": pysr_r2,
            "pysr_equation": pysr_eq,
            "pysr_error": pysr_err,
        }

        print(f"\n=== {name} ({dtype}) ===")
        print(f"  Naive linear (susc_fixed only):  R^2_test = {naive_r2:.4f}  (sign={np.sign(naive_coef):+.0f})")
        print(f"  Brute-force best law:            R^2_test = {bf_r2:.4f}")
        if pysr_r2 is not None:
            print(f"  PySR discovered equation:        R^2_test = {pysr_r2:.4f}   [{pysr_eq}]")
        else:
            print(f"  PySR discovered equation:        FAILED to evaluate ({pysr_err})")

    with open(f"{OUT_DIR}/baseline_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/baseline_comparison.json")


if __name__ == "__main__":
    main()
