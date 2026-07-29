"""
Re-runs the structural law search on FIRST-DIFFERENCED series to correct
for the non-stationarity found in stationarity_check.py. If the sign-split
law survives differencing, it's genuine; if it disappears, the original
result was largely a trend artifact.
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dataclasses import dataclass

DATA_DIR = "./data"
OUT_DIR = f"{DATA_DIR}/law_discovery"

DATASET_TYPES = {
    "icews14": "event_stream",
    "icews18": "event_stream",
    "gdelt":   "event_stream",
    "wiki":    "interval_snapshot",
    "yago":    "interval_snapshot",
}

FEATURES = ["susc_fixed", "susc_active"]
TARGET = "n_events"


def load_and_difference():
    dfs = {}
    for name, dtype in DATASET_TYPES.items():
        path = f"{DATA_DIR}/audit_signal_{name}.csv"
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path)
        diff = raw[["n_events", "susc_fixed", "susc_active"]].diff().dropna().reset_index(drop=True)
        diff["dataset_type"] = dtype
        diff["dataset_name"] = name
        dfs[name] = diff
    return dfs


UNARY_OPS = {
    "identity": lambda x: x,
    "sign_abs_log1p": lambda x: np.sign(x) * np.log1p(np.abs(x)),  # log1p doesn't work on negatives post-diff
    "zscore": lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8),
    "rank": lambda x: pd.Series(x).rank(pct=True).values,
}

BINARY_OPS = {
    "add":   lambda a, b: a + b,
    "mul":   lambda a, b: a * b,
    "diff":  lambda a, b: a - b,
}


@dataclass
class Candidate:
    expr_str: str
    values_fn: callable


def enumerate_candidates():
    cands = []
    for feat in FEATURES:
        for uname, ufn in UNARY_OPS.items():
            cands.append(Candidate(
                expr_str=f"{uname}(d_{feat})",
                values_fn=(lambda df, f=feat, fn=ufn: fn(df[f].values)),
            ))
    for f1, f2 in itertools.combinations(FEATURES, 2):
        for u1n, u1f in UNARY_OPS.items():
            for u2n, u2f in UNARY_OPS.items():
                for bn, bf in BINARY_OPS.items():
                    expr = f"{bn}({u1n}(d_{f1}), {u2n}(d_{f2}))"
                    cands.append(Candidate(
                        expr_str=expr,
                        values_fn=(lambda df, a=f1, b=f2, u1=u1f, u2=u2f, bfn=bf:
                                   bfn(u1(df[a].values), u2(df[b].values))),
                    ))
    return cands


def verify_candidate(cand, dfs, target=TARGET):
    results = {}
    for name, df in dfs.items():
        try:
            x = cand.values_fn(df)
        except Exception:
            continue
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = df[target].values
        if np.std(x) < 1e-10:
            continue
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        pred = reg.predict(x.reshape(-1, 1))
        r2 = r2_score(y, pred)
        results[name] = {
            "coef": float(reg.coef_[0]),
            "sign": int(np.sign(reg.coef_[0])),
            "r2": float(r2),
            "dataset_type": df["dataset_type"].iloc[0],
        }
    return results


def score_law(cand, vr, min_r2=0.02):
    good = {k: v for k, v in vr.items() if v["r2"] >= min_r2}
    if len(good) < 2:
        return None
    by_type = {}
    for name, v in good.items():
        by_type.setdefault(v["dataset_type"], []).append(v["sign"])
    consistent_within = all(len(set(s)) == 1 for s in by_type.values())
    if not consistent_within:
        return None
    types_present = list(by_type.keys())
    sign_split = len(types_present) >= 2 and len({by_type[t][0] for t in types_present}) > 1
    return {
        "expr": cand.expr_str,
        "n_datasets_fit": len(good),
        "avg_r2": float(np.mean([v["r2"] for v in good.values()])),
        "sign_split_across_types": sign_split,
        "per_dataset": good,
    }


def main():
    dfs = load_and_difference()
    print(f"Loaded and differenced {len(dfs)} datasets: {list(dfs.keys())}")
    for name, df in dfs.items():
        print(f"  {name}: {len(df)} differenced observations")

    candidates = enumerate_candidates()
    print(f"\nEnumerating {len(candidates)} candidates on DIFFERENCED series...")

    discovered = []
    all_r2 = []
    for cand in candidates:
        vr = verify_candidate(cand, dfs)
        for v in vr.values():
            all_r2.append(v["r2"])
        law = score_law(cand, vr)
        if law is not None:
            discovered.append(law)

    discovered.sort(key=lambda l: (-l["sign_split_across_types"], -l["avg_r2"]))

    print(f"\nMax single-dataset R^2 on differenced data: {max(all_r2) if all_r2 else 'N/A':.4f}")
    print(f"Discovered {len(discovered)} laws surviving differencing. Top 10:")
    for law in discovered[:10]:
        print(f"  {law['expr']:45s} avg_r2={law['avg_r2']:.4f}  "
              f"sign_split={law['sign_split_across_types']}  n_fit={law['n_datasets_fit']}")

    with open(f"{OUT_DIR}/discovered_laws_differenced.json", "w") as f:
        json.dump(discovered, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/discovered_laws_differenced.json")


if __name__ == "__main__":
    main()
