"""
Structural Law Discovery Engine for TKGs (v2 - matched to actual schema)
Searches over transforms of susc_fixed / susc_active / n_events from
audit_signal_*.csv window-level files, validates discovered laws against
the ground-truth sign-split already in cross_benchmark_audit_summary.csv,
and confirms the recency/MRR finding from stratified_activity_results.csv.
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

# dataset_type assignment per your mechanistic explanation:
# event-stream (raw event timestamps) vs interval-snapshot (aggregated intervals)
DATASET_TYPES = {
    "icews14": "event_stream",
    "icews18": "event_stream",
    "gdelt":   "event_stream",
    "wiki":    "interval_snapshot",
    "yago":    "interval_snapshot",
}

FEATURES = ["susc_fixed", "susc_active"]
TARGET = "n_events"


def load_audit_signals():
    dfs = {}
    for name, dtype in DATASET_TYPES.items():
        path = f"{DATA_DIR}/audit_signal_{name}.csv"
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            continue
        df = pd.read_csv(path)
        df["dataset_type"] = dtype
        df["dataset_name"] = name
        dfs[name] = df
    return dfs


# ---------------------------------------------------------------------------
# DSL primitives (unary + binary), same 10-op family as Synth-TKG DSL
# ---------------------------------------------------------------------------
UNARY_OPS = {
    "identity": lambda x: x,
    "log1p":    lambda x: np.log1p(np.clip(x, 0, None)),
    "sqrt":     lambda x: np.sqrt(np.clip(x, 0, None)),
    "inv":      lambda x: 1.0 / (1.0 + np.abs(x)),
    "zscore":   lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8),
    "rank":     lambda x: pd.Series(x).rank(pct=True).values,
}

BINARY_OPS = {
    "add":   lambda a, b: a + b,
    "mul":   lambda a, b: a * b,
    "ratio": lambda a, b: a / (b + 1e-8),
    "diff":  lambda a, b: a - b,
}


@dataclass
class Candidate:
    expr_str: str
    values_fn: callable
    depth: int


def enumerate_candidates():
    cands = []
    for feat in FEATURES:
        for uname, ufn in UNARY_OPS.items():
            cands.append(Candidate(
                expr_str=f"{uname}({feat})",
                values_fn=(lambda df, f=feat, fn=ufn: fn(df[f].values)),
                depth=1,
            ))
    for f1, f2 in itertools.combinations(FEATURES, 2):
        for u1n, u1f in UNARY_OPS.items():
            for u2n, u2f in UNARY_OPS.items():
                for bn, bf in BINARY_OPS.items():
                    expr = f"{bn}({u1n}({f1}), {u2n}({f2}))"
                    cands.append(Candidate(
                        expr_str=expr,
                        values_fn=(lambda df, a=f1, b=f2, u1=u1f, u2=u2f, bfn=bf:
                                   bfn(u1(df[a].values), u2(df[b].values))),
                        depth=2,
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


def score_law(cand, vr, min_r2=0.05):
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


# ---------------------------------------------------------------------------
# Ground-truth cross-check against your already-computed sign split
# ---------------------------------------------------------------------------
def load_ground_truth_signs():
    path = f"{DATA_DIR}/cross_benchmark_audit_summary.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["dataset_key"] = df["dataset"].str.lower().str.replace("-", "").str.strip()
    return df


# ---------------------------------------------------------------------------
# Held-out confirmation: recency/MRR gap from stratified_activity_results.csv
# rank columns -> reciprocal rank -> MRR
# ---------------------------------------------------------------------------
def holdout_confirmation():
    path = f"{DATA_DIR}/stratified_activity_results.csv"
    if not os.path.exists(path):
        print("[WARN] stratified_activity_results.csv not found")
        return None
    df = pd.read_csv(path)
    active_ranks = df["rank_active_true"].dropna()
    cold_ranks = df["rank_cold_true"].dropna()
    active_mrr = float(np.mean(1.0 / active_ranks))
    cold_mrr = float(np.mean(1.0 / cold_ranks))
    return {
        "active_mrr": active_mrr,
        "cold_mrr": cold_mrr,
        "gap": active_mrr - cold_mrr,
        "n_active": len(active_ranks),
        "n_cold": len(cold_ranks),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = load_audit_signals()
    print(f"Loaded {len(dfs)} datasets: {list(dfs.keys())}")

    candidates = enumerate_candidates()
    print(f"Enumerating {len(candidates)} candidate laws...")

    discovered = []
    for i, cand in enumerate(candidates):
        vr = verify_candidate(cand, dfs)
        law = score_law(cand, vr)
        if law is not None:
            discovered.append(law)

    discovered.sort(key=lambda l: (-l["sign_split_across_types"], -l["avg_r2"]))
    print(f"\nDiscovered {len(discovered)} laws. Top 10:")
    for law in discovered[:10]:
        print(f"  {law['expr']:45s} avg_r2={law['avg_r2']:.3f}  "
              f"sign_split={law['sign_split_across_types']}  n_fit={law['n_datasets_fit']}")

    with open(f"{OUT_DIR}/discovered_laws.json", "w") as f:
        json.dump(discovered, f, indent=2)

    gt = load_ground_truth_signs()
    if gt is not None:
        print("\nGround-truth sign check (cross_benchmark_audit_summary.csv):")
        print(gt[["dataset", "r_fixed", "r_active", "artifact_severity"]].to_string(index=False))
        gt.to_json(f"{OUT_DIR}/ground_truth_signs.json", orient="records", indent=2)

    holdout = holdout_confirmation()
    if holdout:
        print("\nHeld-out confirmation (recency/MRR gap):")
        print(json.dumps(holdout, indent=2))
        with open(f"{OUT_DIR}/holdout_confirmation.json", "w") as f:
            json.dump(holdout, f, indent=2)

    print(f"\nSaved all results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
