"""
Cycle 2, Experiment 1: Cross-benchmark artifact audit.
For each standard TKG benchmark (ICEWS14, ICEWS18, GDELT, WIKI, YAGO), compute
BOTH fixed-node-set and active-node-set susceptibility over time, and report:
  - how strongly fixed-node susceptibility correlates with raw event volume (the artifact)
  - whether that correlation collapses once restricted to active nodes (the fix)
  - an "artifact severity score" = |fixed_corr| - |active_corr| per dataset
This produces a reusable audit table across the whole benchmark family.
"""
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats

RE_NET_DATA = "./data"
DATASETS = ["ICEWS14", "ICEWS18", "GDELT", "WIKI", "YAGO"]

def susceptibility(G):
    if G.number_of_nodes() == 0:
        return 0.0
    comp_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    if len(comp_sizes) <= 1:
        return 0.0
    return sum(s**2 for s in comp_sizes[1:]) / G.number_of_nodes()

def load_dataset(name):
    d = f"{RE_NET_DATA}/{name}"
    cols = ["head", "rel", "tail", "time", "extra"]
    dfs = []
    for split in ["train.txt", "valid.txt", "test.txt"]:
        try:
            dfs.append(pd.read_csv(f"{d}/{split}", sep="\t", names=cols))
        except FileNotFoundError:
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    return df

def build_windows(df, n_windows_target=150):
    times = sorted(df["time"].unique())
    tmin, tmax = times[0], times[-1]
    span = tmax - tmin
    if span <= 0:
        return None, None
    window = span / (n_windows_target / 2)   # overlapping windows, denser stepping
    step = span / n_windows_target
    if window <= 0 or step <= 0:
        return None, None
    return window, step

def run_audit(name):
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    df = load_dataset(name)
    if df is None:
        print("  Data not found, skipping.")
        return None

    all_entities = set(df["head"]) | set(df["tail"])
    print(f"  Total events: {len(df)}, total entities: {len(all_entities)}, unique timestamps: {df['time'].nunique()}")

    window, step = build_windows(df)
    if window is None:
        print("  Could not determine window size (insufficient time range), skipping.")
        return None

    times = sorted(df["time"].unique())
    tmin, tmax = times[0], times[-1]

    records = []
    t = tmin
    while t + window <= tmax:
        win_df = df[(df["time"] >= t) & (df["time"] < t + window)]
        n_events = len(win_df)
        active_entities = set(win_df["head"]) | set(win_df["tail"])

        G_fixed = nx.Graph()
        G_fixed.add_nodes_from(all_entities)
        G_fixed.add_edges_from(zip(win_df["head"], win_df["tail"]))

        G_active = nx.Graph()
        G_active.add_nodes_from(active_entities)
        G_active.add_edges_from(zip(win_df["head"], win_df["tail"]))

        records.append({
            "window_start": t,
            "n_events": n_events,
            "susc_fixed": susceptibility(G_fixed),
            "susc_active": susceptibility(G_active),
        })
        t += step

    sig = pd.DataFrame(records)
    if len(sig) < 20:
        print(f"  Only {len(sig)} windows built, too few for a reliable correlation -- skipping.")
        return None
    print(f"  Built {len(sig)} windows")

    r_fixed, p_fixed = stats.pearsonr(sig["susc_fixed"], sig["n_events"])
    r_active, p_active = stats.pearsonr(sig["susc_active"], sig["n_events"])
    severity = abs(r_fixed) - abs(r_active)

    print(f"  Fixed-node susceptibility  vs n_events: r={r_fixed:+.3f} (p={p_fixed:.4f})")
    print(f"  Active-node susceptibility vs n_events: r={r_active:+.3f} (p={p_active:.4f})")
    print(f"  Artifact severity score (|r_fixed| - |r_active|): {severity:+.3f}")

    sig.to_csv(f"audit_signal_{name.lower()}.csv", index=False)
    return {
        "dataset": name, "n_entities": len(all_entities), "n_events": len(df),
        "n_windows": len(sig), "r_fixed": r_fixed, "p_fixed": p_fixed,
        "r_active": r_active, "p_active": p_active, "artifact_severity": severity,
    }

results = []
for name in DATASETS:
    res = run_audit(name)
    if res:
        results.append(res)

print(f"\n\n{'='*70}\nCROSS-BENCHMARK AUDIT SUMMARY\n{'='*70}")
summary = pd.DataFrame(results)
pd.set_option("display.width", 200)
print(summary.to_string(index=False))
summary.to_csv("cross_benchmark_audit_summary.csv", index=False)
print("\nSaved cross_benchmark_audit_summary.csv")
print("\nHow to read this:")
print("  - Large positive artifact_severity -> that benchmark's fixed-vocabulary structural")
print("    statistics are strongly volume-confounded (like our GDELT global result today).")
print("  - Near-zero severity -> that benchmark doesn't show the artifact much, less concerning.")
print("  - This table itself is a candidate figure/table for a Datasets & Benchmarks paper.")
