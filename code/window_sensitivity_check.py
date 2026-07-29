"""
Cycle 2, Experiment 1b: Window-sensitivity check.
Re-run the fixed-vs-active susceptibility artifact test at THREE different
window granularities per dataset (coarse/medium/fine) to check whether the
severity scores from the first pass are robust, or just an artifact of one
arbitrary window choice.
"""
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats

RE_NET_DATA = "./data"
DATASETS = ["ICEWS14", "ICEWS18", "GDELT", "WIKI", "YAGO"]
N_WINDOWS_OPTIONS = [75, 150, 300]  # coarse, medium (~matches first pass), fine

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
    return pd.concat(dfs, ignore_index=True)

def run_at_granularity(df, all_entities, n_windows_target):
    times = sorted(df["time"].unique())
    tmin, tmax = times[0], times[-1]
    span = tmax - tmin
    if span <= 0:
        return None
    window = span / (n_windows_target / 2)
    step = span / n_windows_target
    if window <= 0 or step <= 0:
        return None

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

        records.append({"n_events": n_events, "susc_fixed": susceptibility(G_fixed), "susc_active": susceptibility(G_active)})
        t += step

    sig = pd.DataFrame(records)
    if len(sig) < 15:
        return None
    r_fixed, _ = stats.pearsonr(sig["susc_fixed"], sig["n_events"])
    r_active, _ = stats.pearsonr(sig["susc_active"], sig["n_events"])
    return {"n_windows_target": n_windows_target, "n_windows_actual": len(sig),
            "r_fixed": r_fixed, "r_active": r_active, "severity": abs(r_fixed) - abs(r_active)}

all_results = []
for name in DATASETS:
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    df = load_dataset(name)
    if df is None:
        print("  Not found, skipping.")
        continue
    all_entities = set(df["head"]) | set(df["tail"])

    for n_win in N_WINDOWS_OPTIONS:
        res = run_at_granularity(df, all_entities, n_win)
        if res is None:
            print(f"  n_windows_target={n_win}: could not build enough windows, skipping.")
            continue
        print(f"  n_windows_target={n_win:4d} (actual={res['n_windows_actual']:3d}): "
              f"r_fixed={res['r_fixed']:+.3f}  r_active={res['r_active']:+.3f}  severity={res['severity']:+.3f}")
        res["dataset"] = name
        all_results.append(res)

print(f"\n\n{'='*70}\nSTABILITY SUMMARY\n{'='*70}")
summary = pd.DataFrame(all_results)
pivot = summary.pivot(index="dataset", columns="n_windows_target", values="severity")
print("Severity score by window granularity (rows=dataset, cols=n_windows_target):")
print(pivot.to_string())

print("\nStability check: range of severity across granularities per dataset")
stability = summary.groupby("dataset")["severity"].agg(["min", "max"])
stability["range"] = stability["max"] - stability["min"]
print(stability.to_string())

summary.to_csv("window_sensitivity_summary.csv", index=False)
print("\nSaved window_sensitivity_summary.csv")
print("\nHow to read this:")
print("  - If severity stays similarly signed/magnitude across all 3 granularities -> robust finding, trust the original table.")
print("  - If severity flips sign or swings wildly across granularities -> the artifact score is window-choice-dependent,")
print("    and the original single-window table needs a caveat (report a range, not one number).")
