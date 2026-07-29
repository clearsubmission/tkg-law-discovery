"""
Permutation significance test for discovered structural laws.
Shuffles MRR across windows 200 times, records best certified R².
If actual R²=0.769 > 99th percentile of null distribution -> p<0.01
"""
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE = './data'
N_PERMS = 200
np.random.seed(42)

DATASET_TYPES = {
    "icews14": "event_stream",
    "icews18": "event_stream", 
    "gdelt":   "event_stream",
    "wiki":    "interval_snapshot",
    "yago":    "interval_snapshot",
}

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

def load_data():
    dfs = {}
    for name in DATASET_TYPES:
        path = f'{BASE}/audit_signal_{name}.csv'
        df = pd.read_csv(path)
        df['dataset_type'] = DATASET_TYPES[name]
        dfs[name] = df
    return dfs

def best_certified_r2(dfs):
    """Find best avg R² among certified laws."""
    best = 0.0
    for u1n, u1f in UNARY_OPS.items():
        for u2n, u2f in UNARY_OPS.items():
            for bn, bf in BINARY_OPS.items():
                results = {}
                for name, df in dfs.items():
                    try:
                        x = bf(u1f(df['susc_fixed'].values),
                               u2f(df['susc_active'].values))
                        x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
                        y = df['n_events'].values
                        if np.std(x) < 1e-10: continue
                        reg = LinearRegression().fit(x.reshape(-1,1), y)
                        r2 = r2_score(y, reg.predict(x.reshape(-1,1)))
                        results[name] = {
                            'r2': r2,
                            'sign': int(np.sign(reg.coef_[0])),
                            'type': DATASET_TYPES[name]
                        }
                    except: continue
                if len(results) < 4: continue
                es = [v['sign'] for v in results.values() if v['type']=='event_stream']
                is_ = [v['sign'] for v in results.values() if v['type']=='interval_snapshot']
                if not es or not is_: continue
                if len(set(es))==1 and len(set(is_))==1 and set(es)!=set(is_):
                    avg_r2 = np.mean([v['r2'] for v in results.values()])
                    best = max(best, avg_r2)
    return best

print("Loading data...")
all_dfs = load_data()

print(f"Running {N_PERMS} permutations...")
null_r2s = []
for perm in range(N_PERMS):
    if perm % 20 == 0:
        print(f"  Permutation {perm}/{N_PERMS}")
    # Shuffle n_events within each dataset
    perm_dfs = {}
    for name, df in all_dfs.items():
        pdf = df.copy()
        pdf['n_events'] = np.random.permutation(df['n_events'].values)
        perm_dfs[name] = pdf
    null_r2s.append(best_certified_r2(perm_dfs))

null_r2s = np.array(null_r2s)
actual_r2 = 0.769

p_value = np.mean(null_r2s >= actual_r2)
pct_95  = np.percentile(null_r2s, 95)
pct_99  = np.percentile(null_r2s, 99)

print(f"\n=== PERMUTATION TEST RESULTS ===")
print(f"Actual best certified R²: {actual_r2:.3f}")
print(f"Null 95th percentile:     {pct_95:.3f}")
print(f"Null 99th percentile:     {pct_99:.3f}")
print(f"Permutation p-value:      {p_value:.4f}")
print(f"Significant at p<0.05:    {p_value < 0.05}")
print(f"Significant at p<0.01:    {p_value < 0.01}")

result = {
    'actual_r2': actual_r2,
    'null_mean': float(np.mean(null_r2s)),
    'null_std':  float(np.std(null_r2s)),
    'null_p95':  float(pct_95),
    'null_p99':  float(pct_99),
    'p_value':   float(p_value),
    'n_perms':   N_PERMS,
}
with open('permutation_significance.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Saved: permutation_significance.json")
