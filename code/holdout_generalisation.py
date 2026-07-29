"""
Held-out generalisation test for Paper 2.
Discovers laws on 4 datasets, tests on the 5th (held-out).
Runs all 5 leave-one-out combinations.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import product

BASE = './data'

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
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['dataset_type'] = DATASET_TYPES[name]
            dfs[name] = df
    print(f"Loaded: {list(dfs.keys())}")
    return dfs

def evaluate_program(p_fn, dfs, target='n_events'):
    results = {}
    for name, df in dfs.items():
        try:
            x = p_fn(df['susc_fixed'].values, df['susc_active'].values)
            x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
            y = df[target].values
            if np.std(x) < 1e-10: continue
            reg = LinearRegression().fit(x.reshape(-1,1), y)
            r2 = r2_score(y, reg.predict(x.reshape(-1,1)))
            results[name] = {
                'r2': r2,
                'coef': float(reg.coef_[0]),
                'sign': int(np.sign(reg.coef_[0])),
                'dataset_type': DATASET_TYPES[name]
            }
        except Exception:
            continue
    return results

def check_sign_split(results):
    es_signs = [v['sign'] for v in results.values()
                if v['dataset_type'] == 'event_stream']
    is_signs = [v['sign'] for v in results.values()
                if v['dataset_type'] == 'interval_snapshot']
    if not es_signs or not is_signs: return False
    return (len(set(es_signs)) == 1 and
            len(set(is_signs)) == 1 and
            set(es_signs) != set(is_signs))

def generate_programs():
    progs = []
    feats = ['susc_fixed', 'susc_active']
    # Depth-1
    for f, (un, uf) in product(feats, UNARY_OPS.items()):
        fn = (lambda df_f, df_uf: lambda x, y:
              df_uf(x if df_f == 'susc_fixed' else y))(f, uf)
        progs.append((f"{un}({f})", fn))
    # Depth-2
    for (u1n, u1f), (u2n, u2f), (bn, bf) in product(
            UNARY_OPS.items(), UNARY_OPS.items(), BINARY_OPS.items()):
        expr = f"{bn}({u1n}(susc_fixed), {u2n}(susc_active))"
        fn = (lambda _u1f, _u2f, _bf:
              lambda x, y: _bf(_u1f(x), _u2f(y)))(u1f, u2f, bf)
        progs.append((expr, fn))
    return progs

print("Loading data...")
all_dfs = load_data()
programs = generate_programs()
print(f"Testing {len(programs)} programs")

all_results = {}

for holdout in list(all_dfs.keys()):
    train_dfs = {k: v for k, v in all_dfs.items() if k != holdout}
    test_df   = {holdout: all_dfs[holdout]}

    # Find top certified law on training datasets
    best_law = None
    best_r2  = -1

    for expr, fn in programs:
        train_res = evaluate_program(fn, train_dfs)
        if len(train_res) < len(train_dfs): continue
        if not check_sign_split(train_res): continue
        avg_r2 = np.mean([v['r2'] for v in train_res.values()])
        if avg_r2 > best_r2:
            best_r2   = avg_r2
            best_law  = (expr, fn, train_res, avg_r2)

    if best_law is None:
        print(f"\nHoldout={holdout}: No certified law found on training set")
        continue

    expr, fn, train_res, train_r2 = best_law

    # Evaluate on held-out dataset
    test_res = evaluate_program(fn, test_df)
    holdout_r2   = test_res[holdout]['r2'] if holdout in test_res else float('nan')
    holdout_sign = test_res[holdout]['sign'] if holdout in test_res else 0
    expected_sign = -1 if DATASET_TYPES[holdout] == 'event_stream' else 1
    sign_correct  = (holdout_sign == expected_sign)

    print(f"\nHoldout={holdout} ({DATASET_TYPES[holdout]}):")
    print(f"  Best law: {expr}")
    print(f"  Train avg R²={train_r2:.3f}")
    print(f"  Holdout R²={holdout_r2:.3f}")
    print(f"  Sign correct: {sign_correct} "
          f"(expected={expected_sign}, got={holdout_sign})")

    all_results[holdout] = {
        'best_law': expr,
        'train_avg_r2': round(train_r2, 4),
        'holdout_r2':   round(holdout_r2, 4),
        'sign_correct': sign_correct,
        'expected_sign': expected_sign,
        'actual_sign':   holdout_sign,
        'holdout_type': DATASET_TYPES[holdout],
    }

print("\n=== SUMMARY ===")
sign_correct_count = sum(1 for v in all_results.values() if v['sign_correct'])
print(f"Sign correct: {sign_correct_count}/{len(all_results)} holdouts")
avg_holdout_r2 = np.mean([v['holdout_r2'] for v in all_results.values()
                           if not np.isnan(v['holdout_r2'])])
print(f"Average holdout R²: {avg_holdout_r2:.3f}")

with open('holdout_generalisation.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("Saved: holdout_generalisation.json")
