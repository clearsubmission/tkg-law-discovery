"""
Find best laws WITHIN each dataset type separately.
Event-stream: train on 2, test on 3rd.
Interval-snapshot: train on 1, test on 2nd.
"""
import numpy as np, pandas as pd, json, os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    "zscore":   lambda x: (x-np.nanmean(x))/(np.nanstd(x)+1e-8),
    "rank":     lambda x: pd.Series(x).rank(pct=True).values,
}

BINARY_OPS = {
    "add":   lambda a,b: a+b,
    "mul":   lambda a,b: a*b,
    "ratio": lambda a,b: a/(b+1e-8),
    "diff":  lambda a,b: a-b,
}

def load_data():
    dfs = {}
    for name in DATASET_TYPES:
        df = pd.read_csv(f'{BASE}/audit_signal_{name}.csv')
        df['dataset_type'] = DATASET_TYPES[name]
        dfs[name] = df
    return dfs

def fit_r2(fn, df):
    try:
        x = fn(df['susc_fixed'].values, df['susc_active'].values)
        x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
        y = df['n_events'].values
        if np.std(x) < 1e-10: return None, None
        reg = LinearRegression().fit(x.reshape(-1,1), y)
        r2 = r2_score(y, reg.predict(x.reshape(-1,1)))
        return r2, float(reg.coef_[0])
    except: return None, None

def make_programs():
    progs = []
    for u1n,u1f in UNARY_OPS.items():
        for u2n,u2f in UNARY_OPS.items():
            for bn,bf in BINARY_OPS.items():
                expr = f"{bn}({u1n}(suscF),{u2n}(suscA))"
                fn = (lambda _u1f,_u2f,_bf:
                      lambda x,y: _bf(_u1f(x),_u2f(y)))(u1f,u2f,bf)
                progs.append((expr, fn))
    return progs

dfs = load_data()
progs = make_programs()
results = {}

# Event-stream: leave-one-out within type
es_datasets = ['icews14', 'icews18', 'gdelt']
print("\n=== EVENT-STREAM TYPE-SPECIFIC LAWS ===")
for holdout in es_datasets:
    train = {k:v for k,v in dfs.items() if k in es_datasets and k != holdout}
    best_expr, best_r2_train, best_r2_holdout = None, -1, -1
    for expr, fn in progs:
        r2s = []
        for name, df in train.items():
            r2, coef = fit_r2(fn, df)
            if r2 is not None: r2s.append(r2)
        if len(r2s) < 2: continue
        avg = np.mean(r2s)
        if avg > best_r2_train:
            best_r2_train = avg
            best_expr = (expr, fn)
    if best_expr:
        r2h, _ = fit_r2(best_expr[1], dfs[holdout])
        print(f"  Holdout={holdout}: best={best_expr[0][:40]} "
              f"train_R²={best_r2_train:.3f} holdout_R²={r2h:.3f}")
        results[f'es_holdout_{holdout}'] = {
            'law': best_expr[0],
            'train_r2': round(best_r2_train,4),
            'holdout_r2': round(r2h,4) if r2h else None
        }

# Interval-snapshot: train on WIKI, test on YAGO and vice versa
is_datasets = ['wiki', 'yago']
print("\n=== INTERVAL-SNAPSHOT TYPE-SPECIFIC LAWS ===")
for holdout in is_datasets:
    train = {k:v for k,v in dfs.items() if k in is_datasets and k != holdout}
    best_expr, best_r2_train, best_r2_holdout = None, -1, -1
    for expr, fn in progs:
        r2s = []
        for name, df in train.items():
            r2, coef = fit_r2(fn, df)
            if r2 is not None: r2s.append(r2)
        if not r2s: continue
        avg = np.mean(r2s)
        if avg > best_r2_train:
            best_r2_train = avg
            best_expr = (expr, fn)
    if best_expr:
        r2h, _ = fit_r2(best_expr[1], dfs[holdout])
        print(f"  Holdout={holdout}: best={best_expr[0][:40]} "
              f"train_R²={best_r2_train:.3f} holdout_R²={r2h:.3f}")
        results[f'is_holdout_{holdout}'] = {
            'law': best_expr[0],
            'train_r2': round(best_r2_train,4),
            'holdout_r2': round(r2h,4) if r2h else None
        }

with open('type_specific_laws.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: type_specific_laws.json")
