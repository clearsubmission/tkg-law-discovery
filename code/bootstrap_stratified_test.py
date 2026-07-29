"""
Bootstrap confidence interval on the active-vs-cold MRR gap, using the
rank results already saved from the stratified test.
"""
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.read_csv("stratified_activity_results.csv")

active_ranks = df["rank_active_true"].dropna().values
cold_ranks = df["rank_cold_true"].dropna().values

print(f"Active group: n={len(active_ranks)}")
print(f"Cold group: n={len(cold_ranks)}")

def mrr(ranks):
    return np.mean(1.0 / ranks)

observed_gap = mrr(active_ranks) - mrr(cold_ranks)
print(f"\nObserved MRR gap (active - cold): {observed_gap:+.4f}")

n_bootstrap = 5000
bootstrap_gaps = []
for _ in range(n_bootstrap):
    boot_active = np.random.choice(active_ranks, size=len(active_ranks), replace=True)
    boot_cold = np.random.choice(cold_ranks, size=len(cold_ranks), replace=True)
    bootstrap_gaps.append(mrr(boot_active) - mrr(boot_cold))

bootstrap_gaps = np.array(bootstrap_gaps)
ci_low, ci_high = np.percentile(bootstrap_gaps, [2.5, 97.5])
prop_positive = np.mean(bootstrap_gaps > 0)

print(f"Bootstrap 95% CI on gap: [{ci_low:+.4f}, {ci_high:+.4f}]")
print(f"Proportion of bootstrap samples with gap > 0: {prop_positive:.4f}")
print(f"\nInterpretation: {'Gap is statistically robust (CI excludes 0)' if ci_low > 0 else 'Gap is NOT statistically robust (CI includes 0) -- treat as suggestive only'}")
