"""
Stratified evaluation by recency (no pool-size confound): full vocabulary
used for ranking in ALL cases. Test triples are split into two groups based
on whether the true answer entity was active in the recent window -- this
isolates whether "activity" predicts prediction difficulty, without ever
shrinking the candidate pool.
"""
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = "./data"
RECENT_WINDOW = 30

cols = ["head", "rel", "tail", "time", "extra"]
train_df = pd.read_csv(f"{DATA_DIR}/train.txt", sep="\t", names=cols)
valid_df = pd.read_csv(f"{DATA_DIR}/valid.txt", sep="\t", names=cols)
test_df = pd.read_csv(f"{DATA_DIR}/test.txt", sep="\t", names=cols)

all_entities = list(set(train_df["head"]) | set(train_df["tail"]) | set(test_df["head"]) | set(test_df["tail"]))
print(f"Total entities: {len(all_entities)}")

history_df = pd.concat([train_df, valid_df], ignore_index=True)
hr_tail_freq = defaultdict(lambda: defaultdict(int))
global_tail_freq = defaultdict(int)
for _, row in history_df.iterrows():
    hr_tail_freq[(row["head"], row["rel"])][row["tail"]] += 1
    global_tail_freq[row["tail"]] += 1

known_triples = set(zip(history_df["head"], history_df["rel"], history_df["tail"])) | \
                set(zip(test_df["head"], test_df["rel"], test_df["tail"]))

def score_candidates(h, r, candidates):
    specific = hr_tail_freq.get((h, r), {})
    return {c: specific.get(c, 0) * 1000 + global_tail_freq.get(c, 0) for c in candidates}

def filtered_rank(h, r, true_t, candidates, known_triples):
    filtered = [c for c in candidates if c == true_t or (h, r, c) not in known_triples]
    scores = score_candidates(h, r, filtered)
    ranked = sorted(filtered, key=lambda c: -scores[c])
    return ranked.index(true_t) + 1

all_times = sorted(history_df["time"].unique())
active_by_time = {}
for t in sorted(test_df["time"].unique()):
    window_df = history_df[(history_df["time"] < t) & (history_df["time"] >= t - RECENT_WINDOW)]
    active_by_time[t] = set(window_df["head"]) | set(window_df["tail"])

SAMPLE_SIZE = len(test_df)  # full test set this time
test_sample = test_df.sample(SAMPLE_SIZE, random_state=42)

results_active_true = []   # true answer WAS recently active
results_cold_true = []     # true answer was NOT recently active

for i, (_, row) in enumerate(test_sample.iterrows()):
    if i % 1000 == 0:
        print(f"  processed {i}/{SAMPLE_SIZE}", flush=True)
    h, r, true_t, t = row["head"], row["rel"], row["tail"], row["time"]

    # ALWAYS rank against the FULL vocabulary -- no pool-size difference between strata
    rank = filtered_rank(h, r, true_t, all_entities, known_triples)

    active_pool = active_by_time.get(t, set())
    if true_t in active_pool:
        results_active_true.append(rank)
    else:
        results_cold_true.append(rank)

def compute_metrics(ranks):
    if len(ranks) == 0:
        return None
    ranks = np.array(ranks)
    return np.mean(1.0/ranks), np.mean(ranks<=1), np.mean(ranks<=3), np.mean(ranks<=10), len(ranks)

print(f"\n{'='*60}\nSTRATIFIED RESULTS (same full vocabulary used throughout, n={len(all_entities)} candidates)\n{'='*60}")

m_active = compute_metrics(results_active_true)
m_cold = compute_metrics(results_cold_true)

if m_active:
    mrr, h1, h3, h10, n = m_active
    print(f"TRUE ANSWER WAS RECENTLY ACTIVE (n={n}, {100*n/SAMPLE_SIZE:.1f}% of test set):")
    print(f"  MRR={mrr:.4f}  Hits@1={h1:.4f}  Hits@3={h3:.4f}  Hits@10={h10:.4f}")

if m_cold:
    mrr, h1, h3, h10, n = m_cold
    print(f"\nTRUE ANSWER WAS COLD/NOT RECENTLY ACTIVE (n={n}, {100*n/SAMPLE_SIZE:.1f}% of test set):")
    print(f"  MRR={mrr:.4f}  Hits@1={h1:.4f}  Hits@3={h3:.4f}  Hits@10={h10:.4f}")

if m_active and m_cold:
    print(f"\nMRR gap (active - cold): {m_active[0] - m_cold[0]:+.4f}")
    print(f"Relative MRR difference: {100*(m_active[0]-m_cold[0])/m_cold[0]:+.1f}%")

pd.DataFrame({"rank_active_true": pd.Series(results_active_true), "rank_cold_true": pd.Series(results_cold_true)}).to_csv("stratified_activity_results.csv", index=False)
print("\nSaved stratified_activity_results.csv")
