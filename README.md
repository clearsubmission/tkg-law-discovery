# Discovering Structural Laws in Temporal Knowledge Graphs via Verified Program Search

Anonymous code release for AAAI 2027 submission.

## Contents

- code/law_search.py - Verified program search engine (Algorithm 1): 156-program DSL, per-dataset OLS, sign-split certification (Sec. 3-4).
- code/permutation_significance.py - Permutation null test, 200 shuffles (Sec. 3.3).
- code/law_search_differenced.py - Differenced-law search (Sec. 6).
- code/stationarity_check.py - ADF stationarity tests (Sec. 6, supp. Tables S2-S4).
- code/holdout_generalisation.py - Leave-one-out holdout, both protocols (Sec. 4.4, Table 2).
- code/pysr_law_search.py - PySR symbolic regression comparison (Sec. 5, Table 4).
- code/baseline_comparison.py - Naive linear baseline (Table 4).
- code/type_specific_laws.py - Type-specific laws for within-type holdout.
- code/stratified_activity_test.py - Query-level confirmation on ICEWS18, Mann-Whitney U (Table 3). Needs raw ICEWS18 from the RE-Net repo under data/ICEWS18/.
- code/bootstrap_stratified_test.py - Bootstrap 95% CI, 10,000 resamples, seed 42.
- code/cross_benchmark_audit.py - Generates the window-level audit CSVs from raw benchmarks.
- code/window_sensitivity_check.py - Robustness across W in {74, 149, 299}.
- code/ground_truth_signs.json - Per-dataset audit statistics.
- data/audit_signal_*.csv - Window-level measurements (149 windows: susc_fixed, susc_active, n_events).
- data/cross_benchmark_audit_summary.csv - Cross-benchmark summary.

## Requirements

Python 3.12, scikit-learn 1.9.0, NumPy 2.4.6, SciPy 1.18.0, PySR 1.5.10, statsmodels.
## Reproducing paper results

    python code/law_search.py
    python code/permutation_significance.py
    python code/law_search_differenced.py
    python code/stationarity_check.py
    python code/holdout_generalisation.py
    python code/pysr_law_search.py

All seeds fixed at 42. Window-level inputs are included, so main results reproduce from this repository alone.

## Data provenance

CSVs derived from public benchmarks: ICEWS14/18 (Ward et al.), GDELT (Leetaru and Schrodt), WIKI/YAGO (Leblay and Chekol). cross_benchmark_audit.py regenerates them from the raw releases.
