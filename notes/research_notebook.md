# Research Notebook

## 2026-03-02

### Entry 1
- Added split diagnostics and found the core issue with `mod5` holdout: test answer tokens `0`/`5` were absent from train answer targets.
- Confirmed this can force near-zero exact match despite high train accuracy.
- Added and validated `coverage` split to preserve answer-token coverage.

### Entry 2
- Implementing three fixes requested:
1. Improve discriminative logit inference.
2. Add autoregressive goodness-separation training signal.
3. Prevent checkpoint collisions across split variants via run tags.
- Added this notebook to track short, periodic experiment notes.

### Entry 3
- Ran baseline with split-aware checkpoint tag (`coverage_s42`):
  - `train exact=1.00`, `test exact=0.20`, `test token=0.55`.
- Confirms tagged checkpoint naming works (`baseline_coverage_s42_step*.pt`).

### Entry 4
- Ran discriminative FF with logit auxiliary loss (`logit_aux_weight=1.0`) on `coverage_s42`.
- Goodness inference improved slightly vs prior (`test exact=0.40`, `test rank=0.40`).
- Logit inference now trains (`train exact=0.9125`) but still fails to generalize (`test exact=0.00`).
- Conclusion: auxiliary objective fixed training signal for logits but not generalization.

### Entry 5
- Ran autoregressive FF with explicit goodness auxiliary BCE (`goodness_aux_weight=1.0`).
- Block separations moved from near-zero/negative to consistently positive by late training:
  - block0 sep ~`+0.0118`, block1 sep ~`+0.0038` at step 2000.
- Logit inference test exact improved slightly to `0.25` on `coverage_s42`.
- Goodness candidate ranking remains weak (`test rank=0.00`), so separation alone is not yet enough for robust goodness inference.

### Entry 6
- Added discriminative logit instrumentation:
  - mean correct rank over 19 candidates
  - per-sum logit accuracy
  - top-k candidate traces saved to JSON
- Diagnostic run (`coverage_s42_la025_dbg`) showed:
  - `train logit exact=0.8125`, `test logit exact=0.10`
  - `test mean_correct_rank=6.0`
  - Only sums `7` and `10` were correct on this 20-example test split.
- This points to selective sum-level generalization failure, not total collapse.

### Entry 7
- Coverage split + run-tagged checkpoints are now default for reproducible comparisons.
- Current reference metrics (coverage split, 80/20):
  - Baseline: test exact `0.20`
  - FF discriminative (goodness): test exact `0.35`
  - FF autoregressive (logits): test exact `0.25`
- Remaining gap: discriminative logit inference generalizes poorly despite training well on train set.

### Entry 8
- Ran quick train-size sweep on coverage split (`n_train = 20/40/60/80`, `steps=800`) for baseline + FF variants.
- Results were noisy and non-monotonic at this small scale/short budget.
- Stronger signal comes from longer runs and larger datasets:
  - baseline (1-digit, coverage split, 2000 steps): test exact `0.20`
  - FF-discriminative goodness (1-digit, coverage split, 2000 steps): test exact `0.35`
  - FF-autoregressive logits (1-digit, coverage split, 2000 steps): test exact `0.25`
  - baseline (4-digit, random split, 10k samples): test exact `~0.85` after 2k steps
  - baseline (5-digit, random split, 12k samples): still low at 2.5k steps (undertrained regime)
- Conclusion: larger-scale baseline clearly improves with enough data/steps; FF scaling to higher digits is not yet tested in current implementation.
