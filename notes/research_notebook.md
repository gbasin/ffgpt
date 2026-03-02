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

### Entry 9
- Started multi-digit FF refactor (AR first):
  - FF trainers now support variable `sequence_length`, `max_answer_tokens`, `max_answer_value`.
  - Candidate pools are no longer hard-coded to `0..18`; derived from full range when small, otherwise from observed sums.
  - Added run-tag-aware checkpoint metadata for these new settings.
- Next: smoke test single-digit compatibility, then run 2-digit AR and 2-digit discriminative pilots.

### Entry 10
- Fixed a regression in multi-digit refactor:
  - `FFAutoregressiveTrainer` was missing `_target_tokens`, causing AR eval/training calls to fail.
  - `FFDiscriminativeTrainer` had duplicate `_target_tokens` definitions from a bad merge.
- Ran smoke checks on `.venv/bin/python` after fix:
  - discriminative 1-digit (`steps=5`) executes end-to-end and saves checkpoint.
  - autoregressive 1-digit (`steps=5`) executes end-to-end and saves checkpoint.
- Multi-digit pilots are now unblocked.

### Entry 11
- Added scalability instrumentation for multi-digit runs:
  - `train_baseline.py` now supports `--operand-digits`, `--samples`, and dynamic sequence/answer sizing.
  - FF trainers now support `--eval-every`, `--eval-train-max-samples`, `--eval-test-max-samples`.
  - Added `--skip-goodness-eval` for FF runs to avoid expensive candidate scoring when scaling digits.
  - Added optional discriminative rank diagnostics toggle (`--enable-logit-rank-diagnostics`, off by default).
- Switched discriminative default logit inference from candidate enumeration to greedy decoding for speed.
- Added throughput logging (`steps_per_sec`) and eval subset sizes in FF logs.
- Validated 2-digit smoke runs for baseline, FF-discriminative, and FF-autoregressive with the new options.

### Entry 12
- 2-digit scaling sweep (`random` split, 400 steps, eval on held-out set):
  - `n=2000`: baseline test exact `0.215`, FF-discriminative(logits) `0.030`, FF-autoregressive(logits) `0.020`.
  - `n=10000`: baseline test exact `0.205`, FF-discriminative(logits) `0.031`, FF-autoregressive(logits) `0.025`.
- Takeaway: with this budget, FF improves only marginally with more data at 2 digits and remains far below baseline exact-match.

### Entry 13
- 3-digit runs (`n=20000`, random split):
  - 400-step regime is undertrained for everyone:
    - baseline test exact `~0.004`
    - FF-discriminative(logits) test exact `~0.002`
    - FF-autoregressive(logits) test exact `~0.000`
  - 2000-step regime:
    - baseline test exact jumps to `0.266` (clear undertraining confirmation)
    - FF-discriminative(logits) stays near `0.000`
    - FF-autoregressive(logits) stays near `0.004`
- Takeaway: baseline is strongly step-limited and scales with training time; current FF objectives remain near chance on exact-match at 3 digits even when baseline learns.

### Entry 14
- 5-digit run (`n=20000`, random split, 400 steps):
  - baseline, FF-discriminative(logits), and FF-autoregressive(logits) all at `0.000` exact-match.
  - token accuracy non-zero but low (`~0.19-0.28`) indicates partial token learning without full-equation correctness.
- Takeaway: this setting is severely undertrained for exact-match at 5 digits across all methods.

### Entry 15
- Ran FF-AR memorization sanity gates on tiny random splits before changing objective:
  - 1-digit (`n=64`, train=48): reached `train exact=1.000` by 1200-2000 steps.
  - 2-digit (`n=256`, train=192): reached `train exact=1.000` by ~1800-3000 steps.
- Takeaway: FF-AR objective can memorize small datasets; the main issue is scaling/generalization, not total inability to fit.

### Entry 16
- Implemented AR ablation knobs:
  - output projection detach toggle (`--no-detach-output-embedding`)
  - optional per-block LM heads (`--use-per-block-output-heads`)
  - block CE weighting (`--final-block-loss-weight`, `--nonfinal-block-loss-weight`)
- Added checkpoint/eval plumbing for these settings and head states.
- Added candidate-loss instrumentation:
  - weighted candidate loss
  - unweighted candidate loss
  - final-block candidate loss
  - non-final mean candidate loss

### Entry 17
- 3-digit long ablation matrix (`n=20000`, random split, 2000 steps, eval=512):
  - prior AR baseline (detach, no heads): test exact `0.0039`
  - `no_detach`: test exact `0.0020` (worse)
  - `per_block_heads` (equal weights): test exact `0.0059` (small gain)
  - `per_block_heads + final_weight=2.0, nonfinal=0.5`: test exact `0.0059` (same exact as equal weights)
- Baseline backprop at same setup remains much higher (`0.2656` exact).
- Takeaway: per-block heads help slightly, detach toggle does not; gap to baseline remains large.
