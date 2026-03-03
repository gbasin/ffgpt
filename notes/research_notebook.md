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

### Entry 18
- Added `train_length_ood.py` for clean digit-length OOD experiments (train on digit length `d`, test on `d+1`).
- First OOD benchmark (`2-digit train -> 3-digit test`, train=10000 exhaustive, test=4000 random, 2000 steps):
  - baseline: train exact `0.9167`, test exact `0.0092`
  - FF-AR default: train exact `0.0898`, test exact `0.0020`
  - FF-AR heads+weighted (`final=2.0`, `nonfinal=0.5`): train exact `0.0996`, test exact `0.0020`
- Important caveat surfaced by instrumentation:
  - test answer position 3 tokens (`0..9`) are completely missing in train targets for `2->3` (new answer digit position).
  - This is a very hard extrapolation regime; low exact-match for all models is expected.

### Entry 19
- Deeper train-behavior debug on `2->3` OOD run (FF-AR default) to test “is FF random on train?”:
  - FF-AR train token accuracy is not random (`~0.666`), but train exact remains low (`~0.089`).
  - Position-wise train answer-token accuracy:
    - pos0 `~0.864`
    - pos1 `~0.311`  <- main failure
    - pos2 `~0.542`
    - pos3 `~0.999` (PAD)
  - Baseline on same setup is much stronger (e.g., pos1 `~0.968`, train exact `~0.917`).
- Interpretation: FF-AR is not random; it underfits structured digit generation (especially the second answer digit), suggesting an optimization/objective mismatch rather than pure data-pipeline failure.

### Entry 20
- Targeted ablation: remove goodness auxiliary pressure during FF-AR training (`goodness_aux_weight=0.0`) in `2->3` OOD setup:
  - FF-AR default (`aux=1.0`): train exact `0.0898`, test exact `0.0020`
  - FF-AR no-aux (`aux=0.0`): train exact `0.1055`, test exact `0.0039`
  - FF-AR no-aux + heads+weighted: train exact `0.0859`, test exact `0.0020`
- Takeaway: goodness auxiliary appears to interfere with token-learning in this regime; removing it helps somewhat, but FF remains far below baseline.

### Entry 21
- Added `analyze_length_ood_checkpoint.py` to measure **greedy decode vs teacher-forced** metrics on the same checkpoint and dataset.
- Key systematic finding on `2->3` OOD (eval subset 512):
  - FF default: greedy train exact `0.0918`, teacher-forced train exact `0.0918`
  - FF no-aux: greedy train exact `0.0996`, teacher-forced train exact `0.0996`
  - FF no-aux final-only CE: greedy train exact `0.0488`, teacher-forced train exact `0.0488`
- Interpretation: decode strategy is **not** the primary bottleneck here; teacher-forced and greedy exact are nearly identical for the best FF runs.
- Additional ablation:
  - final-block-only CE (`nonfinal=0`) under no-aux underperformed (train exact `0.041-0.049`, test exact `~0.000-0.002`).
  - This suggests some non-final supervision is still useful in current FF setup.

## 2026-03-03

### Entry 22
- Implemented a new FF-discriminative ablation mode: **layerwise single-block training**.
  - New trainer flags:
    - `layerwise_train_single_block`
    - `layerwise_phase_steps`
  - In this mode, only one block is updated per phase (`block_i` active in phase `i`), with earlier blocks frozen.
  - Logit aux loss is enabled only when the final block is active.
  - Checkpoints now save layerwise metadata for reproducibility.
- Added CLI support in `train_discriminative.py`:
  - `--layerwise-train-single-block`
  - `--layerwise-phase-steps` (comma-separated, must match `n_blocks` and sum to `--steps`)

### Entry 23
- Ran comparison on 1-digit `coverage_s42` (same split as prior references), `steps=2000`.
  - Joint FF-disc (prior reference checkpoint):
    - goodness test exact `0.40`
    - block diagnostics on test:
      - block0 acc `0.40`, block1 acc `0.20`
      - rescue `0->1`: `0.25` (12 cases)
      - degrade `0->1`: `0.875` (8 cases)
  - Layerwise FF-disc (`phase_steps=1000,1000`):
    - goodness test exact `0.45`
    - block diagnostics on test:
      - block0 acc `0.50`, block1 acc `0.10`
      - combined acc `0.45` (still below block0-only)
      - rescue `0->1`: `0.10` (10 cases)
      - degrade `0->1`: `0.90` (10 cases)
- Interpretation:
  - Layerwise training improved final combined goodness accuracy slightly (`0.40 -> 0.45`), but it did **not** produce healthy specialization.
  - Block1 remains mostly harmful as a fixer (high degrade, low rescue), indicating the core objective/aggregation mismatch remains.

### Entry 24
- Tested “attenuate block contribution when not confident” and “10x data” hypotheses.
- Confidence attenuation probe (3-digit FF-disc checkpoints, candidate-goodness diagnostics on 128-sample subsets):
  - `n=20k` (`d3_rnd_n20000_s42_long`): goodness combined accuracy on test subset was ~`0.008`; block0 and block1 each near `0.0`.
  - `n=200k` (`d3_rnd_n200000_s42_long`, 10x dataset size): goodness combined/test remained `0.0`.
  - Margin-gated use of block1 (`tau` sweep on block1 top-2 margin) did not recover accuracy in this regime.
- Strict “10x data seen” probe (2-digit, same dataset, 10x training steps):
  - baseline reference checkpoint (`n=10k`, `steps=400`): eval-subset test logit exact `0.0352`.
  - new run (`n=10k`, `steps=4000`, tag `d2_rnd_n10000_s42_step4000`):
    - final reported full-test logit exact `0.0391`
    - eval-subset test logit exact remained `0.0352`
    - train logit exact increased modestly (`~0.035 -> ~0.055-0.066` depending on full-vs-subset metric).
- Takeaway:
  - More data diversity (10x `3-digit` samples) did not improve FF-disc at this budget.
  - More exposures (10x steps on 2-digit) mostly increased train fit with negligible test gains.
  - This is consistent with objective/credit-assignment limitations rather than simple undertraining.

### Entry 25
- Implemented requested ablation stack for FF-discriminative:
  1. **Inter-block decoupling/normalization** (`inter_block_norm`: `none|layernorm|rmsnorm|l2`, with `inter_block_norm_eps`) applied between blocks in `FFTransformer.forward`.
  2. **Per-block predictive supervision** via optional per-block logit aux CE (`use_per_block_logit_aux`) with final/non-final block weighting knobs.
  3. **Non-equal block aggregation at inference** through `goodness_aggregation=weighted_sum`, optional manual `goodness_block_weights`, and optional eval-time fitting (`fit_goodness_block_weights`) on train-eval subset.
- Added checkpoint/eval plumbing so these settings serialize and reload correctly.
- Smoke-tested combined path (`layernorm + per-block aux + weighted_sum + fit weights`) with `steps=20` on `coverage_s42`; run executed end-to-end and produced checkpoints.

### Entry 26
- Ran a thorough retest matrix on `1-digit coverage_s42` (`steps=2000`, same split/seed) for the new FF-disc knobs:
  - **Default retest** (`coverage_s42_retest_default`):
    - goodness test exact `0.40`
    - logit test exact `0.00`
    - test block acc: block0 `0.40`, block1 `0.20`
    - rescue `0->1`: `0.25` (den=12), degrade `0->1`: `0.875` (den=8)
  - **Inter-block norm only** (`inter_block_norm=layernorm`):
    - goodness test exact `0.40` (no gain)
    - logit test exact `0.00`
  - **Per-block aux only** (`use_per_block_logit_aux`):
    - goodness test exact `0.05` (major drop)
    - logit test exact `0.10` (improved)
  - **Weighted goodness aggregation + fitted weights**:
    - learned weights converged to roughly `[0.25, 0.0]` (effectively block0-only)
    - goodness test exact `0.40` (neutral)
    - logit test exact `0.00`
  - **Combined** (`layernorm + per-block aux + weighted/fitted`):
    - goodness test exact `0.05` (still low)
    - logit test exact `0.15` (best in this matrix)
- Interpretation on this split:
  - New knobs are useful for **logit inference** improvements.
  - They hurt or do not improve **goodness inference** in current form, so they do not solve the block-collaboration issue for goodness scoring yet.

### Entry 27
- Added 2-digit sanity retest (`operand_digits=2`, `n=10000`, random split, `steps=400`, skip-goodness-eval):
  - default retest (`d2_rnd_n10000_retest_default_post123`): test logit exact `0.0039`
  - combined (`layernorm + per-block aux`): test logit exact `0.0195`
- So in this short-budget 2-digit regime, combined settings improved logit test exact by ~5x relative (still low in absolute terms).

### Entry 28
- Parameter-sanity check focused on FF-disc aux scale:
  - Original per-block-aux settings (`logit_aux_weight=1.0`, nonfinal=1.0) caused strong goodness collapse on `coverage_s42`:
    - perblockaux: goodness test exact `0.05`, logit test exact `0.10`
    - combined: goodness test exact `0.05`, logit test exact `0.15`
- Tuned lower aux scale (`logit_aux_weight=0.25`, `nonfinal_block_logit_aux_weight=0.25`, final=1.0):
  - `coverage_s42_retest_perblockaux_tuned025`:
    - goodness test exact `0.20`
    - logit test exact `0.20`
  - `coverage_s42_retest_combined_tuned025`:
    - goodness test exact `0.15`
    - logit test exact `0.15`
- Interpretation:
  - We are **not** yet in a globally “sane by default” regime for the new knobs.
  - Aux scale is a critical stability parameter; high defaults over-prioritize predictive aux and can destroy goodness ranking.
  - Lower aux improves balance and avoids catastrophic collapse, but still does not beat the original goodness baseline (`0.40`) on this split.

### Entry 29
- Added reproducible sweep tool: `sweep_ffdisc_aux_grid.py`.
  - Coarse stage: modes `{perblockaux, combined}`, grid over `aux_weight x nonfinal_aux_weight`, seeds `{42,43}`, steps `1000`.
  - Refine stage: top-4 coarse configs, seeds `{42,43,44}`, steps `2000`.
  - Outputs saved to `checkpoints/sweeps/coverage_auxgrid_20260303_085152.{json,csv}`.
- Refine top results (mean over seeds 42/43/44):
  - **combined, aux=0.05, nonfinal=0.25**:
    - test goodness exact `0.5833`
    - test logit exact `0.2167`
    - balanced score `0.4000`
  - **perblockaux, aux=0.05, nonfinal=0.25**:
    - test goodness exact `0.5500`
    - test logit exact `0.2333`
    - balanced score `0.3917`
  - **perblockaux, aux=0.1, nonfinal=0.1**:
    - test goodness exact `0.4333`
    - test logit exact `0.2833`
    - balanced score `0.3583`
- Multi-seed default baseline (same setup, seeds 42/43/44):
  - mean test goodness exact `0.3333`
  - mean test logit exact `0.1333`
- Takeaway:
  - Earlier ad hoc tuned point (`aux=0.25, nonfinal=0.25`) was not near peak.
  - Best region is lower aux scale (`~0.05-0.1`) with moderate nonfinal weight (`~0.1-0.25`).

### Entry 30
- Implemented Step-8 collaboration variants in `FFDiscriminativeTrainer`:
  - **D1:** two-phase collaborative goodness offset with stop-grad cached cross-block terms (`collaborative_global_offset_weight`).
  - **D2:** KL sync from final block to non-final blocks on answer positions (`kl_sync_weight`).
- Added CLI/eval/checkpoint plumbing:
  - `train_discriminative.py`: `--collaborative-global-offset-weight`, `--kl-sync-weight`.
  - `evaluate.py` now reloads these checkpoint fields.
- Added instrumentation:
  - train logs now include `kl_sync=...`.
  - per-block diagnostics include collaborative offsets (`co+`, `co-`), and history stores per-block collab offsets + KL-sync loss.

### Entry 31
- Added `sweep_ffdisc_collab_grid.py` for systematic D1/D2 sweeps (coarse+refine, multi-seed), fixed base aux at the current best combined setting:
  - `mode=combined`, `aux=0.05`, `nonfinal_aux=0.25`, `final_aux=1.0`.
- Full run:
  - `coverage_collab_20260303_095101.json` (+ CSV summaries), coarse seeds `{42,43}`, refine seeds `{42,43,44}`.
- Main result: **best refined config is still no collaboration terms**:
  - `collab=0.0`, `kl=0.0`: test goodness exact `0.5833`, test logit exact `0.2167`, balanced `0.4000`.
- Best non-zero setting:
  - `collab=0.1`, `kl=0.0`: test goodness exact `0.4833`, test logit exact `0.2500`, balanced `0.3667`.
- KL sync settings were consistently harmful to goodness in this regime:
  - e.g. `collab=0.0`, `kl=0.05`: goodness `0.3000`, logit `0.1667`, balanced `0.2333`.
  - e.g. `collab=0.0`, `kl=0.2`: goodness `0.0833`, logit `0.1167`, balanced `0.1000`.

### Entry 32
- Explicit D1+D2 probe to avoid missing interactions:
  - `collab=0.2`, `kl=0.05`, seeds `{42,43,44}`, `steps=2000`.
  - Output: `coverage_collab_probe_c02k005_20260303_102209.json`.
- Result:
  - mean test goodness exact `0.3833`
  - mean test logit exact `0.1167`
  - balanced `0.2500`
- Conclusion: at current architecture/objective scale, D1+D2 does **not** outperform the no-collaboration best combined configuration.
