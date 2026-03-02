# FF-GPT: Forward-Forward Trained Transformer for Addition

## Context

Nobody has successfully trained an autoregressive language model with Hinton's Forward-Forward algorithm. This project aims to be the first proof-of-concept, starting with single-digit addition (0+0=0 through 9+9=18). The approach is incremental: prove FF can learn anything, then analyze representations vs backprop.

Based on the user's ChatGPT research notes and our interview, we're building two FF training modes (discriminative + autoregressive), a backprop baseline, and an incremental ablation path.

## File Structure

```
ffgpt/
├── notes/                      # existing research notes
├── requirements.txt            # torch, numpy, matplotlib (pinned versions)
├── ffgpt/
│   ├── __init__.py
│   ├── data.py                 # Vocab, tokenization, dataset classes, negative sampling
│   ├── model.py                # Transformer architecture (FF + baseline variants)
│   ├── goodness.py             # Goodness functions + FF loss functions
│   ├── ff_trainer.py           # FF training loops (discriminative + autoregressive)
│   ├── bp_trainer.py           # Backprop baseline trainer
│   └── utils.py                # Plotting, seeds, logging, checkpointing
├── train_discriminative.py     # Entry: discriminative FF
├── train_autoregressive.py     # Entry: autoregressive FF
├── train_baseline.py           # Entry: backprop baseline
├── evaluate.py                 # Compare all approaches
└── checkpoints/                # Saved model states (gitignored)
```

## Key Design Decisions

### Vocab & Tokenization
- 13 tokens: `{0-9, +, =, <PAD>}`
- Fixed max length = 6 tokens: `"3+5=8<PAD>"` or `"9+9=18"`
- 100 total problems (55 single-digit answers, 45 double-digit)

### Train/Test Split
- **80 train / 20 test**, structured holdout: hold out all pairs where `(a + b) % 5 == 0` (i.e., sums 0, 5, 10, 15) — gives ~20 problems spanning both single-digit and double-digit answers
- This tests generalization to unseen sums, not just memorization
- Backprop baseline should still reach ~100% on train; test accuracy reveals whether the model learns arithmetic structure

### Architecture: Pre-LN Transformer
- 2 blocks, d_model=64, 2 heads, MLP hidden=256, GELU
- MLP exposes intermediate activation (post-GELU, pre-projection) for goodness measurement
- Each block returns `(output, goodness_activations)`
- **Critical:** Block 0 does NOT detach its input (so embeddings get gradients from block 0's loss). Detach between block 0→1, 1→2, etc.

### Two FF Training Modes

**Discriminative:** Model scores complete equations. Positive = `"3+5=8"`, Negative = `"3+5=7"`. Bidirectional attention. Per-block mean goodness → `BCEWithLogitsLoss(goodness - threshold, label)` with **per-block adaptive thresholds**. Inference: score all 19 candidate answers (0-18), pick highest goodness (sum across blocks). **PAD-masked goodness**: only compute mean over non-PAD positions to avoid length bias.

**Autoregressive:** Model predicts next token at positions `t=0..4` (target = token at `t+1`). Position `t=5` has no target. Each block has a local candidate-set CE loss (restricted softmax over correct token + K=12 wrong tokens). Causal attention. Inference: prompt = `[a, +, b, =]`, final-layer hidden states dot **detached** embedding table → greedy decode up to 2 tokens, stop on PAD. Must decode to integer 0-18 or mark incorrect. **Embedding isolation**: the `E` used in `h_k @ E.T` logit computation is `E.detach()` so that only block 0's loss trains the shared embedding table.

### Per-Block Optimizers
- Separate Adam per block (lr=1e-3)
- Separate Adam for embeddings (trained via block 0's loss only)
- Detach between blocks keeps gradients block-local
- **Embedding gradient isolation in autoregressive mode:** `E.detach()` in logit computation prevents gradient leakage from blocks 1+ into E

### Goodness Function
- **Mean goodness** (inspired by DeeperForward, which uses ReLU): `mean(post_GELU_activations)` over non-PAD seq positions and features. POC confirmed GELU goodness is always positive (~0.134 at init vs ReLU ~0.252), safe as goodness function.
- LN-compatible: goodness measured on pre-projection MLP activations, not on the LN-normalized residual stream
- **PAD masking**: goodness only over positions where `token != PAD` — prevents length-dependent bias between `"3+5=8<PAD>"` and `"9+9=18"`
- Discriminative loss: `BCE(sigmoid(goodness - threshold), label)` with **adaptive threshold**
- Autoregressive loss: restricted softmax CE over candidate tokens at each position

### Adaptive Threshold (Discriminative Mode)
- **Do NOT use a fixed threshold.** At init, mean GELU goodness ≈ 0.134 (confirmed by POC); a fixed threshold=2.0 makes sigmoid(g-θ)→0 for ALL examples, giving zero gradient signal.
- **Per-block thresholds**: each block gets its own EMA threshold, since goodness scales differ between blocks
- `θ_k = momentum * θ_k + (1 - momentum) * (mean(g_pos_k) + mean(g_neg_k)) / 2` with momentum=0.9
- Initialize each block's threshold from first batch's actual goodness values for that block
- **Note:** At batch_size=64/pool=80, the EMA converges in ~10 steps and is effectively a fixed auto-initialized threshold. This is fine — the main value is auto-calibration, not ongoing adaptation.
- Use `BCEWithLogitsLoss(g_k - θ_k, label)` — numerically more stable than manual sigmoid + BCE
- Log per-block `g_pos`, `g_neg`, `θ`, and separation every step for diagnostics
- The margin between g_pos and g_neg per block is what matters, not the absolute values

### Metrics Definitions
- **Token accuracy**: fraction of individual output tokens predicted correctly (partial credit)
- **Sequence exact match**: fraction of full equations where every output token is correct (the primary metric)
- **Candidate ranking accuracy** (discriminative only): fraction of problems where the correct answer has highest goodness among all 19 candidates

## Build Order

### Step 1: `data.py` — Dataset generation
- Vocab/tokenization with round-trip tests
- `generate_all_problems()` → list of 100 `(a, b, a+b)` tuples
- `train_test_split(problems, holdout_fn)` → 80 train, 20 test
- `DiscriminativeDataset`: positive/negative equation pairs with configurable negative strategy (random, then near-miss)
- `AutoregressiveDataset`: batches of tokenized sequences
- `sample_negatives(correct_token, vocab_size)`: for autoregressive candidate sets

### Step 2: `model.py` — Transformer architecture
- `MLP` returning `(output, intermediate)` for goodness
- `MultiHeadAttention` with optional causal mask
- `TransformerBlock` returning `(output, goodness_acts)`
- `FFTransformer`: iterates blocks with detach (except block 0 input), returns list of per-block goodness activations
- `BaselineTransformer`: same arch, no detach, has lm_head
- `causal_mask()` helper

### Step 3: `goodness.py` — Loss functions
- `mean_goodness(activations, pad_mask)` → scalar per example, **masked to exclude PAD positions**
- `ff_loss_bce(g_pos, g_neg, threshold)` → discriminative loss
- `update_threshold_ema(g_pos, g_neg, current_threshold, momentum=0.9)` → new threshold (per-block)
- `candidate_set_ce_loss(logits, temperature)` → autoregressive loss

### Step 4: `bp_trainer.py` + `train_baseline.py` — Backprop baseline first
- Standard autoregressive CE with full vocab softmax
- Single Adam optimizer, no detach
- **Validates architecture + data pipeline work correctly**
- Expected: 100% train accuracy in <1000 steps
- Report both train and test accuracy
- Simple argparse: `--lr`, `--steps`, `--seed`, `--checkpoint-dir`

### Step 5: `ff_trainer.py` (discriminative) + `train_discriminative.py`
- Per-block optimizers, mean goodness with PAD masking, BCE loss
- **Adaptive threshold** initialized from first batch
- Random negatives to start (replace answer tokens with random digits)
- Near-miss negatives as second phase (answer ± 1)
- Discriminative inference: score 19 candidates (answers 0-18), pick highest mean goodness
- Also implement logit-based inference (final layer hidden dot embeddings)
- **Diagnostics every N steps:** per-block g_pos mean, g_neg mean, threshold, separation ratio, per-block accuracy
- **Divergence detection:** if `g_pos == g_neg` (separation < epsilon) for 100 consecutive steps, or NaN detected, log warning and optionally stop
- Checkpoint every 1000 steps

### Step 6: `ff_trainer.py` (autoregressive) + `train_autoregressive.py`
- Candidate-set CE at each block
- K=12 (all wrong tokens — full vocab is tiny enough)
- Causal masking
- **`E.detach()` in logit computation** to keep embedding gradients block-local
- Both inference methods: logits + goodness scoring
- Same diagnostics and divergence detection as discriminative
- Checkpoint every 1000 steps

### Step 7: `evaluate.py` + `utils.py`
- Compare all 3 approaches on **both train and test sets** (80 + 20 problems)
- Loss curves, accuracy curves, goodness separation plots
- Per-block diagnostics, confusion matrices
- **Key metric: test accuracy** — distinguishes generalization from memorization
- utils.py: `set_seed()`, `save_checkpoint()`, `load_checkpoint()`, plotting helpers

### Step 8 (v2): Harder negatives + collaboration
- Near-miss negatives (answer ± 1, ± 2) — already available from Step 5
- Two-phase collaborative training (cache goodness → train with global offset)
- KL sync between blocks
- **Model-generated negatives deferred** — requires solving the inference-from-detached-blocks problem, which is a research question in itself

## Critical Technical Details

1. **Embedding gradients:** Block 0 input is NOT detached — embeddings train via block 0's loss. In autoregressive mode, the logit computation uses `E.detach()` so only block 0 trains E. In discriminative mode, E only gets gradients from block 0 (no logit computation needed).

2. **PAD handling:** Source mask = `(tokens != PAD)`. Used in two places: (a) attention masking — don't attend to PAD, (b) goodness masking — don't include PAD positions in mean goodness. DO predict PAD as a target in autoregressive mode (model must learn when answers end).

3. **Per-block adaptive thresholds:** Each block has its own EMA threshold with momentum=0.9, initialized from first batch. Formula: `θ_k = momentum * θ_k + (1 - momentum) * (g_pos_k.mean() + g_neg_k.mean()) / 2`. Log per-block g_pos, g_neg, θ every step.

4. **Temperature:** Start at 1.0 for autoregressive mode. If loss plateaus for 500+ steps, halve it (min 0.1).

5. **Variable-length answers:** "3+5=8\<PAD\>" (5 tokens + pad) vs "9+9=18" (6 tokens). PAD-masked goodness handles the length difference.

6. **Divergence detection:** Check every step: (a) any NaN in loss/goodness → stop, (b) `|g_pos.mean() - g_neg.mean()| < 1e-6` for any block for 100 consecutive steps → warn "goodness collapsed". (c) loss increasing for 500+ steps → warn "diverging".

7. **Checkpointing:** Use `torch.save` with model state_dict, optimizer state_dicts, step, thresholds, temperature, and RNG states (torch + numpy + python). Load with `torch.load(..., weights_only=False)` — this is a local research script, security risk of pickle is irrelevant. Save to `checkpoints/{mode}_step{N}.pt`.

## Starting Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| d_model | 64 | Tiny but real transformer |
| n_heads | 2 | head_dim=32 |
| n_blocks | 2 | Enough to test collaboration |
| mlp_hidden | 256 | 4x expansion |
| lr | 1e-3 | Adam, per-block |
| batch_size | 64 | Out of 80 training examples |
| num_steps | 5000 | Increase if needed |
| threshold | adaptive/per-block | EMA, momentum=0.9, init from batch 1 |
| temperature | 1.0 | For autoregressive CE, halve on plateau |
| K_negatives | 12 | All wrong tokens |
| seed | 42 | Reproducibility |
| checkpoint_every | 1000 | Steps between saves |

## Verification

1. **Backprop baseline reaches 100% train accuracy** — confirms architecture works
2. **Backprop baseline test accuracy** — establishes generalization ceiling
3. **Discriminative FF > 5.3% test accuracy** (random = 1/19) — confirms FF learns something
4. **Autoregressive FF > random test accuracy** — confirms next-token FF works
5. **Per-block goodness separation > 0** — confirms blocks are discriminating (log g_pos, g_neg, threshold)
6. **No goodness collapse** — g_pos and g_neg stay separated throughout training
7. **Block 1 accuracy >= block 0 accuracy** (ideally) — confirms depth helps
8. Run all three training scripts, then `evaluate.py` to compare train vs test accuracy across all approaches

## Ablation Schedule

The build order naturally creates ablations. Each step adds one component; diagnostics at each step measure its contribution.

### Phase 1: Core Comparison (Steps 4-6)
| Run | Mode | Negatives | Blocks | E.detach? | What it tests |
|-----|------|-----------|--------|-----------|---------------|
| A1 | Backprop baseline | N/A | 2 | N/A | Architecture sanity check |
| A2 | FF discriminative | Random | 2 | N/A | Does FF learn anything? |
| A3 | FF autoregressive | K=12 (all wrong) | 2 | Yes | Does AR FF learn anything? |

### Phase 2: Negative Strategy (Step 5, after A2 works)
| Run | Mode | Negatives | What it tests |
|-----|------|-----------|---------------|
| A2 | FF discriminative | Random | Baseline negatives |
| B1 | FF discriminative | Near-miss (±1) | Do harder negatives help? |
| B2 | FF discriminative | Near-miss (±1,±2) | Diminishing returns? |

### Phase 3: Depth (free from per-block diagnostics)
| Metric | What it tests |
|--------|---------------|
| Block 0 accuracy vs Block 1 accuracy | Does depth help in FF? |
| Block 0 goodness separation vs Block 1 | Which block discriminates better? |
| 1-block FF vs 2-block FF (optional run) | Is the second block helping or hurting? |

### Phase 4: Embedding Isolation (Step 6, after A3 works)
| Run | E.detach? | What it tests |
|-----|-----------|---------------|
| A3 | Yes (plan default) | Block-local embeddings |
| C1 | No (all blocks train E) | Does shared embedding training help or hurt? |

### Phase 5: Collaboration (Step 8/v2, if Phase 1 shows promise)
| Run | Collaboration | What it tests |
|-----|--------------|---------------|
| A2/A3 | None (greedy) | Baseline FF |
| D1 | Two-phase with global offset | Does inter-block coordination help? |
| D2 | KL sync between blocks | Alternative coordination method |

**Device:** CPU only (M1 Mac). Each 5000-step run takes ~10-30 seconds. MPS available as `--device mps` flag if we scale up later.

## Stress-Test Findings (Adversarial Review)

### Verified Claims
- **BCEWithLogitsLoss(g - θ, label)**: Correct PyTorch API, numerically stable
- **Train/test split**: Exactly 20 holdout problems (7 single-digit, 13 double-digit)
- **GELU goodness**: Always positive (~0.134 at init), safe drop-in for ReLU. Confirmed by POC.
- **E.detach() pattern**: Supported by SimSiam/BYOL/CLIP/LayerSkip analogues
- **.detach() gradient isolation**: Standard PyTorch behavior, confirmed works

### Issues Found and Fixed
1. **Threshold 2.0 catastrophically wrong** → Fixed: per-block adaptive EMA (auto-init from batch 1)
2. **Embedding gradient leakage in AR mode** → Fixed: `E.detach()` in logit computation
3. **No train/test split** → Fixed: 80/20 structured holdout
4. **Length bias in discriminative goodness** → Fixed: PAD-masked mean goodness
5. **weights_only=True fails with RNG states** → Fixed: use `weights_only=False`
6. **Batch 64/80 makes EMA meaningless** → Accepted: EMA is really just auto-init, which is the main value

### Known Risks (Accepted)
- **Per-block autoregressive CE for text is completely novel** — no prior art in any FF paper (all vision-only). This is the core research contribution.
- **E quality depends on block 0** — if block 0's loss signal is weak, E could drift into a subspace not useful for deeper blocks (SimSiam analogy suggests this works, but not guaranteed)
- **DeeperForward uses ReLU, we use GELU** — departure from paper, but POC confirms GELU works
