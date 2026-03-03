# FF Literature Review & Next Steps

## Key Papers and What They Do Differently

### 1. Hinton's Original FF (2212.13345)
- **Peer normalization (L2)** between layers is critical — strips goodness magnitude, forces next layer to use only direction of activations
- Tested only on MLPs (MNIST), not transformers
- Negative generation via label overlay on input pixels

### 2. DeeperForward (ICLR 2025)
- **Mean goodness** instead of sum-of-squares — our codebase already does this
- **LayerNorm between blocks** is the core enabler for depth scaling — with BatchNorm, goodness info leaks through batch statistics and deeper layers degrade. LayerNorm avoids this
- Scales FF to 17-layer CNNs (prior work broke at 4-6 layers)
- Still CNN-only, no transformers

### 3. Contrastive Forward-Forward for ViT (2502.00571) — Most directly relevant
- Applied FF to Vision Transformers
- **Replaced BCE goodness entirely with supervised contrastive loss** — cosine similarity between same-class representations = positive, different-class = negative
- Each block output is average-pooled across sequence positions for loss, but full representation passes forward (detached)
- **Marginal contrastive loss**: adds increasing margin per layer depth, encouraging progressive refinement rather than forcing max similarity at every layer
- Results: **+10% accuracy and 5-20x faster convergence** over baseline FF on ViT
- CIFAR-10 ViT: 76.21% (baseline FF) → 80.42% (contrastive FF)

### 4. Scalable Forward-Forward / SFF (2501.03176) — Strongest results
- **Hybrid approach**: backprop within each block, FF (no backprop) between blocks
- Uses log-sum-exp loss: `L = -E[g_pos - log(sum exp(g_class_j))]` — no explicit negatives needed, all classes compete via softmax
- **SFF outperforms full backprop** on CIFAR-10/100 and ImageNet subsets (81.38% vs 78.49% on CIFAR-10 CNN)
- Key: allowing attention + MLP within a block to coordinate via backprop while keeping inter-block learning local

### 5. Layer Collaboration in FF (AAAI 2024)
- Each layer's update incorporates information from later layers via a **unified loss** — not just an offset/cache
- Individual layers perform *worse* under unified loss, but collectively outperform per-layer training
- Theoretical grounding in functional entropy maximization
- Tested on MLPs, not transformers

### 6. Self-Contrastive FF / SCFF (Nature Communications 2025)
- Negative generation via concatenating matching vs mismatching input pairs — no augmentation needed
- First to show FF working on **RNNs for audio data** (90.3% on FSDD)
- Uses per-image standardization + "triangle method" channel competition between layers

### 7. SymBa (2303.08418)
- Identifies **gradient asymmetry** as a core FF training problem — positive and negative passes produce different gradient magnitudes, causing conflicting optimization
- Symmetric loss reformulation fixes convergence instability
- MLP-only

## Core Insight From the Literature

Every paper that successfully scales FF beyond toy problems does at least one of:

1. **Replace BCE goodness with contrastive or CE loss** (CFF-ViT, SFF, ASGE)
2. **Allow some gradient flow between layers** — hybrid block-wise backprop (SFF)
3. **Explicit inter-layer collaboration** via unified losses (AAAI 2024) or marginal contrastive loss (CFF-ViT)

Our current setup uses all three of the weakest options simultaneously: fully detached blocks, BCE goodness as primary signal, and minimal inter-block communication. For multi-digit addition where carry propagation requires inter-position and inter-block coordination, this is a fundamental mismatch.

## Proposed Experiments (Ranked by Expected Impact)

### E1. Hybrid block-wise training (from SFF)
Allow backprop within pairs of adjacent blocks but detach between pairs. E.g., with 4 blocks: backprop within `[block0,block1]` and `[block2,block3]`, detach between the pairs. This is the minimal change that enables carry coordination (one block detects carry, the adjacent one applies it). SFF shows this can actually *outperform* full backprop.

### E2. Per-block CE as primary loss instead of auxiliary
We already have `candidate_set_ce_loss` and `logit_aux_weight`. The SFF paper validates that making CE/softmax-over-classes the primary signal (not an aux) is more effective than goodness BCE. Try making per-block CE the only loss with `goodness_aux_weight=0.0` and higher `logit_aux_weight`.

### E3. Contrastive loss per block (from CFF-ViT)
Replace BCE goodness with supervised contrastive loss: L2-normalize each block's output, compute pairwise cosine similarities within the batch, same-answer pairs = positive, different-answer = negative. The marginal contrastive variant (increasing margin per layer) encourages progressive refinement.

### E4. Answer-only CE weighting
Concentrate CE on positions after `=` (answer tokens) rather than averaging over the full sequence. This focuses gradient budget on carry-sensitive digits instead of easy structural tokens (`+`, `=`, input copying).

### E5. Fix the inference bug
`predict_with_logits` uses `embedding_weight.detach()` directly instead of `_project_block_logits`, so trained per-block output heads are never used during eval. This invalidates the `use_per_block_output_heads` ablation conclusions.

### E6. Unified loss (from AAAI 2024)
Instead of collaborative offset (adding cached goodness from other blocks), compute a weighted sum of all blocks' losses and backprop each block's contribution with detached terms from other blocks. Fundamentally different from the current offset approach — it changes the loss landscape, not just the threshold.

## References
- Hinton - The Forward-Forward Algorithm (arXiv 2212.13345)
- DeeperForward (ICLR 2025, OpenReview kOYnXVQCtA)
- Contrastive Forward-Forward for ViT (arXiv 2502.00571)
- Scalable Forward-Forward (arXiv 2501.03176)
- Layer Collaboration in FF (AAAI 2024)
- Self-Contrastive FF (Nature Communications 2025, arXiv 2409.11593)
- SymBa (arXiv 2303.08418)
