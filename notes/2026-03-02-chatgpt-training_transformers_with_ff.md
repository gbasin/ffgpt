# ChatGPT Conversation: Training transformers with FF

Source: https://chatgpt.com/share/69a5dec1-de08-8013-8a1f-22944ea5d3d7
Retrieved: 2026-03-02T19:03:32.770Z

## User

Figure out the best way to train a transformer using hintons forward forward so it maintains its global characteristics without falling back to the highly tangled representations encouraged by backprop

## Assistant

## What breaks when you naively port Forward–Forward to Transformers

Hinton’s Forward–Forward (FF) replaces end-to-end backprop with _layer-local_ objectives trained from two forward passes: a **positive** pass (real/consistent input) and a **negative** pass (corrupted/inconsistent input). Each layer learns to yield **high goodness** on positives and **low goodness** on negatives.

For Transformers, the main failure modes are:

1.  **Inter-layer isolation kills “global” behavior**
    Plain FF trains each layer “as if it were the whole model,” so early layers don’t get pressured to produce features that later layers can actually use. This lack of collaboration has been analyzed explicitly, and a simple modification was proposed to inject global information into each layer’s local objective.
    
2.  **Normalization collides with classic goodness choices**
    If goodness is something like “sum of squared activations,” layer normalization can wipe out the signal (or make it trivial/constant), so you must compute goodness at the right point or redesign it. Hinton notes this directly.
    There are newer “LN-compatible” goodness designs (e.g., mean-goodness + layer norm) intended to make deep FF training stable.
    
3.  **Negatives must be _globally_ hard, not just locally noisy**
    If negatives are too easy (random garbage), layers learn shortcuts; if they’re too different from data, you don’t learn data structure. Work on self-contrastive FF emphasizes that the _quality/structure of negatives_ is a key limiter and proposes systematic generation from the data distribution.
    
4.  **If you “finish” with backprop, you often undo the point**
    If you FF-pretrain then do full end-to-end BP fine-tuning, you’ll typically reintroduce the very representation entanglement you’re trying to avoid. One reason FF is interesting here is that FF objectives can yield sparse, ensemble-like internal structure that’s not typically seen with standard BP training.
    

* * *

## A strong recipe: Collaborative + LN-compatible + globally-structured negatives

The most defensible “best way” (based on what’s worked in the literature so far) is to combine four ideas:

1.  **Blockwise FF (treat each Transformer block as a “layer”)**
    
2.  **Collaborative FF (inject global goodness into each block’s local objective)**
    
3.  **LN-compatible goodness (so Transformers don’t trivialize goodness)**
    
4.  **Negatives that specifically break _global coherence_ while preserving local plausibility**
    

A practical instantiation is:

### 1) Architecture constraints that make FF work in a Transformer

**Use a Pre-LN Transformer** (standard in modern LLMs anyway), but be explicit about _where_ you measure goodness.

For block lll, you have something like:

-   ul\=LN(xl)u\_l = \\mathrm{LN}(x\_l)ul​\=LN(xl​)
    
-   al\=Attnl(ul)a\_l = \\mathrm{Attn}\_l(u\_l)al​\=Attnl​(ul​), xl′\=xl+alx'\_l = x\_l + a\_lxl′​\=xl​+al​
    
-   vl\=LN(xl′)v\_l = \\mathrm{LN}(x'\_l)vl​\=LN(xl′​)
    
-   ml\=MLPl(vl)m\_l = \\mathrm{MLP}\_l(v\_l)ml​\=MLPl​(vl​), xl+1\=xl′+mlx\_{l+1} = x'\_l + m\_lxl+1​\=xl′​+ml​
    

**Key FF requirement:** goodness must be computed on an activation that is not made invariant/constant by LN.

Two robust options:

-   **Option A (Hinton-style):** compute goodness on _pre-LN_ hidden activities (he explicitly defines goodness using activities “before layer normalization”).
    
-   **Option B (DeeperForward-style):** use **mean goodness** while using LN for the representation stream (decoupling “goodness” from “features”); this is explicitly motivated as fixing scaling + deactivated neuron issues.
    

For Transformers, Option B is usually easier:

-   Let HlH\_lHl​ be a block-local activation tensor you choose (common: the MLP pre-activation after a ReLU/GELU or an equivalent “post-nonlinearity” state).
    
-   Define **goodness** as a mean over features (and optionally tokens):
    
    gl\=mean(Hl)g\_l = \\mathrm{mean}(H\_l)gl​\=mean(Hl​)
-   Define the **representation passed upward** as the LN-normalized stream (standard Transformer residual stream).
    

This mirrors the “goodness is mean; features are LN outputs” separation.

### 2) Positive/negative construction that forces global structure

To preserve “global characteristics,” your negatives must break _global_ constraints while keeping _local_ statistics plausible.

Use a **mixture of negative types**:

-   **Local-preserving, global-breaking (the important ones):**
    
    -   swap two long spans (A … B → B … A)
        
    -   replace a middle span with one from another sample (in-batch “span transplant”)
        
    -   sentence-order shuffles (for multi-sentence text)
        
    -   entity/attribute swaps (replace all mentions of one entity with another from batch)
        
-   **Light local corruption (for robustness, not for global structure):**
    
    -   small token masking/dropout
        
    -   small token substitutions sampled from unigram or same-POS bucket
        

If you want a simple default that avoids bespoke negative generators, borrow the **self-contrastive** pattern: positives/negatives formed directly from the batch rather than a separate negative model.
(You will still want the _span-level_ variants above if your explicit goal is “global characteristics.”)

### 3) Layer-local losses that don’t collapse early layers

A plain FF layer loss is “classify positive vs negative from goodness,” using a logistic mapping and a threshold.

For Transformers, you will usually get better behavior if you add a **contrastive term** at each block in addition to (or replacing) the thresholded goodness classification—this is exactly the direction taken by work extending FF to Vision Transformers using contrastive-inspired losses and a depth-dependent margin to avoid early-layer collapse.

So for each block lll, compute a pooled representation rlr\_lrl​ (e.g., CLS token or mean pooling over tokens). Then use:

**(a) FF goodness classification (local):**

-   gl+g\_l^{+}gl+​ from positive sample, gl−g\_l^{-}gl−​ from negative
    
-   logistic probability p\=σ(gl−θ)p = \\sigma(g\_l - \\theta)p\=σ(gl​−θ)
    
-   loss:
    
    LlFF\=BCE(p+,1)+BCE(p−,0)\\mathcal{L}^{FF}\_l = \\mathrm{BCE}(p^{+}, 1) + \\mathrm{BCE}(p^{-}, 0)LlFF​\=BCE(p+,1)+BCE(p−,0)

**(b) Contrastive alignment term (optional but recommended):**

-   Use InfoNCE or supervised-contrastive if you have labels
    
-   If you want to prevent early layers from over-clustering (a known problem), use a **layer-dependent margin** (a “marginal contrastive” idea) so early blocks only learn _some_ similarity, later blocks enforce tighter grouping.
    

Total:

Ll\=λffLlFF+λctrLlCTR\\mathcal{L}\_l = \\lambda\_{ff}\\mathcal{L}^{FF}\_l + \\lambda\_{ctr}\\mathcal{L}^{CTR}\_lLl​\=λff​LlFF​+λctr​LlCTR​

### 4) Add collaboration without backprop: “global goodness injection”

This is the piece that most directly targets your “maintains global characteristics” requirement.

Layer-collaboration work proposes modifying each layer’s local probability by adding a **stop-gradient global term** based on other layers’ goodness, so each layer optimizes its parameters with awareness of the network-wide state, but still _without_ end-to-end backprop.

Concretely:

-   Compute goodness values g1,…,gLg\_1,\\dots,g\_Lg1​,…,gL​ for the current sample (positive or negative).
    
-   For layer lll, define:
    
    γl\=∑t≠lstopgrad(gt)\\gamma\_l = \\sum\_{t\\neq l} \\mathrm{stopgrad}(g\_t)γl​\=t\=l∑​stopgrad(gt​)
-   Use:
    
    pl\=σ(gl+αγl−θ)p\_l = \\sigma(g\_l + \\alpha\\gamma\_l - \\theta)pl​\=σ(gl​+αγl​−θ)
-   Then do BCE on plp\_lpl​ vs {positive=1, negative=0}.
    

This gives you a mechanism for global coordination while keeping updates local in the FF sense.

* * *

## Preventing “backprop-like entanglement” inside FF training

Even with FF, a Transformer can learn overly distributed, hard-to-parse features unless you bias it toward modularity. Two pragmatic additions:

### A) Head/feature-group specialization pressure

A channel-wise competitive learning variant of FF explicitly aims to improve feature partitioning and compositionality (developed for CNNs, but the principle maps cleanly to Transformer “groups” such as heads or MLP channel blocks).

Transformer adaptation (direct, effective):

-   Partition the hidden dimension into GGG groups (or treat attention heads as groups).
    
-   Compute goodness per-group gl,gg\_{l,g}gl,g​.
    
-   Add a mild regularizer that encourages _few_ groups to be strongly active per sample (k-winners / sparsemax-like behavior), which tends to reduce entanglement.
    

### B) Keep FF “pure”: don’t do end-to-end BP fine-tuning

If you BP-fine-tune all blocks, you’ll typically lose the sparse-ensemble structure that FF objectives can produce. Evidence that FF-trained networks can develop sparse, category-specific ensembles (and that this sparsity pattern differs from typical BP training) supports keeping the training regime consistent if interpretability/modularity is a goal.

If you must add a supervised head:

-   Train **only** a small readout / classifier head with BP and freeze the FF-trained backbone (or use tiny adapters), so you don’t re-entangle the internal representation.
    

* * *

## Minimal pseudo-algorithm (blockwise, collaborative, globally-hard negatives)

For batch XXX:

1.  Create two versions:
    

-   X+X^{+}X+: original / consistent
    
-   X−X^{-}X−: globally-corrupted but locally plausible (span swap/transplant, etc.)
    

2.  Forward both through the model, but **detach between blocks** so each block’s gradients don’t flow into previous blocks.
    
3.  For each block lll:
    

-   Extract chosen activation tensor Hl+,Hl−H\_l^{+}, H\_l^{-}Hl+​,Hl−​
    
-   Compute goodness gl+\=mean(Hl+),  gl−\=mean(Hl−)g\_l^{+}=\\mathrm{mean}(H\_l^{+}),\\; g\_l^{-}=\\mathrm{mean}(H\_l^{-})gl+​\=mean(Hl+​),gl−​\=mean(Hl−​) (LN-compatible design)
    
-   Compute collaboration terms:
    
    γl+\=∑t≠lstopgrad(gt+),γl−\=∑t≠lstopgrad(gt−)\\gamma\_l^{+}=\\sum\_{t\\neq l}\\mathrm{stopgrad}(g\_t^{+}),\\quad \\gamma\_l^{-}=\\sum\_{t\\neq l}\\mathrm{stopgrad}(g\_t^{-})γl+​\=t\=l∑​stopgrad(gt+​),γl−​\=t\=l∑​stopgrad(gt−​)
    
-   Local FF loss:
    
    LlFF\=BCE(σ(gl++αγl+−θ),1)+BCE(σ(gl−+αγl−−θ),0)\\mathcal{L}^{FF}\_l = \\mathrm{BCE}(\\sigma(g\_l^{+}+\\alpha\\gamma\_l^{+}-\\theta), 1) + \\mathrm{BCE}(\\sigma(g\_l^{-}+\\alpha\\gamma\_l^{-}-\\theta), 0)LlFF​\=BCE(σ(gl+​+αγl+​−θ),1)+BCE(σ(gl−​+αγl−​−θ),0)
-   Optional contrastive term on pooled representations with depth-dependent margin (helps prevent early-layer collapse)
    
-   Update **only block lll**’s parameters.
    

This retains FF’s local update property while injecting global coordination and forcing global structure via negatives.

* * *

## Notes specific to “Transformer as an LLM” (if that’s your intent)

The literature is much stronger on FF for classification/representation learning and on Transformer-like models in vision than on full-scale autoregressive LLM training with FF. The most direct Transformer evidence I can point to is ViT-focused FF extensions using contrastive losses.

If you want autoregressive next-token prediction without falling back to BP, the most workable FF-style framing is to treat the model as an **energy/goodness scorer** for (context, candidate next token) and train it with **sampled negatives** (NCE-like), rather than full softmax over the vocabulary. This matches FF’s positive/negative discrimination spirit.
The practical constraint is inference cost: you need a candidate set (proposal model or retrieval) rather than scoring all vocab tokens.

* * *

## The shortest “best way” summary

If your goal is “FF-trained Transformer that stays globally coherent and avoids BP-style entanglement,” the highest-leverage combination is:

-   **Global-coherence negatives** (span-level swaps/transplants, sentence shuffles) + **self-contrastive in-batch negatives**
    
-   **Collaborative FF** (stop-gradient global goodness term injected into each block’s local objective)
    
-   **LN-compatible goodness** (e.g., mean-goodness + LN-separated representation stream)
    
-   **Depth-scheduled contrastive pressure** (margin increases with depth) to prevent early blocks collapsing into brittle clusters
    
-   **No end-to-end BP fine-tuning** (or freeze backbone if you add a supervised head), to preserve the sparse/modular structure FF can induce
    

If you specify the exact setting (encoder-only MLM, decoder-only AR LM, or encoder–decoder), I can pin down: (1) the exact positive/negative construction, (2) where to measure goodness in the block, and (3) the lowest-cost inference scheme consistent with FF.

## User

Do you train each layer separately in order or what

## Assistant

You still _compute_ the network in order (block 1 → block 2 → …), but you do **not** have to “train layer 1 to completion, freeze it, then move to layer 2” unless you choose that schedule.

With Forward–Forward on a Transformer, the most common/usable answer is:

-   **Each block has its own local objective**
    
-   You **update all blocks during training**, but you **block gradients between blocks** (detach/stop‑grad) so no end-to-end backprop happens.
    

Below are the three practical training schedules, in order of what usually works best.

* * *

## 1) Recommended: interleaved blockwise training (all blocks, every step)

**Idea:** On each minibatch, you do pos/neg forward passes, compute a _local loss per block_, and update blocks. The key is that the input to each block is **detached**, so gradients do not flow into earlier blocks.

### What this means concretely

-   You run embeddings once to get the initial residual stream.
    
-   For block
    
    ```
    l
    ```
    
    , you feed
    
    ```
    x.detach()
    ```
    
    into the block.
    
-   You compute block
    
    ```
    l
    ```
    
    ’s goodness and local loss from block-local activations.
    
-   That loss only updates block
    
    ```
    l
    ```
    
    parameters (because the input was detached).
    

### Minimal pseudocode sketch

```
Python# x_pos, x_neg: tokenized positives / negativesh_pos = embed(x_pos)h_neg = embed(x_neg)total_loss = 0for l, block in enumerate(blocks):    h_pos_in = h_pos.detach()    h_neg_in = h_neg.detach()    h_pos, aux_pos = block(h_pos_in, return_goodness_acts=True)    h_neg, aux_neg = block(h_neg_in, return_goodness_acts=True)    g_pos = goodness(aux_pos)   # block-local    g_neg = goodness(aux_neg)    loss_l = ff_loss(g_pos, g_neg)   # BCE / margin / etc.    total_loss = total_loss + loss_ltotal_loss.backward()optimizer.step()optimizer.zero_grad()
```

This is “train all layers at once,” but still strictly **local learning** because every block’s input was detached.

* * *

## 2) If you use “collaborative/global goodness”: two-phase per step (still not sequential training)

If your block loss uses a term like “other layers’ goodness” (global coordination), you usually want a **two-phase** step so each block can see the other blocks’ goodness values without letting gradients leak.

### Two-phase pattern

**Phase A (no gradients):** compute goodness values for all blocks on pos/neg.

-   Cache
    
    ```
    g_pos[l]
    ```
    
    ,
    
    ```
    g_neg[l]
    ```
    
    .
    

**Phase B (with gradients, local):** run blocks again with detached inputs and compute each block’s loss using:

-   its own
    
    ```
    g_l
    ```
    
    (with grad)
    
-   other blocks’
    
    ```
    g_{t≠l}
    ```
    
    as **stop-grad cached scalars**
    

This keeps the “global coordination signal” while preserving the FF constraint (no end-to-end BP).

* * *

## 3) Optional: greedy sequential (train in order, freezing earlier blocks)

This is the “train block 1, freeze it; then block 2; …” approach.

It can work, but it has downsides for Transformers:

-   later blocks are forced to adapt to whatever quirks early blocks learned
    
-   it tends to reduce cross-block coordination unless you add extra mechanisms
    

I only use this as a stabilization trick when training is collapsing:

-   train first 2–4 blocks until they produce non-trivial goodness separation
    
-   then switch to interleaved training for all blocks
    

* * *

## So the direct answer

You usually **do not** train each layer “separately in order” to completion.

You typically:

-   **update every block every step**, but
    
-   **prevent gradients from flowing across blocks** (detach between blocks),
    so each block is trained with its own FF objective while still receiving inputs produced by earlier blocks.
    

* * *

## One implementation detail that matters

There are two clean ways to implement “local updates”:

1.  **Single optimizer + one backward on summed local losses**
    Works if you detach between blocks. Each block only gets gradients from its own local loss.
    
2.  **Per-block optimizers + per-block backward/step**
    Lets you step blocks independently (useful if you want different learning rates or schedules per depth). Slightly more bookkeeping.
    

Both are compatible with “not training in order.”

* * *

If you tell me whether you mean **decoder-only (autoregressive)** vs **encoder-only / masked** Transformer, I can give the exact pos/neg construction and where to tap activations for goodness so the training doesn’t collapse under layer norm.

## User

what are teh most relevant papers for me to review? has anyone tried to do autoregressive or similar language models with FF or similar?

## Assistant

## Most relevant Forward-Forward papers to review (with an eye toward Transformers)

### 1) Original / baseline Forward-Forward

-   **Hinton (2022): “The Forward-Forward Algorithm: Some Preliminary Investigations”**
    The canonical definition of FF: two forward passes (positive vs negative), layer-local “goodness” objectives, and greedy layer-wise training as the default scheme.
    

### 2) Fixing the “layers don’t collaborate” problem (directly relevant to your concern)

-   **Lorberbom et al. (AAAI 2024): “Layer Collaboration in the Forward-Forward Algorithm”**
    Explicitly targets the main failure mode of vanilla FF in deep nets: independently trained layers don’t coordinate well, which matters even more in Transformers.
    

### 3) Better negatives / more stable training (important for anything beyond toy tasks)

-   **Ororbia & Mali (2023): “The Predictive Forward-Forward Algorithm”**
    Combines FF ideas with predictive-coding style circuitry (representation + generative circuit). If you’re thinking about sequence prediction or “world-model-ish” training, this is one of the most conceptually relevant extensions.
    
-   **Chen et al. (Nature Communications 2025): “Self-Contrastive Forward-Forward (SCFF)”**
    A modernized FF-style method emphasizing self-contrast and improved negative handling; one of the more prominent “FF, but competitive” directions.
    

### 4) Scaling/depth and “goodness” design

-   **Sun et al. (ICLR 2025): “DEEPERFORWARD”**
    Focuses on making FF work better as depth increases (goodness design + normalization choices + efficiency strategy). Useful if you’re thinking “Transformer-depth-like” even if experiments are mostly CNN-ish.
    
-   **Wu et al. (2024): “Distance-Forward Learning”**
    Reformulates/improves FF using a distance/metric-learning lens; relevant because Transformers often benefit from representation shaping objectives that look more like metric learning than “sum of squares.”
    
-   **Shah et al. (2025): “Large Scale Benchmarking of Goodness Functions for the Forward-Forward Algorithm”**
    If you’re trying to prevent “tangled representations,” you’ll likely end up changing the goodness function—this paper is directly about that design space.
    
-   **Gong et al. (2025): “FAUST: Reshaping the Forward-Forward Algorithm with a Similarity-Based Objective”**
    Connects FF to similarity-learning losses; helpful if your plan is to enforce more global/consistent representational geometry.
    

* * *

## “Has anyone tried Transformers with FF?” (and what kind)

### Vision Transformers (closest “clean” Transformer-architecture FF work)

-   **Aghagolzadeh & Ezoji (2025): “Contrastive Forward-Forward: A Training Algorithm of Vision Transformer”**
    One of the more direct “FF-style training applied to a Transformer family architecture” papers (ViT).
    

### Transformers for NLP classification (not autoregressive generation, but NLP + Transformers + FF)

-   **Kim & Yoo (2025): “Replacing Backpropagation with the Forward-Forward (FF) Algorithm in Transformer Models…”**
    Proposes FF-based training in a Transformer setting and reports experiments on **IMDB sentiment** (classification). This is relevant if you want to see how someone mapped FF onto attention blocks in practice—but it is not an end-to-end autoregressive LM paper.
    

* * *

## Autoregressive / causal language models with FF: what exists as of 2026-03-02

### What I found (published / citable)

I did **not** find a widely cited, mainstream paper showing **end-to-end pretraining of a decoder-only autoregressive language model** (GPT-like next-token LM) using **pure Hinton-style FF** at meaningful scale.

What _is_ out there falls into three buckets:

1.  **Transformer + FF, but on classification tasks** (e.g., IMDB sentiment).
    
2.  **Patent literature** explicitly discussing forward-forward training applied to tasks including _masked_ or _causal language modeling_\-type setups (useful for ideas about how to construct positives/negatives for language), but it’s not a peer-reviewed scaling result.
    
3.  **Community / repo-level attempts** at “mini-GPT with forward-forward,” but not established as a solid research baseline. The Hugging Face thread explicitly notes the lack of generative LM examples and points to experimental repos.
    

### Closest “similar, forward-only” methods actually used on language models (not FF, but highly relevant)

If your real requirement is “no backprop / no backward pass” (or “avoid global credit assignment”), these are the most relevant LM-adjacent works:

-   **Malladi et al. (NeurIPS 2023): “Fine-Tuning Language Models with Just Forward Passes” (MeZO)**
    This is a _big deal_ practically: forward-pass-only optimization for fine-tuning, evaluated on both masked and autoregressive LMs. It’s not FF; it’s zeroth-order optimization. But it’s the most concrete “LMs without backprop” result line I found.
    
-   **Pau & Aymone (2024): “Forward Learning of Large Language Models…” (PEPITA / MEMPEPITA on Transformer LLMs)**
    Again, not FF, but explicitly about forward-learning rules on Transformer-based LLMs (and discusses CLM/MLM token-level error setups).
    

* * *

## “Do you train each layer separately in order or what?” (short, concrete answer)

-   **Vanilla Hinton-FF** is **greedy layer-wise**: train layer 1 with its local objective (pos/neg), then freeze it and train layer 2, etc.
    
-   The main reason people moved beyond vanilla FF is exactly what you’re pointing at: **greedy, isolated layer training tends to harm cross-layer coordination**. The **Layer Collaboration** paper is explicitly about addressing that.
    
-   Many “FF-like” successors can be interpreted as: keep locality, but introduce **inter-layer signals** (collaboration) and/or **better contrastive setups** (better negatives) so deeper stacks don’t become dysfunctional.
    

* * *

## If your target is specifically “autoregressive LM with FF”: what to look for in papers (based on what exists)

Given the current literature, the practical path is usually:

-   Start from **FF improvements that fix negatives + collaboration** (Layer Collaboration, SCFF, goodness-function benchmarking).
    
-   Use the **Transformer implementations that exist (even if classification)** to see where people place local losses inside attention/FFN blocks and how they handle normalization/residuals.
    
-   Treat **MeZO / PEPITA** as “reference points” for what _actually works today_ on language models without standard backprop, even if they are not FF.
    
-   Mine the **patent** for explicit suggestions on applying forward-forward training to masked/causal language modeling setups (positive/negative construction ideas).
    

If you want, I can convert the above into a tighter “reading order” (5–7 items) depending on whether your immediate goal is (a) a working prototype on a small GPT-like model, or (b) inventing a better FF objective that preserves global structure in deep attention stacks.

## User

review all this lit to figure out how we can train an autoregressive generative language model using FF or somethign very similar, and figure out why no one has (and check experimental nano gpt projects to learn more)

## Assistant

## What “train each layer separately” means in Forward-Forward

In Hinton’s Forward-Forward (FF), _each layer/block has its own local objective_ (“goodness” high on positives, low on negatives), and you typically **update layers using only their local loss** while **stopping gradients between layers** (detach the input to the layer you’re updating).

So in practice you do something like:

-   Run a **positive** forward pass (real example) and a **negative** forward pass (corrupted example).
    
-   For **layer i**, compute a local loss from its activations on those two passes and update **only layer i** (no cross-layer backprop).
    
-   Move to layer i+1, etc.
    

The big caveat is that “vanilla” FF can have poor deep-layer usefulness: the **first layer alone can outperform deeper stacks**, indicating weak inter-layer collaboration.
That’s why the more relevant question for transformers is not only “separately or together?”, but **how to preserve collaboration / global behavior without end-to-end backprop**.

## The most relevant papers to review (prioritized)

### Core FF + the sequence/LM clue (and warning)

-   **Hinton (2022): “The Forward-Forward Algorithm: Some Preliminary Investigations.”**
    Defines FF, why normalization is needed to prevent trivial solutions, and explicitly flags sequential learning as a target.
    Critically: Hinton reports an early draft result on **next-character prediction** (predict next character from a window) that he later **could not replicate** and suspected was due to a bug.
    This is one of the strongest “why isn’t there an FF GPT already?” signals.
    

### Fixing the “deep networks don’t collaborate” failure mode

-   **Lorberbom et al. (2023/2024): “Layer Collaboration in the Forward-Forward Algorithm.”**
    Shows FF layers don’t effectively communicate; deeper layers can hurt; proposes **Collaborative Forward-Forward** by using a **global goodness term** (sum of other layers’ goodness) while _not_ backpropagating through those other layers (they use a frozen copy / stop-grad idea).
    For transformers, this is directly relevant to “maintain global characteristics.”
    

### Making FF train deeper/stabler (goodness + normalization + residuals)

-   **Sun et al. (ICLR 2025): “DeeperForward.”**
    Diagnoses squared-norm goodness + normalization issues in deep FF and proposes **mean goodness + layer norm**, plus residual-structure adaptations to help deep learning in FF settings.
    Even though it’s CNN-focused, the _design lessons_ (goodness definition, normalization, residual structure role) map to transformers.
    

### Moving beyond “hand-crafted negatives”

-   **Chen et al. (Nature Communications 2025): “Self-Contrastive Forward-Forward.”**
    A major direction: reduce dependence on external negative sampling and extend FF-style training to broader regimes including time-series/unsupervised settings.
    

### “FF but with an explicit generative circuit”

-   **Ororbia & Mali (2023): “The Predictive Forward-Forward Algorithm.”**
    Proposes a recurrent system that jointly learns a **representation circuit + directed generative circuit**, and reports classification/reconstruction/synthesis without backprop.
    This is highly relevant if your plan is “use a generator to produce negatives” (or to avoid manual corruptions).
    

### First steps in NLP (not language modeling yet)

-   **Gandhi et al. (2023): “Extending the Forward Forward Algorithm.”**
    Replicates MNIST and extends FF to **IMDb sentiment** (classification), explicitly positioning it as an early step beyond vision.
    

## Has anyone trained an autoregressive GPT-like LM with FF?

I did not find a peer-reviewed result demonstrating a competitive **causal/autoregressive transformer language model** trained end-to-end with Hinton-style FF as the primary learning rule.

What exists is:

-   **Hinton’s own attempted next-character prediction** in an early draft that he later could not reproduce.
    
-   Community discussion where people explicitly say they’re attempting “mini-GPT” with FF, and the referenced repos were “not working” / hard to find; one “forward\_forward\_gpt” repo is referenced but is no longer accessible at the linked location.
    
-   NLP extensions that are **classification** (IMDb) rather than **autoregressive generation**.
    

## Why “FF GPT” hasn’t happened (the concrete blockers)

### 1) The label space is enormous (vocab), and FF-style inference/training can be _label-iterative_

Classical FF classification often evaluates “goodness” for each possible label and picks the max, which scales poorly when the label set is large.
For language modeling, the “label” is the next token (often 50k–200k). Doing multiple forward passes per candidate token is a non-starter without a redesign.

### 2) Deep-layer usefulness/collaboration is already fragile in FF

Layer Collaboration shows vanilla FF can have the pathological behavior where early layers dominate and deeper layers don’t help, because layers are “not aware” of later layers.
Transformers depend on depth; if deeper blocks don’t reliably add value, you won’t get GPT-like scaling.

### 3) Negatives are easy to define for tokens, but hard to make _informative_ at scale

Random wrong-token negatives can be too easy; “hard negatives” require either:

-   a strong proposal model (which is usually trained with backprop), or
    
-   additional machinery (self-contrastive schemes, learned generators, etc.).
    

### 4) Sequential learning instability shows up early

Hinton’s inability to replicate the character-prediction result suggests that even toy LM setups can be unstable depending on goodness, learning rates, scheduling of positive/negative phases, etc.

### 5) Engineering reality: the best “nanoGPT-style” codebases are backprop-first

Useful for experimentation, but they won’t give you an FF LM “for free.” Examples like Transformer-Sandbox and cleanGPT are explicitly forks/research frameworks around nanoGPT-style training and architecture exploration, not FF training.

## A practical design to train an autoregressive generative LM with FF-like learning

If the goal is **autoregressive generation**, you must avoid “one forward pass per vocab item.”

The key move is to treat FF “positive vs negative” as **contrastive learning of (context, next-token)** with _sampled negatives_, so your compute per token is O(K) negatives, not O(|V|).

### Core formulation: conditional energy / goodness for next token

For each position _t_ in a sequence:

-   Context representation at layer i: hi,t\=Blocki(detach(hi−1))h\_{i,t} = \\mathrm{Block}\_i(\\text{detach}(h\_{i-1}))hi,t​\=Blocki​(detach(hi−1​))
    
-   Token embedding table: E\[w\]E\[w\]E\[w\]
    

Define a **goodness** for the correct next token w+\=xt+1w^+=x\_{t+1}w+\=xt+1​ and sampled negatives wk−∼q(⋅)w^-\_k\\sim q(\\cdot)wk−​∼q(⋅):

-   Simple and scalable goodness:
    
    gi(hi,t,w)\=⟨LN(hi,t),LN(E\[w\])⟩g\_i(h\_{i,t}, w) = \\langle \\mathrm{LN}(h\_{i,t}), \\mathrm{LN}(E\[w\])\\ranglegi​(hi,t​,w)\=⟨LN(hi,t​),LN(E\[w\])⟩
    
    (dot-product similarity; you can also use cosine)
    

Train each layer i with a local NCE/logistic loss:

-   Increase gi(⋅,w+)g\_i(\\cdot, w^+)gi​(⋅,w+)
    
-   Decrease gi(⋅,wk−)g\_i(\\cdot, w^-\_k)gi​(⋅,wk−​)
    

This is “FF-like” because it’s **purely forward signals + local discrimination between positives/negatives**, matching FF’s stated link to contrastive learning/NCE ideas.

### How to keep it FF (no cross-layer backprop)

For each transformer block i, you:

1.  Compute its output using **detached** input states (stop-grad boundary).
    
2.  Compute its local contrastive loss using its output only.
    
3.  Backprop **only within the block** to update its parameters.
    

This matches the “train each layer locally” spirit.

### How to preserve _global characteristics_ (avoid “layer collapse”)

Use **Collaborative Forward-Forward** ideas:

-   When updating layer i, include a **global goodness offset** computed from other layers’ goodness but treated as constant (stop-grad / frozen copy).
    
-   Operationally, you maintain a “shadow” copy f^\\hat ff^​ of other layers used only to compute the offset γ\\gammaγ, without taking gradients through it.
    

This is a direct mechanism to make early layers “care” about downstream behavior without end-to-end backprop.

### Goodness choice and normalization (important in deep nets)

Vanilla FF’s squared-norm goodness can create scaling/dead-unit issues; DeeperForward specifically argues for:

-   **layer normalization** for feature scaling
    
-   **mean goodness** to decouple representation from the goodness scalar and avoid dead-neuron update problems
    

For transformers (already LN-heavy), the takeaway is:

-   Your goodness should not be trivially encoded in magnitude that later layers can exploit.
    
-   If you rely on magnitude-based goodness, ensure LN placement truly removes it (and doesn’t leak it through residual pathways).
    

DeeperForward also highlights that residual structures help integrate features across levels in FF-style deep training.
For transformers, that suggests being very deliberate about where goodness is measured (pre/post residual, pre/post LN).

### Negative sampling strategies that actually work for LMs

You need negatives that are _not_ trivially separable.

A workable progression:

1.  **Unigram noise** negatives (word2vec-style) for a first working baseline.
    
2.  Move to **in-batch negatives** (other targets in the batch).
    
3.  Add **proposal-based hard negatives**: sample from a cheap auxiliary head (could be shallow, even trained with standard methods) and use top-k as negatives.
    This keeps training O(K) while making negatives informative.
    

### Inference: how do you actually generate text?

Once you train a final-layer representation hL,th\_{L,t}hL,t​, you can produce logits by:

ℓ\[w\]\=⟨hL,t,E\[w\]⟩\\ell\[w\] = \\langle h\_{L,t}, E\[w\]\\rangleℓ\[w\]\=⟨hL,t​,E\[w\]⟩

and sample via softmax like a standard LM.

If you also want FF’s “global goodness” to matter at inference:

-   Use **two-stage decoding**:
    
    1.  propose top-k tokens by dot-product logits
        
    2.  re-rank with a more expensive “global goodness” score that includes multiple layers
        This avoids O(|V|) FF scoring.
        

## Recommended experimental path (nanoGPT-scale, but FF-style)

### Phase 0: prove the mechanics on tiny char-level LM

-   Dataset: Tiny Shakespeare (char-level)
    
-   Model: 2–4 transformer blocks, very small d\_model
    
-   Objective: local contrastive next-char (K negatives)
    
-   Metric: bits-per-character
    

This aligns with Hinton’s own reported-but-not-reproduced direction (next-character prediction).

### Phase 1: fix deep-layer collapse early

Add layer-collaboration offset γ\\gammaγ (frozen-copy style) and compare:

-   vanilla per-layer loss vs collaborative loss
    

### Phase 2: stabilize deep training

Adopt the “goodness design + normalization” lessons from DeeperForward:

-   treat goodness in a way that doesn’t break scaling across layers
    
-   ensure LN placement prevents trivial “magnitude channel” solutions
    

### Phase 3: move to subword vocab and real LM

-   Switch tokenizer, larger vocab
    
-   Use proposal-based negatives to keep compute manageable
    
-   Evaluate perplexity and sample quality
    

## What I would conclude from the literature, in one sentence

The literature strongly suggests you **should not** train a transformer LM with naive, strictly greedy FF; you likely need (i) **collaboration/global-goodness tricks** to prevent deep-layer uselessness, (ii) **a scalable goodness definition/normalization scheme** to keep deep learning stable, and (iii) an **NCE-style sampled-negative formulation** so “label iteration over vocab” is avoided.

If you want, I can convert the above into a concrete block-by-block training pseudocode for a GPT-style transformer (including where to detach, what tensors the local loss uses, and how to structure the optimizer steps per block).

## User

should we be tweaking FF to measure goodness at a distribution of logits level? instead of a classifier. or is that the same as an already existing different approach

## Assistant

In an autoregressive LM, “a classifier” and “a distribution over logits” are basically the same object: at each position you are doing **multiclass classification over the vocabulary**. The real question is which _training objective_ you attach to those logits and whether you keep the **FF constraint** (purely local learning, no end-to-end gradient flow).

There are two distinct interpretations of what you’re proposing.

## 1) Full distribution-level goodness (cross-entropy / KL on logits)

If by “goodness at a distribution level” you mean something like:

-   **Goodness** g\=−CE(softmax(ℓ),y)g = -\\mathrm{CE}(\\text{softmax}(\\ell), y)g\=−CE(softmax(ℓ),y), or
    
-   **Goodness** g\=−KL(ptarget ∥ softmax(ℓ))g = -\\mathrm{KL}(p\_\\text{target}\\,\\|\\,\\text{softmax}(\\ell))g\=−KL(ptarget​∥softmax(ℓ))
    

then you’re just using a **proper scoring rule** on the predicted distribution.

That is not uniquely “Forward–Forward”; it’s essentially:

-   **Deep supervision / auxiliary heads** (losses applied at intermediate layers), e.g. Deeply-Supervised Nets.
    
-   Or, if you _stop gradients between blocks and train each block with its own local head_, it falls under **local-learning / local error signals** and related decoupled training families (local classifiers per layer, no global backprop to earlier layers).
    
-   Closely adjacent to “decoupled greedy learning” style decompositions (again: module-local objectives rather than one global one).
    

So: **yes**, “distribution-level goodness” largely collapses into an already-existing approach family (deep supervision / local losses). The novelty would be your _specific way of enforcing locality, collaboration, and negative construction_, not the fact you’re scoring distributions.

Practical blocker: full softmax CE requires summing over ∣V∣|V|∣V∣, which is expensive for big vocabs. Classic LM work uses sampling/importance approximations to avoid full normalization.

## 2) FF-style goodness on _token candidates_ (NCE / negative sampling / sampled softmax)

If instead you mean: “Don’t treat it as binary ‘is this input real?’; treat goodness as the **logit/energy score** of (context, candidate next token) and learn by contrasting true vs false tokens,” then you’re squarely in the territory FF already points toward.

Hinton explicitly frames FF as inspired by **Noise-Contrastive Estimation (NCE)**.
NCE itself is: train by **logistic regression discriminating data from noise**, rather than computing a full normalized likelihood.

For an LM at position ttt:

-   context representation: hth\_tht​ (from some layer/block)
    
-   token embedding: E\[w\]E\[w\]E\[w\]
    
-   score/logit (“goodness” per token): s(w)\=⟨ht,E\[w\]⟩s(w)=\\langle h\_t, E\[w\]\\rangles(w)\=⟨ht​,E\[w\]⟩
    

Then train with:

-   positive token w+\=xt+1w^+=x\_{t+1}w+\=xt+1​
    
-   negatives w1−,…,wK−∼q(⋅)w^-\_1,\\dots,w^-\_K \\sim q(\\cdot)w1−​,…,wK−​∼q(⋅)
    

A standard NCE/negative-sampling objective is:

L\=−log⁡σ(s(w+))−∑k\=1Klog⁡σ(−s(wk−))\\mathcal{L} = -\\log \\sigma(s(w^+)) - \\sum\_{k=1}^K \\log \\sigma(-s(w^-\_k))L\=−logσ(s(w+))−k\=1∑K​logσ(−s(wk−​))

This is the same basic shape as widely used negative-sampling objectives in NLP.

This variant is much more “FF-like” because:

-   it’s still **positive vs negative**
    
-   you can implement it with **two forward evaluations per sample** (or one forward for the context + multiple candidate dot-products)
    
-   you avoid the ∣V∣|V|∣V∣ softmax
    

If you want the _distribution_ eventually for generation, you can still decode with a softmax over logits at inference; the training objective just avoids full normalization.

## Which one should you do for an autoregressive FF-like LM?

If the goal is “FF or something very similar” **and** “scales to large vocab,” the more natural move is:

-   Use **candidate-level goodness** (token energy/logit) + **sampled negatives** (NCE/negative sampling / sampled softmax family).
    

If you instead do full-distribution CE/KL at each layer:

-   you are effectively doing **local losses / deep supervision**, not classic FF.
    
-   it’s viable, but the “FF-ness” is mostly in your **gradient stopping between blocks** and your **module-local objectives**, not in the loss form itself.
    

## The important nuance for “why this isn’t just ‘a classifier’”

A next-token softmax head _is_ a classifier. The distinction is:

-   FF classification (original) often relies on **evaluating goodness per label** by “clamping” the label and comparing goodness across labels. That is infeasible for vocab-size labels.
    
-   Switching to logits/energies with **sampling** is how language modeling has historically avoided the “evaluate every label” trap.
    

## Bottom line

-   **Distribution-level goodness** = you’re mostly rediscovering **cross-entropy / KL** plus **deep supervision / local losses**.
    
-   **Candidate-logit goodness with negatives** = you’re mostly rediscovering **NCE / negative sampling / sampled-softmax-style** training, which is already tightly aligned with FF’s “positive vs negative” roots.
    

What could be genuinely new is not “goodness on logits,” but:

-   enforcing **strict block-local updates** in a transformer (stop-grad boundaries),
    
-   adding **inter-layer collaboration signals** without end-to-end backprop,
    
-   and designing **hard negatives/proposals** that make the model learn global sequence constraints rather than local token heuristics.
    

If you want, I can give a concrete “FF-like GPT training objective” that is (a) token-energy NCE, (b) block-local, (c) compatible with tied embeddings, and (d) uses a proposal mechanism for hard negatives so it doesn’t collapse into unigram discrimination.

## User

so for hard negatives, Hinton tried using the model's own predictions and that didn't work? or we're not sure

## Assistant

Hinton did propose (and in one draft explicitly _demonstrated_) using the model’s own predictions to generate negatives for a toy next-character task — but in the later “final” PDF he says he could not reproduce the key result and suspects it was a bug. So it’s not a clean “it didn’t work”; it’s “he reported it working, then later wasn’t sure / couldn’t replicate.”

What the paper(s) say, precisely:

-   In the core FF setup he explicitly allows negatives to be **predicted by the net itself** (via top‑down connections) or supplied externally.
    
-   In an earlier version of the paper (the one that includes a full “predict the next character” section), he describes constructing negatives by taking a real 10‑character context and replacing the last character with **a prediction from the model** (“teacher forcing” style). He also describes generating **entire negative sequences** from the predictive model starting from an initial real context, and says this worked “very nearly as well” as teacher forcing.
    
-   In the later “FFXfinal” PDF, he revisits that as a “sleep/offline negatives” motivation and says: he previously reported doing many positive updates followed by many negative updates **generated by the net itself** with little loss, but he has been **unable to replicate** and suspects a **bug**. He adds that, with squared-activity goodness, separating positive/negative phases seems to require very low learning rate and extremely high momentum, and that maybe a different goodness function could make phase separation work (open question).
    

So the most accurate summary is:

-   **Yes**, he tried (in a toy sequence setting) using model-generated predictions as negatives.
    
-   **He initially reported it worked**, including the idea that negatives could be generated “offline/sleep.”
    
-   **Later he could not replicate the result and suspected a bug**, so we don’t have a reliable, reproduced claim that this works as stated.
    

Also note: his “couldn’t replicate” comment in the final version is tied to _separating positive and negative phases into large blocks of updates_ (wake/sleep style), not necessarily to the basic idea “use model predictions as hard negatives” in an online interleaved way.

## User

did he share code for that? low learning and high momentum may be ok, or we could play with diff goodness funcs or global sync like we discussed (unless he was already doing that)

## Assistant

### Did Hinton share code for the sequence / “sleep-phase” hard-negative experiment?

Not that I can find.

What he _did_ share publicly is:

-   **`ffcode.zip` (Matlab)**: “matlab code for the supervised version of FF with the first 10 pixels being the labels,” linked from his homepage, plus a short **README.txt** describing how to run it on MNIST.
    

That release is for the **supervised MNIST setup** (labels embedded into the input), not the **next-character prediction / offline negative generation** experiment.

In the final paper, he explicitly says the next-character “sleep-phase” result he reported in an early draft was **not reproducible** and he suspects it was due to a **bug**.
Given that, it’s unsurprising there’s no “official” code drop for the sequence case.

### What exactly was the “low learning rate / high momentum” about?

It’s specifically about **separating positive and negative learning into long blocks**, i.e.:

-   “thousands of weight updates on positive data”
    followed by
    
-   “thousands on negative data” (generated by the net)
    

He says that with **sum-of-squared-activities goodness**, that long-block alternation “only works if the learning rate is very low and the momentum is extremely high,” and that a different goodness function _might_ allow phase separation.

So two important implications:

1.  That warning is about the **wake/sleep separation regime**, not necessarily about **interleaving pos/neg within each minibatch/step** (which is closer to standard contrastive/NCE practice).
    
2.  It’s also tied to that specific goodness choice (squared activity), and he explicitly flags “different goodness function” as the likely lever.
    

### Was he “already doing global sync” like we discussed?

Depends what you mean by “global sync.”

**He _was_ doing synchronous updates of hidden states in a recurrent/iterative version of FF** (his “treat a static image as a boring video” idea). He reports that synchronous updates of _all hidden layers_ with damping actually learned slightly better than alternating layer updates.

That is “global synchronization” in the sense of **global, simultaneous state updates during iterative inference**.

But **he was not doing “global synchronization” of learning via a cross-layer coordination term in the loss** (the kind of “layer collaboration / global goodness injection” we discussed earlier). In the FF paper(s), the main coordination mechanism is architectural/iterative (the recurrent framing), not an explicit multi-layer coupling term in the objective.

### Did he already use “model-based hard negatives”?

Yes, in two different senses (but with caveats):

1.  **Sequence draft (FFA13)**: he describes negatives where the last character in a window is replaced by **a prediction from the previous 10 chars** (“teacher forcing”), and also negatives **entirely generated by the predictive model** starting from the initial real context, which he says worked nearly as well _in that draft_.
    
2.  **Final paper (FFXfinal)**: he later says he could not replicate the sleep-phase block-alternation result and suspects a bug; plus it only worked under extreme optimizer settings with squared-activity goodness.
    

Also, in the recurrent MNIST setup he generates “harder” negatives by sampling **incorrect labels in proportion to the model’s predicted probabilities** (instead of uniform wrong labels), explicitly noting it makes training more efficient.

### Practical takeaways for your direction

-   **Low LR + very high momentum** is plausibly “OK” _if_ you really want the long wake/sleep separation, but Hinton’s final writeup strongly suggests that regime is currently brittle and goodness-dependent.
    
-   A more robust path is usually: **don’t separate phases into long blocks**; instead **interleave pos/neg every step** (or every minibatch) so you’re not relying on momentum to average two opposing objectives.
    
-   Exploring **different goodness functions** is explicitly endorsed by Hinton as the likely way to make separation feasible.
    
-   If you want “global coherence” without backprop, Hinton’s paper gives one mechanism (iterative/recurrent inference with synchronous updates) , while later work (not Hinton’s) gives another mechanism (explicit layer-collaboration terms).
    

If you want, I can propose 2–3 concrete “goodness” definitions for an autoregressive LM that (a) avoid vocab-wide scoring per step and (b) are compatible with either interleaved pos/neg or a stabilized wake/sleep separation.

## User

yes

## Assistant

Below are three concrete “goodness” definitions for an autoregressive LM that (a) never require scoring the full vocabulary and (b) can be implemented in an FF‑like, block‑local way (detach between blocks). I’ll also include how each interacts with “hard negatives,” and how to add the global synchronization term we discussed without turning it into end‑to‑end backprop.

I’ll use this notation:

-   Tokens: x1,…,xTx\_1,\\dots,x\_Tx1​,…,xT​. Target at position ttt: yt\=xt+1y\_t = x\_{t+1}yt​\=xt+1​.
    
-   Transformer blocks l\=1..Ll=1..Ll\=1..L. Block output hidden state at position ttt: hl,t∈Rdh\_{l,t}\\in\\mathbb{R}^dhl,t​∈Rd.
    
-   Token embedding matrix E∈R∣V∣×dE\\in\\mathbb{R}^{|V|\\times d}E∈R∣V∣×d (often tied to output).
    
-   Candidate set at position ttt: Ct\={yt}∪{wt,1−,…,wt,K−}C\_t = \\{y\_t\\}\\cup\\{w^-\_{t,1},\\dots,w^-\_{t,K}\\}Ct​\={yt​}∪{wt,1−​,…,wt,K−​} with small KKK (e.g. 16–256).
    
-   “Score” (logit/energy) from layer lll: sl,t(w)s\_{l,t}(w)sl,t​(w). You compute this only for w∈Ctw\\in C\_tw∈Ct​ via embedding gathers.
    

A standard choice that avoids scale hacks:

sl,t(w)\=⟨LN(hl,t), LN(E\[w\])⟩τs\_{l,t}(w) = \\frac{\\langle \\mathrm{LN}(h\_{l,t}),\\, \\mathrm{LN}(E\[w\])\\rangle}{\\tau}sl,t​(w)\=τ⟨LN(hl,t​),LN(E\[w\])⟩​

(or sl,t(w)\=E\[w\]⊤LN(hl,t)s\_{l,t}(w)= E\[w\]^\\top \\mathrm{LN}(h\_{l,t})sl,t​(w)\=E\[w\]⊤LN(hl,t​) if you don’t want LN on embeddings).

* * *

## Goodness 1: Pairwise margin “energy wins” (no softmax, very FF‑like)

**Goodness idea:** the correct token should score higher than negatives by a margin. This is the most direct FF analogue: “positive > negative” locally.

For each negative w−∈Ct∖{yt}w^- \\in C\_t\\setminus\\{y\_t\\}w−∈Ct​∖{yt​}, define pairwise margin:

Δl,t(w−)\=sl,t(yt)−sl,t(w−)\\Delta\_{l,t}(w^-) = s\_{l,t}(y\_t) - s\_{l,t}(w^-)Δl,t​(w−)\=sl,t​(yt​)−sl,t​(w−)

Layer‑local loss (smooth hinge / logistic margin):

Ll,t(1)\=∑w−softplus(m−Δl,t(w−))\\mathcal{L}^{(1)}\_{l,t} = \\sum\_{w^-} \\mathrm{softplus}(m - \\Delta\_{l,t}(w^-))Ll,t(1)​\=w−∑​softplus(m−Δl,t​(w−))

-   mmm is a margin (often depth‑scheduled: smaller for early blocks, larger for later blocks).
    

**Why it’s attractive**

-   No normalization, no full distribution.
    
-   Directly matches “goodness should separate positives and negatives.”
    
-   Stable with small candidate sets.
    

**Hard negatives**

-   This objective benefits strongly from _hard_ negatives because easy negatives saturate.
    
-   Good default: for each position ttt, pick negatives as a mixture:
    
    -   top‑k1k\_1k1​ tokens from a proposal distribution (see below)
        
    -   -   k2k\_2k2​ in‑batch other targets (other yt′y\_{t'}yt′​ in the batch)
            
    -   -   k3k\_3k3​ random/unigram tokens for diversity
            (so you don’t overfit to one proposal mechanism)
            

**Wake/sleep compatibility**

-   Works fine interleaved (pos/neg in same step).
    
-   Also compatible with “sleep”: negatives can be model‑generated continuations; you then compare correct token(s) vs tokens that the sleep generator proposes (or appears in a negative continuation).
    

* * *

## Goodness 2: Candidate‑set softmax cross‑entropy (distribution‑level, but cheap)

This is the cleanest way to do “distribution‑level goodness” without ∣V∣|V|∣V∣.

Define a restricted softmax over the candidate set CtC\_tCt​:

pl,t(w∣Ct)\=exp⁡(sl,t(w))∑u∈Ctexp⁡(sl,t(u))p\_{l,t}(w\\mid C\_t) = \\frac{\\exp(s\_{l,t}(w))}{\\sum\_{u\\in C\_t}\\exp(s\_{l,t}(u))}pl,t​(w∣Ct​)\=∑u∈Ct​​exp(sl,t​(u))exp(sl,t​(w))​

Local loss:

Ll,t(2)\=−log⁡pl,t(yt∣Ct)\\mathcal{L}^{(2)}\_{l,t} = -\\log p\_{l,t}(y\_t\\mid C\_t)Ll,t(2)​\=−logpl,t​(yt​∣Ct​)

You can define “goodness” as gl,t\=−Ll,t(2)g\_{l,t}=-\\mathcal{L}^{(2)}\_{l,t}gl,t​\=−Ll,t(2)​ if you want to keep FF language.

**What this is (and isn’t)**

-   It is “classifier-like,” but only over K+1K+1K+1 candidates, not the full vocab.
    
-   It’s effectively sampled-softmax / contrastive CE. The novelty is not CE itself; it’s using it as a **strictly local objective per block** with detach boundaries, plus your negative strategy and layer collaboration.
    

**Hard negatives**

-   Same mixture strategy as above.
    
-   This objective is somewhat more forgiving than pairwise margin if some negatives are easy (because normalization still shapes gradients).
    

**Wake/sleep compatibility**

-   Interleaved training works well.
    
-   For “sleep,” you can populate CtC\_tCt​ using top‑kkk from the sleep generator at that position (plus extras).
    

**Optional correction**

-   If you care about unbiased likelihood estimation, you can add importance corrections based on the negative sampling distribution q(w)q(w)q(w). Practically, many systems just treat it as a contrastive objective and rely on good negatives.
    

* * *

## Goodness 3: Sequence/continuation goodness (global coherence beyond one token)

Token‑level training can still yield locally good but globally incoherent behavior. A direct way to encode “global characteristics” is to define goodness over a _short continuation_ window.

Pick a short horizon HHH (e.g. 8–64 tokens). For each start position ttt, define the continuation:

-   Positive continuation: Yt+\=(xt+1,…,xt+H)Y^+\_{t} = (x\_{t+1},\\dots,x\_{t+H})Yt+​\=(xt+1​,…,xt+H​)
    
-   Negative continuation: Yt−Y^-\_{t}Yt−​ created by:
    
    -   model-generated continuation (“sleep”)
        
    -   or splice/swap/shuffle spans to break global structure while keeping local plausibility
        

Define layer‑wise continuation score:

Gl(t,Y)\=∑j\=1Hsl,t+j−1(Y\[j\])G\_l(t, Y) = \\sum\_{j=1}^{H} s\_{l,t+j-1}\\big(Y\[j\]\\big)Gl​(t,Y)\=j\=1∑H​sl,t+j−1​(Y\[j\])

(where sl,ts\_{l,t}sl,t​ is still computed cheaply using embedding gathers; for negatives you only need to score the actual negative tokens present in Y−Y^-Y−, not the whole vocab.)

Now do a contrastive “positive continuation beats negative continuation” loss:

Ll,t(3)\=softplus(m−(Gl(t,Yt+)−Gl(t,Yt−)))\\mathcal{L}^{(3)}\_{l,t} = \\mathrm{softplus}\\big(m - (G\_l(t,Y^+\_t) - G\_l(t,Y^-\_t))\\big)Ll,t(3)​\=softplus(m−(Gl​(t,Yt+​)−Gl​(t,Yt−​)))

Optionally with multiple negatives per ttt.

**Why it’s attractive**

-   Forces multi-token, cross-time constraints.
    
-   Makes “sleep-phase” style negatives conceptually natural: the model generates coherent-looking continuations that are still wrong relative to the data’s continuation.
    

**Hard negatives**

-   Model-generated continuations are naturally hard (especially if generated from a stronger proposal head or a deeper layer).
    
-   You can mix in “structural” negatives (span swap, entity swap) to explicitly target global consistency failures.
    

**Wake/sleep compatibility**

-   This is the most directly compatible with wake/sleep: wake uses real continuations; sleep generates negative continuations.
    

* * *

## How to implement these in a block-local FF-like Transformer

A practical block-local training skeleton (for any of the above losses):

1.  Run embeddings once to get h0,:h\_{0,:}h0,:​.
    
2.  For each block lll:
    
    -   Feed **detached** input to the block: hl,:\=Blockl(stopgrad(hl−1,:))h\_{l,:} = \\mathrm{Block}\_l(\\mathrm{stopgrad}(h\_{l-1,:}))hl,:​\=Blockl​(stopgrad(hl−1,:​))
        
    -   Compute the block’s local loss ∑tLl,t\\sum\_t \\mathcal{L}\_{l,t}∑t​Ll,t​ using hl,th\_{l,t}hl,t​ only.
        
3.  Sum losses across blocks (or step each block separately).
    
4.  Backward once; because inputs were detached, gradients stay inside each block.
    

Key point: candidate scoring never needs full vocab; it’s gather dot-products against E\[w\]E\[w\]E\[w\] for w∈Ctw\\in C\_tw∈Ct​.

* * *

## Proposal mechanism for hard negatives (critical)

Hard negatives are easiest if you have a _proposal distribution_ πt(w)\\pi\_t(w)πt​(w) that’s already somewhat aligned with the task.

Three practical proposals (in increasing “FF purity risk”):

1.  **Final-layer proposal, stop-grad**
    Use the current model’s _top-layer_ logits to select top‑kkk tokens for each ttt, but treat those selections and scores as stop-grad when training lower blocks.
    
    -   Cheap, very effective.
        
    -   Risk: if the final layer is bad early, negatives are not hard.
        
2.  **Deeper-layer proposal for shallower blocks**
    When training block lll, sample negatives from block l+ql+ql+q (with stop-grad). This encourages a coarse-to-fine hierarchy.
    
    -   Helps keep deeper layers relevant.
        
3.  **External lightweight proposal (frozen or separately trained)**
    E.g., a small n‑gram, a tiny backprop LM, or even a cached KNN proposal. This breaks circularity in early training.
    
    -   “Less pure,” but often necessary for bootstrapping.
        

In all cases: mix proposal negatives with in-batch and random for diversity.

* * *

## Add “global sync” without end-to-end backprop

Two low-cost “global coordination” add-ons that do not require full vocab:

### A) Cross-layer distribution alignment on candidate sets

Compute candidate-set distributions pl,t(⋅∣Ct)p\_{l,t}(\\cdot\\mid C\_t)pl,t​(⋅∣Ct​) and align them to a reference (typically the top layer) with stop-grad:

Ll,tsync\=KL(stopgrad(pL,t(⋅∣Ct)) ∥ pl,t(⋅∣Ct))\\mathcal{L}^{sync}\_{l,t} = \\mathrm{KL}\\left(\\mathrm{stopgrad}\\left(p\_{L,t}(\\cdot\\mid C\_t)\\right)\\ \\|\\ p\_{l,t}(\\cdot\\mid C\_t)\\right)Ll,tsync​\=KL(stopgrad(pL,t​(⋅∣Ct​)) ∥ pl,t​(⋅∣Ct​))

-   Only computed on CtC\_tCt​, so it’s cheap.
    
-   Encourages global agreement while each block still trains locally.
    

### B) Global goodness offset (“collaborative” scalar)

If you’re using a scalar continuation goodness GlG\_lGl​ (Goodness 3) or even token-level goodness summaries, you can add a stop-grad global term:

g~l\=gl+α⋅stopgrad(∑r≠lgr)\\tilde{g}\_l = g\_l + \\alpha \\cdot \\mathrm{stopgrad}\\Big(\\sum\_{r\\neq l} g\_r\\Big)g~​l​\=gl​+α⋅stopgrad(r\=l∑​gr​)

and train with g~l\\tilde g\_lg~​l​. This is the scalar version of “global coordination” and tends to reduce the “only early layers matter” failure mode.

* * *

## Recommendation: what to try first

If you want the fastest path to “works on nanoGPT-scale”:

1.  **Goodness 2 (candidate-set CE)** + mixed hard negatives + detach boundaries
    

-   It most closely matches standard LM training signals while avoiding full softmax.
    

2.  Add **sync KL on candidate sets** (align intermediate blocks to top block with stop-grad)
    

-   This is an inexpensive coordination mechanism.
    

3.  Add **Goodness 3 (continuation contrast)** once token-level learning works
    

-   This is the most direct lever for “global characteristics.”
    

Then, if you still want wake/sleep separation:

-   Try it _after_ interleaved training is stable, and start with short sleep bursts (not thousands of steps) to avoid optimizer brittleness.
    

* * *

## Two design choices you should decide early (they matter)

1.  **Do you tie embeddings across layers?**
    If all layers share the same EEE, then every block is trying to shape the same token geometry. This can help alignment, but it can also create coupling. A compromise:
    

-   tie EEE globally, but only allow some layers to update it (e.g., top layer + maybe one auxiliary).
    

2.  **Depth scheduling (avoid “early block does everything”)**
    To keep deep blocks relevant:
    

-   smaller margin / weaker objective early, stronger later
    
-   or restrict early blocks to coarse targets (e.g., token clusters) and refine later
    

* * *

If you specify:

-   model size (e.g., nanoGPT 6-layer / d=384)
    
-   vocab type (char vs BPE)
    
-   and whether you want strict wake/sleep or interleaved training
    

I can translate the above into a concrete training loop with exact tensors, where to detach, candidate set construction, and a default set of hyperparameters (K, margins/temperatures, sync weights) that are usually stable.

## User

Figure out the best way to train a transformer using hintons forward forward so it maintains its global characteristics without falling back to the highly tangled representations encouraged by backprop

## Assistant

Thought for 12m 12s

## What breaks when you naively port Forward–Forward to Transformers

Hinton’s Forward–Forward (FF) replaces end-to-end backprop with _layer-local_ objectives trained from two forward passes: a **positive** pass (real/consistent input) and a **negative** pass (corrupted/inconsistent input). Each layer learns to yield **high goodness** on positives and **low goodness** on negatives.

For Transformers, the main failure modes are:

1.  **Inter-layer isolation kills “global” behavior**
    Plain FF trains each layer “as if it were the whole model,” so early layers don’t get pressured to produce features that later layers can actually use. This lack of collaboration has been analyzed explicitly, and a simple modification was proposed to inject global information into each layer’s local objective.
    
2.  **Normalization collides with classic goodness choices**
    If goodness is something like “sum of squared activations,” layer normalization can wipe out the signal (or make it trivial/constant), so you must compute goodness at the right point or redesign it. Hinton notes this directly.
    There are newer “LN-compatible” goodness designs (e.g., mean-goodness + layer norm) intended to make deep FF training stable.
    
3.  **Negatives must be _globally_ hard, not just locally noisy**
    If negatives are too easy (random garbage), layers learn shortcuts; if they’re too different from data, you don’t learn data structure. Work on self-contrastive FF emphasizes that the _quality/structure of negatives_ is a key limiter and proposes systematic generation from the data distribution.
    
4.  **If you “finish” with backprop, you often undo the point**
    If you FF-pretrain then do full end-to-end BP fine-tuning, you’ll typically reintroduce the very representation entanglement you’re trying to avoid. One reason FF is interesting here is that FF objectives can yield sparse, ensemble-like internal structure that’s not typically seen with standard BP training.
    

* * *

## A strong recipe: Collaborative + LN-compatible + globally-structured negatives

The most defensible “best way” (based on what’s worked in the literature so far) is to combine four ideas:

1.  **Blockwise FF (treat each Transformer block as a “layer”)**
    
2.  **Collaborative FF (inject global goodness into each block’s local objective)**
    
3.  **LN-compatible goodness (so Transformers don’t trivialize goodness)**
    
4.  **Negatives that specifically break _global coherence_ while preserving local plausibility**
    

A practical instantiation is:

### 1) Architecture constraints that make FF work in a Transformer

**Use a Pre-LN Transformer** (standard in modern LLMs anyway), but be explicit about _where_ you measure goodness.

For block lll, you have something like:

-   ul\=LN(xl)u\_l = \\mathrm{LN}(x\_l)ul​\=LN(xl​)
    
-   al\=Attnl(ul)a\_l = \\mathrm{Attn}\_l(u\_l)al​\=Attnl​(ul​), xl′\=xl+alx'\_l = x\_l + a\_lxl′​\=xl​+al​
    
-   vl\=LN(xl′)v\_l = \\mathrm{LN}(x'\_l)vl​\=LN(xl′​)
    
-   ml\=MLPl(vl)m\_l = \\mathrm{MLP}\_l(v\_l)ml​\=MLPl​(vl​), xl+1\=xl′+mlx\_{l+1} = x'\_l + m\_lxl+1​\=xl′​+ml​
    

**Key FF requirement:** goodness must be computed on an activation that is not made invariant/constant by LN.

Two robust options:

-   **Option A (Hinton-style):** compute goodness on _pre-LN_ hidden activities (he explicitly defines goodness using activities “before layer normalization”).
    
-   **Option B (DeeperForward-style):** use **mean goodness** while using LN for the representation stream (decoupling “goodness” from “features”); this is explicitly motivated as fixing scaling + deactivated neuron issues.
    

For Transformers, Option B is usually easier:

-   Let HlH\_lHl​ be a block-local activation tensor you choose (common: the MLP pre-activation after a ReLU/GELU or an equivalent “post-nonlinearity” state).
    
-   Define **goodness** as a mean over features (and optionally tokens):
    
    gl\=mean(Hl)g\_l = \\mathrm{mean}(H\_l)gl​\=mean(Hl​)
-   Define the **representation passed upward** as the LN-normalized stream (standard Transformer residual stream).
    

This mirrors the “goodness is mean; features are LN outputs” separation.

### 2) Positive/negative construction that forces global structure

To preserve “global characteristics,” your negatives must break _global_ constraints while keeping _local_ statistics plausible.

Use a **mixture of negative types**:

-   **Local-preserving, global-breaking (the important ones):**
    
    -   swap two long spans (A … B → B … A)
        
    -   replace a middle span with one from another sample (in-batch “span transplant”)
        
    -   sentence-order shuffles (for multi-sentence text)
        
    -   entity/attribute swaps (replace all mentions of one entity with another from batch)
        
-   **Light local corruption (for robustness, not for global structure):**
    
    -   small token masking/dropout
        
    -   small token substitutions sampled from unigram or same-POS bucket
        

If you want a simple default that avoids bespoke negative generators, borrow the **self-contrastive** pattern: positives/negatives formed directly from the batch rather than a separate negative model.
(You will still want the _span-level_ variants above if your explicit goal is “global characteristics.”)

### 3) Layer-local losses that don’t collapse early layers

A plain FF layer loss is “classify positive vs negative from goodness,” using a logistic mapping and a threshold.

For Transformers, you will usually get better behavior if you add a **contrastive term** at each block in addition to (or replacing) the thresholded goodness classification—this is exactly the direction taken by work extending FF to Vision Transformers using contrastive-inspired losses and a depth-dependent margin to avoid early-layer collapse.

So for each block lll, compute a pooled representation rlr\_lrl​ (e.g., CLS token or mean pooling over tokens). Then use:

**(a) FF goodness classification (local):**

-   gl+g\_l^{+}gl+​ from positive sample, gl−g\_l^{-}gl−​ from negative
    
-   logistic probability p\=σ(gl−θ)p = \\sigma(g\_l - \\theta)p\=σ(gl​−θ)
    
-   loss:
    
    LlFF\=BCE(p+,1)+BCE(p−,0)\\mathcal{L}^{FF}\_l = \\mathrm{BCE}(p^{+}, 1) + \\mathrm{BCE}(p^{-}, 0)LlFF​\=BCE(p+,1)+BCE(p−,0)

**(b) Contrastive alignment term (optional but recommended):**

-   Use InfoNCE or supervised-contrastive if you have labels
    
-   If you want to prevent early layers from over-clustering (a known problem), use a **layer-dependent margin** (a “marginal contrastive” idea) so early blocks only learn _some_ similarity, later blocks enforce tighter grouping.
    

Total:

Ll\=λffLlFF+λctrLlCTR\\mathcal{L}\_l = \\lambda\_{ff}\\mathcal{L}^{FF}\_l + \\lambda\_{ctr}\\mathcal{L}^{CTR}\_lLl​\=λff​LlFF​+λctr​LlCTR​

### 4) Add collaboration without backprop: “global goodness injection”

This is the piece that most directly targets your “maintains global characteristics” requirement.

Layer-collaboration work proposes modifying each layer’s local probability by adding a **stop-gradient global term** based on other layers’ goodness, so each layer optimizes its parameters with awareness of the network-wide state, but still _without_ end-to-end backprop.

Concretely:

-   Compute goodness values g1,…,gLg\_1,\\dots,g\_Lg1​,…,gL​ for the current sample (positive or negative).
    
-   For layer lll, define:
    
    γl\=∑t≠lstopgrad(gt)\\gamma\_l = \\sum\_{t\\neq l} \\mathrm{stopgrad}(g\_t)γl​\=t\=l∑​stopgrad(gt​)
-   Use:
    
    pl\=σ(gl+αγl−θ)p\_l = \\sigma(g\_l + \\alpha\\gamma\_l - \\theta)pl​\=σ(gl​+αγl​−θ)
-   Then do BCE on plp\_lpl​ vs {positive=1, negative=0}.
    

This gives you a mechanism for global coordination while keeping updates local in the FF sense.

* * *

## Preventing “backprop-like entanglement” inside FF training

Even with FF, a Transformer can learn overly distributed, hard-to-parse features unless you bias it toward modularity. Two pragmatic additions:

### A) Head/feature-group specialization pressure

A channel-wise competitive learning variant of FF explicitly aims to improve feature partitioning and compositionality (developed for CNNs, but the principle maps cleanly to Transformer “groups” such as heads or MLP channel blocks).

Transformer adaptation (direct, effective):

-   Partition the hidden dimension into GGG groups (or treat attention heads as groups).
    
-   Compute goodness per-group gl,gg\_{l,g}gl,g​.
    
-   Add a mild regularizer that encourages _few_ groups to be strongly active per sample (k-winners / sparsemax-like behavior), which tends to reduce entanglement.
    

### B) Keep FF “pure”: don’t do end-to-end BP fine-tuning

If you BP-fine-tune all blocks, you’ll typically lose the sparse-ensemble structure that FF objectives can produce. Evidence that FF-trained networks can develop sparse, category-specific ensembles (and that this sparsity pattern differs from typical BP training) supports keeping the training regime consistent if interpretability/modularity is a goal.

If you must add a supervised head:

-   Train **only** a small readout / classifier head with BP and freeze the FF-trained backbone (or use tiny adapters), so you don’t re-entangle the internal representation.
    

* * *

## Minimal pseudo-algorithm (blockwise, collaborative, globally-hard negatives)

For batch XXX:

1.  Create two versions:
    

-   X+X^{+}X+: original / consistent
    
-   X−X^{-}X−: globally-corrupted but locally plausible (span swap/transplant, etc.)
    

2.  Forward both through the model, but **detach between blocks** so each block’s gradients don’t flow into previous blocks.
    
3.  For each block lll:
    

-   Extract chosen activation tensor Hl+,Hl−H\_l^{+}, H\_l^{-}Hl+​,Hl−​
    
-   Compute goodness gl+\=mean(Hl+),  gl−\=mean(Hl−)g\_l^{+}=\\mathrm{mean}(H\_l^{+}),\\; g\_l^{-}=\\mathrm{mean}(H\_l^{-})gl+​\=mean(Hl+​),gl−​\=mean(Hl−​) (LN-compatible design)
    
-   Compute collaboration terms:
    
    γl+\=∑t≠lstopgrad(gt+),γl−\=∑t≠lstopgrad(gt−)\\gamma\_l^{+}=\\sum\_{t\\neq l}\\mathrm{stopgrad}(g\_t^{+}),\\quad \\gamma\_l^{-}=\\sum\_{t\\neq l}\\mathrm{stopgrad}(g\_t^{-})γl+​\=t\=l∑​stopgrad(gt+​),γl−​\=t\=l∑​stopgrad(gt−​)
    
-   Local FF loss:
    
    LlFF\=BCE(σ(gl++αγl+−θ),1)+BCE(σ(gl−+αγl−−θ),0)\\mathcal{L}^{FF}\_l = \\mathrm{BCE}(\\sigma(g\_l^{+}+\\alpha\\gamma\_l^{+}-\\theta), 1) + \\mathrm{BCE}(\\sigma(g\_l^{-}+\\alpha\\gamma\_l^{-}-\\theta), 0)LlFF​\=BCE(σ(gl+​+αγl+​−θ),1)+BCE(σ(gl−​+αγl−​−θ),0)
-   Optional contrastive term on pooled representations with depth-dependent margin (helps prevent early-layer collapse)
    
-   Update **only block lll**’s parameters.
    

This retains FF’s local update property while injecting global coordination and forcing global structure via negatives.

* * *

## Notes specific to “Transformer as an LLM” (if that’s your intent)

The literature is much stronger on FF for classification/representation learning and on Transformer-like models in vision than on full-scale autoregressive LLM training with FF. The most direct Transformer evidence I can point to is ViT-focused FF extensions using contrastive losses.

If you want autoregressive next-token prediction without falling back to BP, the most workable FF-style framing is to treat the model as an **energy/goodness scorer** for (context, candidate next token) and train it with **sampled negatives** (NCE-like), rather than full softmax over the vocabulary. This matches FF’s positive/negative discrimination spirit.
The practical constraint is inference cost: you need a candidate set (proposal model or retrieval) rather than scoring all vocab tokens.

* * *

## The shortest “best way” summary

If your goal is “FF-trained Transformer that stays globally coherent and avoids BP-style entanglement,” the highest-leverage combination is:

-   **Global-coherence negatives** (span-level swaps/transplants, sentence shuffles) + **self-contrastive in-batch negatives**
    
-   **Collaborative FF** (stop-gradient global goodness term injected into each block’s local objective)
    
-   **LN-compatible goodness** (e.g., mean-goodness + LN-separated representation stream)
    
-   **Depth-scheduled contrastive pressure** (margin increases with depth) to prevent early blocks collapsing into brittle clusters
    
-   **No end-to-end BP fine-tuning** (or freeze backbone if you add a supervised head), to preserve the sparse/modular structure FF can induce
    

If you specify the exact setting (encoder-only MLM, decoder-only AR LM, or encoder–decoder), I can pin down: (1) the exact positive/negative construction, (2) where to measure goodness in the block, and (3) the lowest-cost inference scheme consistent with FF.

## User

Do you train each layer separately in order or what

## Assistant

Thought for 6m 37s

You still _compute_ the network in order (block 1 → block 2 → …), but you do **not** have to “train layer 1 to completion, freeze it, then move to layer 2” unless you choose that schedule.

With Forward–Forward on a Transformer, the most common/usable answer is:

-   **Each block has its own local objective**
    
-   You **update all blocks during training**, but you **block gradients between blocks** (detach/stop‑grad) so no end-to-end backprop happens.
    

Below are the three practical training schedules, in order of what usually works best.

* * *

## 1) Recommended: interleaved blockwise training (all blocks, every step)

**Idea:** On each minibatch, you do pos/neg forward passes, compute a _local loss per block_, and update blocks. The key is that the input to each block is **detached**, so gradients do not flow into earlier blocks.

### What this means concretely

-   You run embeddings once to get the initial residual stream.
    
-   For block
    
    ```
    l
    ```
    
    , you feed
    
    ```
    x.detach()
    ```
    
    into the block.
    
-   You compute block
    
    ```
    l
    ```
    
    ’s goodness and local loss from block-local activations.
    
-   That loss only updates block
    
    ```
    l
    ```
    
    parameters (because the input was detached).
    

### Minimal pseudocode sketch

```
Python# x_pos, x_neg: tokenized positives / negativesh_pos = embed(x_pos)h_neg = embed(x_neg)total_loss = 0for l, block in enumerate(blocks):    h_pos_in = h_pos.detach()    h_neg_in = h_neg.detach()    h_pos, aux_pos = block(h_pos_in, return_goodness_acts=True)    h_neg, aux_neg = block(h_neg_in, return_goodness_acts=True)    g_pos = goodness(aux_pos)   # block-local    g_neg = goodness(aux_neg)    loss_l = ff_loss(g_pos, g_neg)   # BCE / margin / etc.    total_loss = total_loss + loss_ltotal_loss.backward()optimizer.step()optimizer.zero_grad()
```

This is “train all layers at once,” but still strictly **local learning** because every block’s input was detached.

* * *

## 2) If you use “collaborative/global goodness”: two-phase per step (still not sequential training)

If your block loss uses a term like “other layers’ goodness” (global coordination), you usually want a **two-phase** step so each block can see the other blocks’ goodness values without letting gradients leak.

### Two-phase pattern

**Phase A (no gradients):** compute goodness values for all blocks on pos/neg.

-   Cache
    
    ```
    g_pos[l]
    ```
    
    ,
    
    ```
    g_neg[l]
    ```
    
    .
    

**Phase B (with gradients, local):** run blocks again with detached inputs and compute each block’s loss using:

-   its own
    
    ```
    g_l
    ```
    
    (with grad)
    
-   other blocks’
    
    ```
    g_{t≠l}
    ```
    
    as **stop-grad cached scalars**
    

This keeps the “global coordination signal” while preserving the FF constraint (no end-to-end BP).

* * *

## 3) Optional: greedy sequential (train in order, freezing earlier blocks)

This is the “train block 1, freeze it; then block 2; …” approach.

It can work, but it has downsides for Transformers:

-   later blocks are forced to adapt to whatever quirks early blocks learned
    
-   it tends to reduce cross-block coordination unless you add extra mechanisms
    

I only use this as a stabilization trick when training is collapsing:

-   train first 2–4 blocks until they produce non-trivial goodness separation
    
-   then switch to interleaved training for all blocks
    

* * *

## So the direct answer

You usually **do not** train each layer “separately in order” to completion.

You typically:

-   **update every block every step**, but
    
-   **prevent gradients from flowing across blocks** (detach between blocks),
    so each block is trained with its own FF objective while still receiving inputs produced by earlier blocks.
    

* * *

## One implementation detail that matters

There are two clean ways to implement “local updates”:

1.  **Single optimizer + one backward on summed local losses**
    Works if you detach between blocks. Each block only gets gradients from its own local loss.
    
2.  **Per-block optimizers + per-block backward/step**
    Lets you step blocks independently (useful if you want different learning rates or schedules per depth). Slightly more bookkeeping.
    

Both are compatible with “not training in order.”

* * *

If you tell me whether you mean **decoder-only (autoregressive)** vs **encoder-only / masked** Transformer, I can give the exact pos/neg construction and where to tap activations for goodness so the training doesn’t collapse under layer norm.

## User

what are teh most relevant papers for me to review? has anyone tried to do autoregressive or similar language models with FF or similar?

## Assistant

Thought for 15m 37s

## Assistant

## Most relevant Forward-Forward papers to review (with an eye toward Transformers)

### 1) Original / baseline Forward-Forward

-   **Hinton (2022): “The Forward-Forward Algorithm: Some Preliminary Investigations”**
    The canonical definition of FF: two forward passes (positive vs negative), layer-local “goodness” objectives, and greedy layer-wise training as the default scheme.
    

### 2) Fixing the “layers don’t collaborate” problem (directly relevant to your concern)

-   **Lorberbom et al. (AAAI 2024): “Layer Collaboration in the Forward-Forward Algorithm”**
    Explicitly targets the main failure mode of vanilla FF in deep nets: independently trained layers don’t coordinate well, which matters even more in Transformers.
    

### 3) Better negatives / more stable training (important for anything beyond toy tasks)

-   **Ororbia & Mali (2023): “The Predictive Forward-Forward Algorithm”**
    Combines FF ideas with predictive-coding style circuitry (representation + generative circuit). If you’re thinking about sequence prediction or “world-model-ish” training, this is one of the most conceptually relevant extensions.
    
-   **Chen et al. (Nature Communications 2025): “Self-Contrastive Forward-Forward (SCFF)”**
    A modernized FF-style method emphasizing self-contrast and improved negative handling; one of the more prominent “FF, but competitive” directions.
    

### 4) Scaling/depth and “goodness” design

-   **Sun et al. (ICLR 2025): “DEEPERFORWARD”**
    Focuses on making FF work better as depth increases (goodness design + normalization choices + efficiency strategy). Useful if you’re thinking “Transformer-depth-like” even if experiments are mostly CNN-ish.
    
-   **Wu et al. (2024): “Distance-Forward Learning”**
    Reformulates/improves FF using a distance/metric-learning lens; relevant because Transformers often benefit from representation shaping objectives that look more like metric learning than “sum of squares.”
    
-   **Shah et al. (2025): “Large Scale Benchmarking of Goodness Functions for the Forward-Forward Algorithm”**
    If you’re trying to prevent “tangled representations,” you’ll likely end up changing the goodness function—this paper is directly about that design space.
    
-   **Gong et al. (2025): “FAUST: Reshaping the Forward-Forward Algorithm with a Similarity-Based Objective”**
    Connects FF to similarity-learning losses; helpful if your plan is to enforce more global/consistent representational geometry.
    

* * *

## “Has anyone tried Transformers with FF?” (and what kind)

### Vision Transformers (closest “clean” Transformer-architecture FF work)

-   **Aghagolzadeh & Ezoji (2025): “Contrastive Forward-Forward: A Training Algorithm of Vision Transformer”**
    One of the more direct “FF-style training applied to a Transformer family architecture” papers (ViT).
    

### Transformers for NLP classification (not autoregressive generation, but NLP + Transformers + FF)

-   **Kim & Yoo (2025): “Replacing Backpropagation with the Forward-Forward (FF) Algorithm in Transformer Models…”**
    Proposes FF-based training in a Transformer setting and reports experiments on **IMDB sentiment** (classification). This is relevant if you want to see how someone mapped FF onto attention blocks in practice—but it is not an end-to-end autoregressive LM paper.
    

* * *

## Autoregressive / causal language models with FF: what exists as of 2026-03-02

### What I found (published / citable)

I did **not** find a widely cited, mainstream paper showing **end-to-end pretraining of a decoder-only autoregressive language model** (GPT-like next-token LM) using **pure Hinton-style FF** at meaningful scale.

What _is_ out there falls into three buckets:

1.  **Transformer + FF, but on classification tasks** (e.g., IMDB sentiment).
    
2.  **Patent literature** explicitly discussing forward-forward training applied to tasks including _masked_ or _causal language modeling_\-type setups (useful for ideas about how to construct positives/negatives for language), but it’s not a peer-reviewed scaling result.
    
3.  **Community / repo-level attempts** at “mini-GPT with forward-forward,” but not established as a solid research baseline. The Hugging Face thread explicitly notes the lack of generative LM examples and points to experimental repos.
    

### Closest “similar, forward-only” methods actually used on language models (not FF, but highly relevant)

If your real requirement is “no backprop / no backward pass” (or “avoid global credit assignment”), these are the most relevant LM-adjacent works:

-   **Malladi et al. (NeurIPS 2023): “Fine-Tuning Language Models with Just Forward Passes” (MeZO)**
    This is a _big deal_ practically: forward-pass-only optimization for fine-tuning, evaluated on both masked and autoregressive LMs. It’s not FF; it’s zeroth-order optimization. But it’s the most concrete “LMs without backprop” result line I found.
    
-   **Pau & Aymone (2024): “Forward Learning of Large Language Models…” (PEPITA / MEMPEPITA on Transformer LLMs)**
    Again, not FF, but explicitly about forward-learning rules on Transformer-based LLMs (and discusses CLM/MLM token-level error setups).
    

* * *

## “Do you train each layer separately in order or what?” (short, concrete answer)

-   **Vanilla Hinton-FF** is **greedy layer-wise**: train layer 1 with its local objective (pos/neg), then freeze it and train layer 2, etc.
    
-   The main reason people moved beyond vanilla FF is exactly what you’re pointing at: **greedy, isolated layer training tends to harm cross-layer coordination**. The **Layer Collaboration** paper is explicitly about addressing that.
    
-   Many “FF-like” successors can be interpreted as: keep locality, but introduce **inter-layer signals** (collaboration) and/or **better contrastive setups** (better negatives) so deeper stacks don’t become dysfunctional.
    

* * *

## If your target is specifically “autoregressive LM with FF”: what to look for in papers (based on what exists)

Given the current literature, the practical path is usually:

-   Start from **FF improvements that fix negatives + collaboration** (Layer Collaboration, SCFF, goodness-function benchmarking).
    
-   Use the **Transformer implementations that exist (even if classification)** to see where people place local losses inside attention/FFN blocks and how they handle normalization/residuals.
    
-   Treat **MeZO / PEPITA** as “reference points” for what _actually works today_ on language models without standard backprop, even if they are not FF.
    
-   Mine the **patent** for explicit suggestions on applying forward-forward training to masked/causal language modeling setups (positive/negative construction ideas).
    

If you want, I can convert the above into a tighter “reading order” (5–7 items) depending on whether your immediate goal is (a) a working prototype on a small GPT-like model, or (b) inventing a better FF objective that preserves global structure in deep attention stacks.

## User

review all this lit to figure out how we can train an autoregressive generative language model using FF or somethign very similar, and figure out why no one has (and check experimental nano gpt projects to learn more)

## Assistant

Thought for 20m 13s

## Assistant

## What “train each layer separately” means in Forward-Forward

In Hinton’s Forward-Forward (FF), _each layer/block has its own local objective_ (“goodness” high on positives, low on negatives), and you typically **update layers using only their local loss** while **stopping gradients between layers** (detach the input to the layer you’re updating).

So in practice you do something like:

-   Run a **positive** forward pass (real example) and a **negative** forward pass (corrupted example).
    
-   For **layer i**, compute a local loss from its activations on those two passes and update **only layer i** (no cross-layer backprop).
    
-   Move to layer i+1, etc.
    

The big caveat is that “vanilla” FF can have poor deep-layer usefulness: the **first layer alone can outperform deeper stacks**, indicating weak inter-layer collaboration.
That’s why the more relevant question for transformers is not only “separately or together?”, but **how to preserve collaboration / global behavior without end-to-end backprop**.

## The most relevant papers to review (prioritized)

### Core FF + the sequence/LM clue (and warning)

-   **Hinton (2022): “The Forward-Forward Algorithm: Some Preliminary Investigations.”**
    Defines FF, why normalization is needed to prevent trivial solutions, and explicitly flags sequential learning as a target.
    Critically: Hinton reports an early draft result on **next-character prediction** (predict next character from a window) that he later **could not replicate** and suspected was due to a bug.
    This is one of the strongest “why isn’t there an FF GPT already?” signals.
    

### Fixing the “deep networks don’t collaborate” failure mode

-   **Lorberbom et al. (2023/2024): “Layer Collaboration in the Forward-Forward Algorithm.”**
    Shows FF layers don’t effectively communicate; deeper layers can hurt; proposes **Collaborative Forward-Forward** by using a **global goodness term** (sum of other layers’ goodness) while _not_ backpropagating through those other layers (they use a frozen copy / stop-grad idea).
    For transformers, this is directly relevant to “maintain global characteristics.”
    

### Making FF train deeper/stabler (goodness + normalization + residuals)

-   **Sun et al. (ICLR 2025): “DeeperForward.”**
    Diagnoses squared-norm goodness + normalization issues in deep FF and proposes **mean goodness + layer norm**, plus residual-structure adaptations to help deep learning in FF settings.
    Even though it’s CNN-focused, the _design lessons_ (goodness definition, normalization, residual structure role) map to transformers.
    

### Moving beyond “hand-crafted negatives”

-   **Chen et al. (Nature Communications 2025): “Self-Contrastive Forward-Forward.”**
    A major direction: reduce dependence on external negative sampling and extend FF-style training to broader regimes including time-series/unsupervised settings.
    

### “FF but with an explicit generative circuit”

-   **Ororbia & Mali (2023): “The Predictive Forward-Forward Algorithm.”**
    Proposes a recurrent system that jointly learns a **representation circuit + directed generative circuit**, and reports classification/reconstruction/synthesis without backprop.
    This is highly relevant if your plan is “use a generator to produce negatives” (or to avoid manual corruptions).
    

### First steps in NLP (not language modeling yet)

-   **Gandhi et al. (2023): “Extending the Forward Forward Algorithm.”**
    Replicates MNIST and extends FF to **IMDb sentiment** (classification), explicitly positioning it as an early step beyond vision.
    

## Has anyone trained an autoregressive GPT-like LM with FF?

I did not find a peer-reviewed result demonstrating a competitive **causal/autoregressive transformer language model** trained end-to-end with Hinton-style FF as the primary learning rule.

What exists is:

-   **Hinton’s own attempted next-character prediction** in an early draft that he later could not reproduce.
    
-   Community discussion where people explicitly say they’re attempting “mini-GPT” with FF, and the referenced repos were “not working” / hard to find; one “forward\_forward\_gpt” repo is referenced but is no longer accessible at the linked location.
    
-   NLP extensions that are **classification** (IMDb) rather than **autoregressive generation**.
    

## Why “FF GPT” hasn’t happened (the concrete blockers)

### 1) The label space is enormous (vocab), and FF-style inference/training can be _label-iterative_

Classical FF classification often evaluates “goodness” for each possible label and picks the max, which scales poorly when the label set is large.
For language modeling, the “label” is the next token (often 50k–200k). Doing multiple forward passes per candidate token is a non-starter without a redesign.

### 2) Deep-layer usefulness/collaboration is already fragile in FF

Layer Collaboration shows vanilla FF can have the pathological behavior where early layers dominate and deeper layers don’t help, because layers are “not aware” of later layers.
Transformers depend on depth; if deeper blocks don’t reliably add value, you won’t get GPT-like scaling.

### 3) Negatives are easy to define for tokens, but hard to make _informative_ at scale

Random wrong-token negatives can be too easy; “hard negatives” require either:

-   a strong proposal model (which is usually trained with backprop), or
    
-   additional machinery (self-contrastive schemes, learned generators, etc.).
    

### 4) Sequential learning instability shows up early

Hinton’s inability to replicate the character-prediction result suggests that even toy LM setups can be unstable depending on goodness, learning rates, scheduling of positive/negative phases, etc.

### 5) Engineering reality: the best “nanoGPT-style” codebases are backprop-first

Useful for experimentation, but they won’t give you an FF LM “for free.” Examples like Transformer-Sandbox and cleanGPT are explicitly forks/research frameworks around nanoGPT-style training and architecture exploration, not FF training.

## A practical design to train an autoregressive generative LM with FF-like learning

If the goal is **autoregressive generation**, you must avoid “one forward pass per vocab item.”

The key move is to treat FF “positive vs negative” as **contrastive learning of (context, next-token)** with _sampled negatives_, so your compute per token is O(K) negatives, not O(|V|).

### Core formulation: conditional energy / goodness for next token

For each position _t_ in a sequence:

-   Context representation at layer i: hi,t\=Blocki(detach(hi−1))h\_{i,t} = \\mathrm{Block}\_i(\\text{detach}(h\_{i-1}))hi,t​\=Blocki​(detach(hi−1​))
    
-   Token embedding table: E\[w\]E\[w\]E\[w\]
    

Define a **goodness** for the correct next token w+\=xt+1w^+=x\_{t+1}w+\=xt+1​ and sampled negatives wk−∼q(⋅)w^-\_k\\sim q(\\cdot)wk−​∼q(⋅):

-   Simple and scalable goodness:
    
    gi(hi,t,w)\=⟨LN(hi,t),LN(E\[w\])⟩g\_i(h\_{i,t}, w) = \\langle \\mathrm{LN}(h\_{i,t}), \\mathrm{LN}(E\[w\])\\ranglegi​(hi,t​,w)\=⟨LN(hi,t​),LN(E\[w\])⟩
    
    (dot-product similarity; you can also use cosine)
    

Train each layer i with a local NCE/logistic loss:

-   Increase gi(⋅,w+)g\_i(\\cdot, w^+)gi​(⋅,w+)
    
-   Decrease gi(⋅,wk−)g\_i(\\cdot, w^-\_k)gi​(⋅,wk−​)
    

This is “FF-like” because it’s **purely forward signals + local discrimination between positives/negatives**, matching FF’s stated link to contrastive learning/NCE ideas.

### How to keep it FF (no cross-layer backprop)

For each transformer block i, you:

1.  Compute its output using **detached** input states (stop-grad boundary).
    
2.  Compute its local contrastive loss using its output only.
    
3.  Backprop **only within the block** to update its parameters.
    

This matches the “train each layer locally” spirit.

### How to preserve _global characteristics_ (avoid “layer collapse”)

Use **Collaborative Forward-Forward** ideas:

-   When updating layer i, include a **global goodness offset** computed from other layers’ goodness but treated as constant (stop-grad / frozen copy).
    
-   Operationally, you maintain a “shadow” copy f^\\hat ff^​ of other layers used only to compute the offset γ\\gammaγ, without taking gradients through it.
    

This is a direct mechanism to make early layers “care” about downstream behavior without end-to-end backprop.

### Goodness choice and normalization (important in deep nets)

Vanilla FF’s squared-norm goodness can create scaling/dead-unit issues; DeeperForward specifically argues for:

-   **layer normalization** for feature scaling
    
-   **mean goodness** to decouple representation from the goodness scalar and avoid dead-neuron update problems
    

For transformers (already LN-heavy), the takeaway is:

-   Your goodness should not be trivially encoded in magnitude that later layers can exploit.
    
-   If you rely on magnitude-based goodness, ensure LN placement truly removes it (and doesn’t leak it through residual pathways).
    

DeeperForward also highlights that residual structures help integrate features across levels in FF-style deep training.
For transformers, that suggests being very deliberate about where goodness is measured (pre/post residual, pre/post LN).

### Negative sampling strategies that actually work for LMs

You need negatives that are _not_ trivially separable.

A workable progression:

1.  **Unigram noise** negatives (word2vec-style) for a first working baseline.
    
2.  Move to **in-batch negatives** (other targets in the batch).
    
3.  Add **proposal-based hard negatives**: sample from a cheap auxiliary head (could be shallow, even trained with standard methods) and use top-k as negatives.
    This keeps training O(K) while making negatives informative.
    

### Inference: how do you actually generate text?

Once you train a final-layer representation hL,th\_{L,t}hL,t​, you can produce logits by:

ℓ\[w\]\=⟨hL,t,E\[w\]⟩\\ell\[w\] = \\langle h\_{L,t}, E\[w\]\\rangleℓ\[w\]\=⟨hL,t​,E\[w\]⟩

and sample via softmax like a standard LM.

If you also want FF’s “global goodness” to matter at inference:

-   Use **two-stage decoding**:
    
    1.  propose top-k tokens by dot-product logits
        
    2.  re-rank with a more expensive “global goodness” score that includes multiple layers
        This avoids O(|V|) FF scoring.
        

## Recommended experimental path (nanoGPT-scale, but FF-style)

### Phase 0: prove the mechanics on tiny char-level LM

-   Dataset: Tiny Shakespeare (char-level)
    
-   Model: 2–4 transformer blocks, very small d\_model
    
-   Objective: local contrastive next-char (K negatives)
    
-   Metric: bits-per-character
    

This aligns with Hinton’s own reported-but-not-reproduced direction (next-character prediction).

### Phase 1: fix deep-layer collapse early

Add layer-collaboration offset γ\\gammaγ (frozen-copy style) and compare:

-   vanilla per-layer loss vs collaborative loss
    

### Phase 2: stabilize deep training

Adopt the “goodness design + normalization” lessons from DeeperForward:

-   treat goodness in a way that doesn’t break scaling across layers
    
-   ensure LN placement prevents trivial “magnitude channel” solutions
    

### Phase 3: move to subword vocab and real LM

-   Switch tokenizer, larger vocab
    
-   Use proposal-based negatives to keep compute manageable
    
-   Evaluate perplexity and sample quality
    

## What I would conclude from the literature, in one sentence

The literature strongly suggests you **should not** train a transformer LM with naive, strictly greedy FF; you likely need (i) **collaboration/global-goodness tricks** to prevent deep-layer uselessness, (ii) **a scalable goodness definition/normalization scheme** to keep deep learning stable, and (iii) an **NCE-style sampled-negative formulation** so “label iteration over vocab” is avoided.

If you want, I can convert the above into a concrete block-by-block training pseudocode for a GPT-style transformer (including where to detach, what tensors the local loss uses, and how to structure the optimizer steps per block).

## User

should we be tweaking FF to measure goodness at a distribution of logits level? instead of a classifier. or is that the same as an already existing different approach

## Assistant

Thought for 8m 25s

## Assistant

In an autoregressive LM, “a classifier” and “a distribution over logits” are basically the same object: at each position you are doing **multiclass classification over the vocabulary**. The real question is which _training objective_ you attach to those logits and whether you keep the **FF constraint** (purely local learning, no end-to-end gradient flow).

There are two distinct interpretations of what you’re proposing.

## 1) Full distribution-level goodness (cross-entropy / KL on logits)

If by “goodness at a distribution level” you mean something like:

-   **Goodness** g\=−CE(softmax(ℓ),y)g = -\\mathrm{CE}(\\text{softmax}(\\ell), y)g\=−CE(softmax(ℓ),y), or
    
-   **Goodness** g\=−KL(ptarget ∥ softmax(ℓ))g = -\\mathrm{KL}(p\_\\text{target}\\,\\|\\,\\text{softmax}(\\ell))g\=−KL(ptarget​∥softmax(ℓ))
    

then you’re just using a **proper scoring rule** on the predicted distribution.

That is not uniquely “Forward–Forward”; it’s essentially:

-   **Deep supervision / auxiliary heads** (losses applied at intermediate layers), e.g. Deeply-Supervised Nets.
    
-   Or, if you _stop gradients between blocks and train each block with its own local head_, it falls under **local-learning / local error signals** and related decoupled training families (local classifiers per layer, no global backprop to earlier layers).
    
-   Closely adjacent to “decoupled greedy learning” style decompositions (again: module-local objectives rather than one global one).
    

So: **yes**, “distribution-level goodness” largely collapses into an already-existing approach family (deep supervision / local losses). The novelty would be your _specific way of enforcing locality, collaboration, and negative construction_, not the fact you’re scoring distributions.

Practical blocker: full softmax CE requires summing over ∣V∣|V|∣V∣, which is expensive for big vocabs. Classic LM work uses sampling/importance approximations to avoid full normalization.

## 2) FF-style goodness on _token candidates_ (NCE / negative sampling / sampled softmax)

If instead you mean: “Don’t treat it as binary ‘is this input real?’; treat goodness as the **logit/energy score** of (context, candidate next token) and learn by contrasting true vs false tokens,” then you’re squarely in the territory FF already points toward.

Hinton explicitly frames FF as inspired by **Noise-Contrastive Estimation (NCE)**.
NCE itself is: train by **logistic regression discriminating data from noise**, rather than computing a full normalized likelihood.

For an LM at position ttt:

-   context representation: hth\_tht​ (from some layer/block)
    
-   token embedding: E\[w\]E\[w\]E\[w\]
    
-   score/logit (“goodness” per token): s(w)\=⟨ht,E\[w\]⟩s(w)=\\langle h\_t, E\[w\]\\rangles(w)\=⟨ht​,E\[w\]⟩
    

Then train with:

-   positive token w+\=xt+1w^+=x\_{t+1}w+\=xt+1​
    
-   negatives w1−,…,wK−∼q(⋅)w^-\_1,\\dots,w^-\_K \\sim q(\\cdot)w1−​,…,wK−​∼q(⋅)
    

A standard NCE/negative-sampling objective is:

L\=−log⁡σ(s(w+))−∑k\=1Klog⁡σ(−s(wk−))\\mathcal{L} = -\\log \\sigma(s(w^+)) - \\sum\_{k=1}^K \\log \\sigma(-s(w^-\_k))L\=−logσ(s(w+))−k\=1∑K​logσ(−s(wk−​))

This is the same basic shape as widely used negative-sampling objectives in NLP.

This variant is much more “FF-like” because:

-   it’s still **positive vs negative**
    
-   you can implement it with **two forward evaluations per sample** (or one forward for the context + multiple candidate dot-products)
    
-   you avoid the ∣V∣|V|∣V∣ softmax
    

If you want the _distribution_ eventually for generation, you can still decode with a softmax over logits at inference; the training objective just avoids full normalization.

## Which one should you do for an autoregressive FF-like LM?

If the goal is “FF or something very similar” **and** “scales to large vocab,” the more natural move is:

-   Use **candidate-level goodness** (token energy/logit) + **sampled negatives** (NCE/negative sampling / sampled softmax family).
    

If you instead do full-distribution CE/KL at each layer:

-   you are effectively doing **local losses / deep supervision**, not classic FF.
    
-   it’s viable, but the “FF-ness” is mostly in your **gradient stopping between blocks** and your **module-local objectives**, not in the loss form itself.
    

## The important nuance for “why this isn’t just ‘a classifier’”

A next-token softmax head _is_ a classifier. The distinction is:

-   FF classification (original) often relies on **evaluating goodness per label** by “clamping” the label and comparing goodness across labels. That is infeasible for vocab-size labels.
    
-   Switching to logits/energies with **sampling** is how language modeling has historically avoided the “evaluate every label” trap.
    

## Bottom line

-   **Distribution-level goodness** = you’re mostly rediscovering **cross-entropy / KL** plus **deep supervision / local losses**.
    
-   **Candidate-logit goodness with negatives** = you’re mostly rediscovering **NCE / negative sampling / sampled-softmax-style** training, which is already tightly aligned with FF’s “positive vs negative” roots.
    

What could be genuinely new is not “goodness on logits,” but:

-   enforcing **strict block-local updates** in a transformer (stop-grad boundaries),
    
-   adding **inter-layer collaboration signals** without end-to-end backprop,
    
-   and designing **hard negatives/proposals** that make the model learn global sequence constraints rather than local token heuristics.
    

If you want, I can give a concrete “FF-like GPT training objective” that is (a) token-energy NCE, (b) block-local, (c) compatible with tied embeddings, and (d) uses a proposal mechanism for hard negatives so it doesn’t collapse into unigram discrimination.

## User

so for hard negatives, Hinton tried using the model's own predictions and that didn't work? or we're not sure

## Assistant

Thought for 7m 45s

## Assistant

Hinton did propose (and in one draft explicitly _demonstrated_) using the model’s own predictions to generate negatives for a toy next-character task — but in the later “final” PDF he says he could not reproduce the key result and suspects it was a bug. So it’s not a clean “it didn’t work”; it’s “he reported it working, then later wasn’t sure / couldn’t replicate.”

What the paper(s) say, precisely:

-   In the core FF setup he explicitly allows negatives to be **predicted by the net itself** (via top‑down connections) or supplied externally.
    
-   In an earlier version of the paper (the one that includes a full “predict the next character” section), he describes constructing negatives by taking a real 10‑character context and replacing the last character with **a prediction from the model** (“teacher forcing” style). He also describes generating **entire negative sequences** from the predictive model starting from an initial real context, and says this worked “very nearly as well” as teacher forcing.
    
-   In the later “FFXfinal” PDF, he revisits that as a “sleep/offline negatives” motivation and says: he previously reported doing many positive updates followed by many negative updates **generated by the net itself** with little loss, but he has been **unable to replicate** and suspects a **bug**. He adds that, with squared-activity goodness, separating positive/negative phases seems to require very low learning rate and extremely high momentum, and that maybe a different goodness function could make phase separation work (open question).
    

So the most accurate summary is:

-   **Yes**, he tried (in a toy sequence setting) using model-generated predictions as negatives.
    
-   **He initially reported it worked**, including the idea that negatives could be generated “offline/sleep.”
    
-   **Later he could not replicate the result and suspected a bug**, so we don’t have a reliable, reproduced claim that this works as stated.
    

Also note: his “couldn’t replicate” comment in the final version is tied to _separating positive and negative phases into large blocks of updates_ (wake/sleep style), not necessarily to the basic idea “use model predictions as hard negatives” in an online interleaved way.

## User

did he share code for that? low learning and high momentum may be ok, or we could play with diff goodness funcs or global sync like we discussed (unless he was already doing that)

## Assistant

Thought for 12m 19s

## Assistant

### Did Hinton share code for the sequence / “sleep-phase” hard-negative experiment?

Not that I can find.

What he _did_ share publicly is:

-   **`ffcode.zip` (Matlab)**: “matlab code for the supervised version of FF with the first 10 pixels being the labels,” linked from his homepage, plus a short **README.txt** describing how to run it on MNIST.
    

That release is for the **supervised MNIST setup** (labels embedded into the input), not the **next-character prediction / offline negative generation** experiment.

In the final paper, he explicitly says the next-character “sleep-phase” result he reported in an early draft was **not reproducible** and he suspects it was due to a **bug**.
Given that, it’s unsurprising there’s no “official” code drop for the sequence case.

### What exactly was the “low learning rate / high momentum” about?

It’s specifically about **separating positive and negative learning into long blocks**, i.e.:

-   “thousands of weight updates on positive data”
    followed by
    
-   “thousands on negative data” (generated by the net)
    

He says that with **sum-of-squared-activities goodness**, that long-block alternation “only works if the learning rate is very low and the momentum is extremely high,” and that a different goodness function _might_ allow phase separation.

So two important implications:

1.  That warning is about the **wake/sleep separation regime**, not necessarily about **interleaving pos/neg within each minibatch/step** (which is closer to standard contrastive/NCE practice).
    
2.  It’s also tied to that specific goodness choice (squared activity), and he explicitly flags “different goodness function” as the likely lever.
    

### Was he “already doing global sync” like we discussed?

Depends what you mean by “global sync.”

**He _was_ doing synchronous updates of hidden states in a recurrent/iterative version of FF** (his “treat a static image as a boring video” idea). He reports that synchronous updates of _all hidden layers_ with damping actually learned slightly better than alternating layer updates.

That is “global synchronization” in the sense of **global, simultaneous state updates during iterative inference**.

But **he was not doing “global synchronization” of learning via a cross-layer coordination term in the loss** (the kind of “layer collaboration / global goodness injection” we discussed earlier). In the FF paper(s), the main coordination mechanism is architectural/iterative (the recurrent framing), not an explicit multi-layer coupling term in the objective.

### Did he already use “model-based hard negatives”?

Yes, in two different senses (but with caveats):

1.  **Sequence draft (FFA13)**: he describes negatives where the last character in a window is replaced by **a prediction from the previous 10 chars** (“teacher forcing”), and also negatives **entirely generated by the predictive model** starting from the initial real context, which he says worked nearly as well _in that draft_.
    
2.  **Final paper (FFXfinal)**: he later says he could not replicate the sleep-phase block-alternation result and suspects a bug; plus it only worked under extreme optimizer settings with squared-activity goodness.
    

Also, in the recurrent MNIST setup he generates “harder” negatives by sampling **incorrect labels in proportion to the model’s predicted probabilities** (instead of uniform wrong labels), explicitly noting it makes training more efficient.

### Practical takeaways for your direction

-   **Low LR + very high momentum** is plausibly “OK” _if_ you really want the long wake/sleep separation, but Hinton’s final writeup strongly suggests that regime is currently brittle and goodness-dependent.
    
-   A more robust path is usually: **don’t separate phases into long blocks**; instead **interleave pos/neg every step** (or every minibatch) so you’re not relying on momentum to average two opposing objectives.
    
-   Exploring **different goodness functions** is explicitly endorsed by Hinton as the likely way to make separation feasible.
    
-   If you want “global coherence” without backprop, Hinton’s paper gives one mechanism (iterative/recurrent inference with synchronous updates) , while later work (not Hinton’s) gives another mechanism (explicit layer-collaboration terms).
    

If you want, I can propose 2–3 concrete “goodness” definitions for an autoregressive LM that (a) avoid vocab-wide scoring per step and (b) are compatible with either interleaved pos/neg or a stabilized wake/sleep separation.

## User

yes

## Assistant

Thought for 9m 32s

## Assistant

Below are three concrete “goodness” definitions for an autoregressive LM that (a) never require scoring the full vocabulary and (b) can be implemented in an FF‑like, block‑local way (detach between blocks). I’ll also include how each interacts with “hard negatives,” and how to add the global synchronization term we discussed without turning it into end‑to‑end backprop.

I’ll use this notation:

-   Tokens: x1,…,xTx\_1,\\dots,x\_Tx1​,…,xT​. Target at position ttt: yt\=xt+1y\_t = x\_{t+1}yt​\=xt+1​.
    
-   Transformer blocks l\=1..Ll=1..Ll\=1..L. Block output hidden state at position ttt: hl,t∈Rdh\_{l,t}\\in\\mathbb{R}^dhl,t​∈Rd.
    
-   Token embedding matrix E∈R∣V∣×dE\\in\\mathbb{R}^{|V|\\times d}E∈R∣V∣×d (often tied to output).
    
-   Candidate set at position ttt: Ct\={yt}∪{wt,1−,…,wt,K−}C\_t = \\{y\_t\\}\\cup\\{w^-\_{t,1},\\dots,w^-\_{t,K}\\}Ct​\={yt​}∪{wt,1−​,…,wt,K−​} with small KKK (e.g. 16–256).
    
-   “Score” (logit/energy) from layer lll: sl,t(w)s\_{l,t}(w)sl,t​(w). You compute this only for w∈Ctw\\in C\_tw∈Ct​ via embedding gathers.
    

A standard choice that avoids scale hacks:

sl,t(w)\=⟨LN(hl,t), LN(E\[w\])⟩τs\_{l,t}(w) = \\frac{\\langle \\mathrm{LN}(h\_{l,t}),\\, \\mathrm{LN}(E\[w\])\\rangle}{\\tau}sl,t​(w)\=τ⟨LN(hl,t​),LN(E\[w\])⟩​

(or sl,t(w)\=E\[w\]⊤LN(hl,t)s\_{l,t}(w)= E\[w\]^\\top \\mathrm{LN}(h\_{l,t})sl,t​(w)\=E\[w\]⊤LN(hl,t​) if you don’t want LN on embeddings).

* * *

## Goodness 1: Pairwise margin “energy wins” (no softmax, very FF‑like)

**Goodness idea:** the correct token should score higher than negatives by a margin. This is the most direct FF analogue: “positive > negative” locally.

For each negative w−∈Ct∖{yt}w^- \\in C\_t\\setminus\\{y\_t\\}w−∈Ct​∖{yt​}, define pairwise margin:

Δl,t(w−)\=sl,t(yt)−sl,t(w−)\\Delta\_{l,t}(w^-) = s\_{l,t}(y\_t) - s\_{l,t}(w^-)Δl,t​(w−)\=sl,t​(yt​)−sl,t​(w−)

Layer‑local loss (smooth hinge / logistic margin):

Ll,t(1)\=∑w−softplus(m−Δl,t(w−))\\mathcal{L}^{(1)}\_{l,t} = \\sum\_{w^-} \\mathrm{softplus}(m - \\Delta\_{l,t}(w^-))Ll,t(1)​\=w−∑​softplus(m−Δl,t​(w−))

-   mmm is a margin (often depth‑scheduled: smaller for early blocks, larger for later blocks).
    

**Why it’s attractive**

-   No normalization, no full distribution.
    
-   Directly matches “goodness should separate positives and negatives.”
    
-   Stable with small candidate sets.
    

**Hard negatives**

-   This objective benefits strongly from _hard_ negatives because easy negatives saturate.
    
-   Good default: for each position ttt, pick negatives as a mixture:
    
    -   top‑k1k\_1k1​ tokens from a proposal distribution (see below)
        
    -   -   k2k\_2k2​ in‑batch other targets (other yt′y\_{t'}yt′​ in the batch)
            
    -   -   k3k\_3k3​ random/unigram tokens for diversity
            (so you don’t overfit to one proposal mechanism)
            

**Wake/sleep compatibility**

-   Works fine interleaved (pos/neg in same step).
    
-   Also compatible with “sleep”: negatives can be model‑generated continuations; you then compare correct token(s) vs tokens that the sleep generator proposes (or appears in a negative continuation).
    

* * *

## Goodness 2: Candidate‑set softmax cross‑entropy (distribution‑level, but cheap)

This is the cleanest way to do “distribution‑level goodness” without ∣V∣|V|∣V∣.

Define a restricted softmax over the candidate set CtC\_tCt​:

pl,t(w∣Ct)\=exp⁡(sl,t(w))∑u∈Ctexp⁡(sl,t(u))p\_{l,t}(w\\mid C\_t) = \\frac{\\exp(s\_{l,t}(w))}{\\sum\_{u\\in C\_t}\\exp(s\_{l,t}(u))}pl,t​(w∣Ct​)\=∑u∈Ct​​exp(sl,t​(u))exp(sl,t​(w))​

Local loss:

Ll,t(2)\=−log⁡pl,t(yt∣Ct)\\mathcal{L}^{(2)}\_{l,t} = -\\log p\_{l,t}(y\_t\\mid C\_t)Ll,t(2)​\=−logpl,t​(yt​∣Ct​)

You can define “goodness” as gl,t\=−Ll,t(2)g\_{l,t}=-\\mathcal{L}^{(2)}\_{l,t}gl,t​\=−Ll,t(2)​ if you want to keep FF language.

**What this is (and isn’t)**

-   It is “classifier-like,” but only over K+1K+1K+1 candidates, not the full vocab.
    
-   It’s effectively sampled-softmax / contrastive CE. The novelty is not CE itself; it’s using it as a **strictly local objective per block** with detach boundaries, plus your negative strategy and layer collaboration.
    

**Hard negatives**

-   Same mixture strategy as above.
    
-   This objective is somewhat more forgiving than pairwise margin if some negatives are easy (because normalization still shapes gradients).
    

**Wake/sleep compatibility**

-   Interleaved training works well.
    
-   For “sleep,” you can populate CtC\_tCt​ using top‑kkk from the sleep generator at that position (plus extras).
    

**Optional correction**

-   If you care about unbiased likelihood estimation, you can add importance corrections based on the negative sampling distribution q(w)q(w)q(w). Practically, many systems just treat it as a contrastive objective and rely on good negatives.
    

* * *

## Goodness 3: Sequence/continuation goodness (global coherence beyond one token)

Token‑level training can still yield locally good but globally incoherent behavior. A direct way to encode “global characteristics” is to define goodness over a _short continuation_ window.

Pick a short horizon HHH (e.g. 8–64 tokens). For each start position ttt, define the continuation:

-   Positive continuation: Yt+\=(xt+1,…,xt+H)Y^+\_{t} = (x\_{t+1},\\dots,x\_{t+H})Yt+​\=(xt+1​,…,xt+H​)
    
-   Negative continuation: Yt−Y^-\_{t}Yt−​ created by:
    
    -   model-generated continuation (“sleep”)
        
    -   or splice/swap/shuffle spans to break global structure while keeping local plausibility
        

Define layer‑wise continuation score:

Gl(t,Y)\=∑j\=1Hsl,t+j−1(Y\[j\])G\_l(t, Y) = \\sum\_{j=1}^{H} s\_{l,t+j-1}\\big(Y\[j\]\\big)Gl​(t,Y)\=j\=1∑H​sl,t+j−1​(Y\[j\])

(where sl,ts\_{l,t}sl,t​ is still computed cheaply using embedding gathers; for negatives you only need to score the actual negative tokens present in Y−Y^-Y−, not the whole vocab.)

Now do a contrastive “positive continuation beats negative continuation” loss:

Ll,t(3)\=softplus(m−(Gl(t,Yt+)−Gl(t,Yt−)))\\mathcal{L}^{(3)}\_{l,t} = \\mathrm{softplus}\\big(m - (G\_l(t,Y^+\_t) - G\_l(t,Y^-\_t))\\big)Ll,t(3)​\=softplus(m−(Gl​(t,Yt+​)−Gl​(t,Yt−​)))

Optionally with multiple negatives per ttt.

**Why it’s attractive**

-   Forces multi-token, cross-time constraints.
    
-   Makes “sleep-phase” style negatives conceptually natural: the model generates coherent-looking continuations that are still wrong relative to the data’s continuation.
    

**Hard negatives**

-   Model-generated continuations are naturally hard (especially if generated from a stronger proposal head or a deeper layer).
    
-   You can mix in “structural” negatives (span swap, entity swap) to explicitly target global consistency failures.
    

**Wake/sleep compatibility**

-   This is the most directly compatible with wake/sleep: wake uses real continuations; sleep generates negative continuations.
    

* * *

## How to implement these in a block-local FF-like Transformer

A practical block-local training skeleton (for any of the above losses):

1.  Run embeddings once to get h0,:h\_{0,:}h0,:​.
    
2.  For each block lll:
    
    -   Feed **detached** input to the block: hl,:\=Blockl(stopgrad(hl−1,:))h\_{l,:} = \\mathrm{Block}\_l(\\mathrm{stopgrad}(h\_{l-1,:}))hl,:​\=Blockl​(stopgrad(hl−1,:​))
        
    -   Compute the block’s local loss ∑tLl,t\\sum\_t \\mathcal{L}\_{l,t}∑t​Ll,t​ using hl,th\_{l,t}hl,t​ only.
        
3.  Sum losses across blocks (or step each block separately).
    
4.  Backward once; because inputs were detached, gradients stay inside each block.
    

Key point: candidate scoring never needs full vocab; it’s gather dot-products against E\[w\]E\[w\]E\[w\] for w∈Ctw\\in C\_tw∈Ct​.

* * *

## Proposal mechanism for hard negatives (critical)

Hard negatives are easiest if you have a _proposal distribution_ πt(w)\\pi\_t(w)πt​(w) that’s already somewhat aligned with the task.

Three practical proposals (in increasing “FF purity risk”):

1.  **Final-layer proposal, stop-grad**
    Use the current model’s _top-layer_ logits to select top‑kkk tokens for each ttt, but treat those selections and scores as stop-grad when training lower blocks.
    
    -   Cheap, very effective.
        
    -   Risk: if the final layer is bad early, negatives are not hard.
        
2.  **Deeper-layer proposal for shallower blocks**
    When training block lll, sample negatives from block l+ql+ql+q (with stop-grad). This encourages a coarse-to-fine hierarchy.
    
    -   Helps keep deeper layers relevant.
        
3.  **External lightweight proposal (frozen or separately trained)**
    E.g., a small n‑gram, a tiny backprop LM, or even a cached KNN proposal. This breaks circularity in early training.
    
    -   “Less pure,” but often necessary for bootstrapping.
        

In all cases: mix proposal negatives with in-batch and random for diversity.

* * *

## Add “global sync” without end-to-end backprop

Two low-cost “global coordination” add-ons that do not require full vocab:

### A) Cross-layer distribution alignment on candidate sets

Compute candidate-set distributions pl,t(⋅∣Ct)p\_{l,t}(\\cdot\\mid C\_t)pl,t​(⋅∣Ct​) and align them to a reference (typically the top layer) with stop-grad:

Ll,tsync\=KL(stopgrad(pL,t(⋅∣Ct)) ∥ pl,t(⋅∣Ct))\\mathcal{L}^{sync}\_{l,t} = \\mathrm{KL}\\left(\\mathrm{stopgrad}\\left(p\_{L,t}(\\cdot\\mid C\_t)\\right)\\ \\|\\ p\_{l,t}(\\cdot\\mid C\_t)\\right)Ll,tsync​\=KL(stopgrad(pL,t​(⋅∣Ct​)) ∥ pl,t​(⋅∣Ct​))

-   Only computed on CtC\_tCt​, so it’s cheap.
    
-   Encourages global agreement while each block still trains locally.
    

### B) Global goodness offset (“collaborative” scalar)

If you’re using a scalar continuation goodness GlG\_lGl​ (Goodness 3) or even token-level goodness summaries, you can add a stop-grad global term:

g~l\=gl+α⋅stopgrad(∑r≠lgr)\\tilde{g}\_l = g\_l + \\alpha \\cdot \\mathrm{stopgrad}\\Big(\\sum\_{r\\neq l} g\_r\\Big)g~​l​\=gl​+α⋅stopgrad(r\=l∑​gr​)

and train with g~l\\tilde g\_lg~​l​. This is the scalar version of “global coordination” and tends to reduce the “only early layers matter” failure mode.

* * *

## Recommendation: what to try first

If you want the fastest path to “works on nanoGPT-scale”:

1.  **Goodness 2 (candidate-set CE)** + mixed hard negatives + detach boundaries
    

-   It most closely matches standard LM training signals while avoiding full softmax.
    

2.  Add **sync KL on candidate sets** (align intermediate blocks to top block with stop-grad)
    

-   This is an inexpensive coordination mechanism.
    

3.  Add **Goodness 3 (continuation contrast)** once token-level learning works
    

-   This is the most direct lever for “global characteristics.”
    

Then, if you still want wake/sleep separation:

-   Try it _after_ interleaved training is stable, and start with short sleep bursts (not thousands of steps) to avoid optimizer brittleness.
    

* * *

## Two design choices you should decide early (they matter)

1.  **Do you tie embeddings across layers?**
    If all layers share the same EEE, then every block is trying to shape the same token geometry. This can help alignment, but it can also create coupling. A compromise:
    

-   tie EEE globally, but only allow some layers to update it (e.g., top layer + maybe one auxiliary).
    

2.  **Depth scheduling (avoid “early block does everything”)**
    To keep deep blocks relevant:
    

-   smaller margin / weaker objective early, stronger later
    
-   or restrict early blocks to coarse targets (e.g., token clusters) and refine later
    

* * *

If you specify:

-   model size (e.g., nanoGPT 6-layer / d=384)
    
-   vocab type (char vs BPE)
    
-   and whether you want strict wake/sleep or interleaved training
    

I can translate the above into a concrete training loop with exact tensors, where to detach, candidate set construction, and a default set of hyperparameters (K, margins/temperatures, sync weights) that are usually stable.
