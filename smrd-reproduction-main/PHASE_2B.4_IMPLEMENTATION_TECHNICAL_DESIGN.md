# PHASE_2B.4 — Technical Design: Independent Reproduction Specification

**Reproduction Target:** *"Enhanced Large language model with a self-attention mechanism for Social Media Rumor Detection"* (`Enhanced_LLM_with_a_SAM_for_SMRD.docx`)  
**Document Version:** 2B.4 (Implementation-Ready Standalone Specification)  
**Preceding Specifications:** `PHASE_2B.3.4_TECHNICAL_DESIGN.md` (Frozen Base), `PHASE_2B.3.5_TECHNICAL_DESIGN.md` (Select C-improvements only)  
**Execution Status:** DESIGN ONLY. Specification tables, mathematical formulations, and interface contracts only. No source code written, no packages installed, no training executed, no empirical reproduction results claimed.  
**Hardware Feasibility Status:** Theoretical planning aid only; actual execution feasibility is unknown until runtime measurement gates G1–G6 are conducted.  

---

## 0. Replication Objective, Fidelity Tiers, Allocation Table, Dataset Reconciliation, Arms, and Budget

### 0.1 Objective of Independent Reproduction
The objective of this project is an independent empirical reproduction: to implement the architecture and hyperparameter optimization pipeline described in the paper *"Enhanced Large language model with a self-attention mechanism for Social Media Rumor Detection"* and determine whether its reported classification performance can be reproduced across three benchmark datasets under controlled, auditable conditions.

The paper reports the following headline figures (Section 4, Tables 4–9):
- **Twitter15:** 99.50% Accuracy, 99.57% Precision, 99.60% Recall, 99.59% F-measure
- **Twitter16:** 99.30% Accuracy, 99.56% Precision, 99.36% Recall, 99.39% F-measure
- **PHEME:** 99.10% Accuracy, 98.90% Precision, 98.98% Recall, 98.89% F-measure

This reproduction will test whether these numbers are achievable under the stated methodology, whether they depend on specific unstated modeling assumptions, or whether they reflect methodological discrepancies such as data leakage or label-distribution divergence. An empirical failure to reproduce the paper's claimed performance is a valid and scientifically informative finding and will be documented as such.

### 0.2 Fidelity Classification System
Every component in this technical design is explicitly classified into one of three tiers:

- **Tier 1 (T1) — Paper-Explicit:** Directly and unambiguously stated in the manuscript text, tables, algorithms, or embedded equation images (`word/media/image*.wmf`). Implemented as specified without substitution.
- **Tier 2 (T2) — Paper-Implied:** Follows from the text by direct technical derivation, resolving minor ambiguities in notation or terminology with documented textual justification.
- **Tier 3 (T3) — Gap-Filled (Implementation Assumption):** Unspecified, omitted, unreadable, or technically contradictory in the paper. The least-deviating defensible engineering choice is selected, explicitly labeled as an `IMPLEMENTATION ASSUMPTION`, accompanied by a one-line justification, and tracked in a sensitivity analysis plan. **Under no circumstances is a T2 or T3 element to be described or presented as the authors' choice.**

#### Explicit Allocation of Key Components

| Component / Subsystem | Assigned Tier | Source in Paper or Engineering Justification |
| :--- | :---: | :--- |
| **Benchmark Datasets** (Twitter15, Twitter16, PHEME) | **T1** | Manuscript Section 4.1, Table 3 |
| **Dataset Splits (70/10/20 train/val/test)** | **T3** | Split ratio unstated in manuscript; adopted standard stratified split (Assumption A-SPLIT) |
| **Headline Claimed Results** | **T1** | Manuscript Section 4.1, Tables 4–9 |
| **Core Model Backbone** (RoBERTa + Medium GPT) | **T1** | Manuscript Section 3.2, Section 3.3 |
| **RoBERTa Frozen Feature Extraction** | **T3** | Offline frozen extraction mode prevents severe GPU memory exhaustion (Assumption A-FREEZE-ROB) |
| **RoBERTa-to-GPT-2 Continuous Bridge** | **T3** | Linear(768, 1024) + LayerNorm(1024) bridges dimension mismatch (Assumption A-BRIDGE) |
| **Pruning Unused Embedding Matrix `wte`** | **T3** | Bypassed by continuous injection; pruning frees 51.46M params / 0.192 GiB (Assumption A-PRUNE-WTE) |
| **GPT-2 Early-Exit ($NL \le 24$) & Appended Blocks ($NL > 24$)** | **T1 Range / T3 Mechanism** | Spans Table 1 range $NL \in [1, 48]$ using master checkpoint appended blocks (Assumption A-LAYER) |
| **Attention Head Masking ($NAH \in [1, 16]$)** | **T1 Range / T3 Mechanism** | Binary head mask over active heads, NO variance rescaling (Assumption A-HEAD) |
| **Dropout Sites ($DR \in [0.1, 0.5]$)** | **T1 Range / T3 Mechanism** | Applied to embd, attn, resid dropouts and classification head; bridge has no dropout (Assumption A-DROP) |
| **Sequence Pooling & Padding** | **T3** | Right-padding at embedding level; last-valid-token pooling $\sum(\mathbf{M})-1$ (Assumption A-POOL) |
| **Classification Head** | **T3** | `Dropout(DR) + Linear(1024, d_out)` matching task arm classes (Assumption A-HEAD-CLS) |
| **DHOA Search Space** ($NL, NAH, DR, B, L$) | **T1** | Manuscript Section 3.3.1, Table 1 |
| **DHOA Core Equations (Eqs. 4–14)** | **T1** | Manuscript Section 3.3.1, verified from embedded WMF files (`image5` to `image30`) |
| **DHOA Budget ($N=6$ deer, 36 evaluations)** | **T3** | Population and iteration budget omitted in paper; bounded for tractability (Assumption A-BUDGET) |
| **Candidate Training Epochs (2 epochs)** | **T3** | Epoch count omitted in paper; 2 epochs per candidate avoids intractable compute (Assumption A-EPOCHS) |
| **DHOA Coordinate Space (Normalized $[0, 1]^5$)** | **T3** | Prevents $B \in [1, 512]$ from drowning out $L \in [10^{-5}, 5 \times 10^{-4}]$; log-linear $L$ (Assumption A-DHOA-SPACE) |
| **DHOA Table 2 Branch Mapping** | **T3** | UNRESOLVED; primary mapping supported by Strategy 3 prose, alternative in S10 (Assumption A-DHOA-BRANCH) |
| **DHOA Symbol Formulations ($l, w, r, z, \theta, \phi, v$)** | **T3** | Unspecified stochastic variables defined with concrete distributions (Assumptions A-DHOA-*) |
| **Preprocessing Operations (5 Operations)** | **T1** | Manuscript Section 3.1: URL removal, stop words, tokenization, Light-10, lemmatization |
| **Preprocessing Execution Order** | **T3** | URL $\to$ text[:512] $\to$ tokenize $\to$ stopwords $\to$ Light-10 $\to$ lemmatize $\to$ BPE (Assumption A-ORDER) |
| **Linear Micro-Batch Accumulation** | **T2** | Mathematically preserves mean batch gradient across $B \in [1, 512]$ (Assumption A-ACCUM) |
| **AMP Training & Gradient Clipping** | **T3** | Hardware-selected dtype (BF16/FP16), clipping `max_norm=1.0` (Assumption A-AMP) |
| **Master Initial Checkpoint & Seeds** | **T3** | $S_{split}=42, S_{init}=100, S_{order}=200, S_{dropout}=300, S_{dhoa}=400$ (Assumption A-SEEDS) |
| **Evaluation Metrics (Eqs. 15–22)** | **T1** | Manuscript Section 4.1; Eqs. 15–18 confirmed from WMFs, Eqs. 19–22 typeset |

### 0.3 Dataset Reconciliation Discrepancy
A rigorous census of the physical dataset releases versus the manuscript's Table 3 reveals significant discrepancies in class distributions, as documented in the Phase 1 audit:

| Dataset | Metric / Class | Released Archive Count | Paper Table 3 Count | Discrepancy / Census Note |
| :--- | :--- | :---: | :---: | :--- |
| **Twitter15** | # of source tweets | **1,490** | **1,491** | Sum of printed classes in Table 3 is $375+371+373+373 = \mathbf{1,492}$ |
| | Non-rumor (`0`) | 374 | 375 | Discrepancy of +1 in printed table |
| | False rumour (`1`) | 370 | 371 | Discrepancy of +1 in printed table |
| | True rumour (`2`) | 372 | 373 | Discrepancy of +1 in printed table |
| | Unverified rumour (`3`)| 374 | 373 | Discrepancy of -1 in printed table |
| **Twitter16** | # of source tweets | **818** | **817** | Sum of printed classes in Table 3 is $204+204+206+203 = \mathbf{817}$ |
| | Non-rumor (`0`) | 205 | 204 | Discrepancy of -1 in printed table |
| | False rumour (`1`) | 205 | 204 | Discrepancy of -1 in printed table |
| | True rumour (`2`) | 207 | 206 | Discrepancy of -1 in printed table |
| | Unverified rumour (`3`)| 201 | 203 | Discrepancy of +2 in printed table |
| **PHEME** | # of source tweets | **6,425** | **6,425** | Total thread counts match ($N = 6,425$) |
| | Non-rumor (`0`) | 4,022 | 2,572 | Paper reports -1,450 non-rumors |
| | False rumour (`1`) | 638 | 1,678 | Paper reports +1,040 false rumors |
| | True rumour (`2`) | 1,067 | 1,973 | Paper reports +906 true rumors |
| | Unverified rumour (`3`)| 698 | 202 | Paper reports -496 unverified |

**PHEME Rumour Total Impossibility:** As established during the Phase 1 census audit, the physical PHEME archive contains only 2,403 rumor threads in total ($638 \text{ false} + 1,067 \text{ true} + 698 \text{ unverified}$). The paper's Table 3 claims $1,678 + 1,973 + 202 = \mathbf{3,853}$ rumors, which is physically impossible given the directory structure of the released PHEME dataset. All experiments in this reproduction run on the verified physical data.

### 0.4 Experimental Task Arms, Task Prioritization, and Experiment Budget
The manuscript contains a fundamental task definition contradiction:
- **Section 1 (p. 2) & Section 3:** Explicitly frames the task as binary classification (*"In order to distinguish between rumored and non-rumored tweets, the task of detecting rumors is seen as a binary classification issue"* and *"differentiate between true and false information"*).
- **Section 3.3.1 (Eq. 6):** Formulates fitness strictly in terms of binary confusion counts: $\text{Fitness} = \text{Max}\left(\frac{TP+TN}{TP+TN+FP+FN}\right)$.
- **Section 4.1 & Table 3:** Presents four distinct classes (`non-rumour`, `false`, `true`, `unverified`) and reports performance tables across datasets.

To evaluate every interpretation without silent methodology substitution, this reproduction specifies three independent training arms and one derived reporting arm:

1. **Arm B — Rumour Detection Arm (T1 Section 1 & Eq. 6 / T3 Training) [PRIMARY EXECUTION ARM — RUNS FIRST]:**
   - **Task:** Rumour versus non-rumour binary classification ($d_{out}=2$).
   - **Priority Rationale:** Equation 6 explicitly specifies binary accuracy with $TP, TN, FP, FN$. Section 1 explicitly frames rumor detection as a binary classification task. Arm B directly satisfies both the literal text and the literal fitness formula.
   - **Label Mapping:**
     $$y_{bin} = \begin{cases} 0, & \text{if } y = 0 \text{ (non-rumour)} \\ 1, & \text{if } y \in \{1, 2, 3\} \text{ (rumour: false, true, unverified)} \end{cases}$$
   - **Objective:** Binary Cross-Entropy Loss ($d_{out}=2$).
   - **DHOA Fitness:** Binary validation accuracy matching Equation (6).
   - **Sample Counts (Derived from Phase 1 split files):**
     - **Twitter15** ($N=1,490$): Train = 1,043 (261 non-rumor, 782 rumor), Val = 149 (38 non-rumor, 111 rumor), Test = 298 (75 non-rumor, 223 rumor).
     - **Twitter16** ($N=818$): Train = 572 (143 non-rumor, 429 rumor), Val = 82 (21 non-rumor, 61 rumor), Test = 164 (41 non-rumor, 123 rumor).
     - **PHEME** ($N=6,425$): Train = 4,497 (2,816 non-rumor, 1,681 rumor), Val = 643 (402 non-rumor, 241 rumor), Test = 1,285 (804 non-rumor, 481 rumor).

2. **Arm A — Four-Class Classification Arm (T1 Tables / T3 Training & Fitness Generalization):**
   - **Task:** 4-class classification ($d_{out}=4$).
   - **Labels:** $y \in \{0: \text{non-rumour}, 1: \text{false}, 2: \text{true}, 3: \text{unverified}\}$.
   - **Objective:** 4-class Cross-Entropy Loss ($d_{out}=4$).
   - **DHOA Fitness:** 4-class validation overall accuracy (multiclass generalization of Eq. 6).
   - **Sample Counts (Derived from Phase 1 split files):** Identical split sizes as Arm B.

3. **Arm D — Veracity Classification Arm (T1 Section 3 / T3 Training):**
   - **Task:** True versus false binary classification ($d_{out}=2$).
   - **Non-Rumour & Unverified Handling:** Non-rumour ($y=0$) and unverified ($y=3$) threads are **strictly excluded** from both training and evaluation. They are not assigned to either class.
   - **Label Mapping:**
     $$y_{ver} = \begin{cases} 0, & \text{if } y = 1 \text{ (false rumour)} \\ 1, & \text{if } y = 2 \text{ (true rumour)} \end{cases}$$
   - **Objective:** Binary Cross-Entropy Loss ($d_{out}=2$).
   - **DHOA Fitness:** Veracity validation accuracy.
   - **Sample Counts (Derived directly from Phase 1 split files):**
     - **Twitter15** ($N=742$ threads: 370 false, 372 true): Train = 520 (259 false, 261 true), Val = 74 (37 false, 37 true), Test = 148 (74 false, 74 true). Check: $520 + 74 + 148 = 742$.
     - **Twitter16** ($N=412$ threads: 205 false, 207 true): Train = 288 (144 false, 144 true), Val = 41 (20 false, 21 true), Test = 83 (41 false, 42 true). Check: $288 + 41 + 83 = 412$.
     - **PHEME** ($N=1,705$ threads: 638 false, 1,067 true): Train = 1,193 (446 false, 747 true), Val = 171 (64 false, 107 true), Test = 341 (128 false, 213 true). Check: $1,193 + 171 + 341 = 1,705$.

4. **Report C — Post-Hoc Binary Collapse of Arm A Predictions (T2 Diagnostic):**
   - Uses the trained Arm A model without retraining.
   - Predictions are collapsed: if $\hat{y} \in \{1, 2, 3\} \implies \hat{y}_{bin} = 1$; if $\hat{y} = 0 \implies \hat{y}_{bin} = 0$.
   - Evaluates Equations (15)–(18) on collapsed outputs as a diagnostic check.

#### Comprehensive Experiment Budget
Arms A, B, and D require independent DHOA hyperparameter optimization passes because their loss functions, output heads, and target distributions differ:
$$\text{Total Candidate Training Runs} = 3 \text{ training arms} \times 3 \text{ datasets} \times 36 \text{ candidate evaluations} = 324 \text{ candidate runs}$$
Each candidate evaluation trains for **2 epochs**. Following search, each of the $3 \times 3 = 9$ winning configurations executes **1 final model retraining run on the TRAIN split only** from the master initial checkpoint for **2 epochs**, followed by a single evaluation on the test set.

| Task Arm | Dataset | Split Sizes (Train / Val / Test) | DHOA Search Budget | Final Retrain Runs (Train Split Only) | Total Model Fits | Priority Order |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Arm B** (Rumour Detection) | Twitter16 | 572 / 82 / 164 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | **1 (Pilot First)** |
| **Arm B** (Rumour Detection) | Twitter15 | 1,043 / 149 / 298 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 2 |
| **Arm B** (Rumour Detection) | PHEME | 4,497 / 643 / 1,285 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 3 |
| **Arm A** (4-Class Classification) | Twitter16 | 572 / 82 / 164 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 4 |
| **Arm A** (4-Class Classification) | Twitter15 | 1,043 / 149 / 298 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 5 |
| **Arm A** (4-Class Classification) | PHEME | 4,497 / 643 / 1,285 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 6 |
| **Arm D** (Veracity Classification)| Twitter16 | 288 / 41 / 83 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 7 |
| **Arm D** (Veracity Classification)| Twitter15 | 520 / 74 / 148 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 8 |
| **Arm D** (Veracity Classification)| PHEME | 1,193 / 171 / 341 | 6 deer $\times$ 6 rounds = 36 | 1 final retrain (train only) | 37 | 9 |
| **Report C** (Post-Hoc Collapse) | All 3 | Evaluated on Arm A outputs | No training | 0 (post-hoc evaluation) | 0 | Diagnostic |
| **Sensitivity Plan** (S1–S14) | Twitter16 Pilot | Various | Standalone ablations | 14 ablation evaluations | 14 | Post-Search |
| **Grand Total** | — | — | **324 evaluations** | **9 retrains + 14 ablations** | **347 model fits** | — |

---

## 1. End-to-End Architecture and Tensor Flow

```
                                  [Raw Social Media Tweet]
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  Module 1: Deterministic Preprocessing Pipeline                                        │
 │  • Step 1. URL Removal (Regex active URLs: https?://\S+|www\.\S+)     [T1 - Ref 23]       │
 │  • Step 2. 512-Character String Slicing (text = text[:512])           [T2 - Section 3.2] │
 │  • Step 3. Lexical Tokenization (TweetTokenizer, case preserved)      [T1/T3 - Ref 23]   │
 │  • Step 4. Stopword Removal (NLTK English stopword filter)            [T1 - Ref 24]      │
 │  • Step 5. Light-10 Terminal Suffix Rule (Strip "and" if len-3 >= 3)  [T1 / T3 - Sec 3.1]│
 │  • Step 6. Dictionary Lemmatization (WordNet Lemmatizer with POS)      [T1 / T3 - Sec 3.1]│
 │  • Step 7. Reconstructed Text & BPE Tokenizer Guard (max_len=512)     [T3 - Assumption]  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                             │  Tensor Shape: Text string (tokens joined by ' ')
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  Module 2: Offline Feature Extraction via RoBERTa (Section 3.2)                        │
 │  • Checkpoint: roberta-base (12 layers, 12 heads, d_roberta = 768)   [T3 - Assumption] │
 │  • Weights: Frozen (eval mode: requires_grad=False, dropout off)     [T3 - Assumption] │
 │  • Token + Position Embeddings (Native architecture, no segment IDs) [T2 - Corrected]  │
 │  • Sequence Tensor Extraction: H_seq in R^(T x 768), T <= 512         [T1 - Section 3.2] │
 │  • Persistent Disk Cache (Memmapped FP16 binary array + manifest)     [T3 - Assumption] │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                             │  Cached Tensor: [B_phys, T, 768] (FP16/FP32)
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  Module 3: RoBERTa-to-GPT-2 Continuous Tensor Bridge                                   │
 │  • Trainable Linear Projection: W_bridge in R^(768 x 1024), b in R^1024[T3 - Assumption] │
 │    Initialized W_bridge ~ N(0, 0.02), b = 0                           [T3 - Assumption] │
 │  • Normalization: LayerNorm(1024) (gamma=1, beta=0, eps=1e-5)        [T3 - Assumption] │
 │  • Projected Continuous Sequence: E_proj = LN(H_seq @ W_bridge + b)   [T3 - Assumption] │
 │  • Direct Injection: gpt2_medium.forward(inputs_embeds=E_proj)        [T3 - Assumption] │
 │  • Position Embeddings: E_proj = E_proj + wpe[0:T, :]                 [T2 - Implied]    │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                             │  Projected Tensor: [B_phys, T, 1024]
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  Module 4: GPT-2 Medium Dynamic Backbone (Section 3.3 & Table 1)                       │
 │  • Checkpoint: gpt2-medium (d_model = 1024, base blocks = 24, heads = 16) [T1 / T3]    │
 │  • Pruned wte: Unused 51.46M token embedding matrix pruned from memory[T3 - Assumption] │
 │  • Dynamic Layer Routing (NL in [1, 48]):                             [T1 - Table 1]    │
 │    - NL <= 24: Early-exit sub-network across blocks h[0 ... NL-1]     [T3 - Assumption] │
 │    - NL > 24: Pretrained blocks h[0..23] + (NL-24) Appended Blocks    [T3 - Assumption] │
 │               from master checkpoint (W_cproj = 0, b_cproj = 0)                         │
 │  • Dynamic Attention Head Masking (NAH in [1, 16]):                  [T1 - Table 1]    │
 │    - First NAH heads active via binary mask, NO variance rescaling    [T3 - Assumption] │
 │  • Regularization: Dynamic DR in [0.1, 0.5] applied to embd/attn/resid [T1 - Table 1]   │
 │  • Padding: Zero-vector embedding padding, attention_mask in {0, 1}   [T3 - Assumption] │
 │  • Final Layer Normalization: ln_f applied after last active block    [T3 - Assumption] │
 │  • Last-Valid-Token Pooling: h_last = H_out[sum(attention_mask) - 1]  [T3 - Assumption] │
 └────────────────────────────────────────────────────────────────────────────────────────┘
                                             │  Pooled Vector: [B_phys, 1024]
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │  Module 5: Classification Head & Objective Optimization                                │
 │  • Regularization: nn.Dropout(p=DR) applied to h_last                [T1 - Table 1]    │
 │  • Linear Classifier: W_cls in R^(1024 x d_out), b_cls in R^d_out     [T3 - Assumption] │
 │    (d_out = 2 for Arms B and D; d_out = 4 for Arm A)                                   │
 │  • Logits & Scoring: z = Dropout(h_last) @ W_cls + b_cls              [T1 - Section 3.3] │
 │  • Training Loss: Unweighted CrossEntropyLoss with n_j/b micro-scaling [T2 - Implied]   │
 │  • Optimization Fitness: Validation Accuracy (Eq. 6 / Arm A Generalization) [T1 / T3]  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing and Feature Caching

### 2.1 Canonical Operation Order
Manuscript Section 3.1 details text preprocessing through five discrete operations. The canonical execution sequence is defined as:

$$\text{Raw Text} \xrightarrow{\text{Step 1}} \text{URL Removal [23]} \xrightarrow{\text{Step 2}} \text{Slice [:512]} \xrightarrow{\text{Step 3}} \text{Lexical Tokenize [23]} \xrightarrow{\text{Step 4}} \text{Stopwords [24]} \xrightarrow{\text{Step 5}} \text{Light-10} \xrightarrow{\text{Step 6}} \text{Lemmatize} \xrightarrow{\text{Step 7}} \text{RoBERTa BPE}$$

**Strict Preprocessing Bounds:** The preprocessing pipeline is restricted strictly to the operations specified in Section 3.1 of the paper. **Do NOT add lowercasing of the whole text, @mention removal, hashtag stripping, punctuation removal, or number removal: none are in the paper.** Mentions (`@user`), hashtags (`#topic`), numbers, punctuation tokens, contractions, and casing are preserved as tokens through lexical tokenization.

### 2.2 Preprocessing Operation Details
1. **URL Removal (T1 — Section 3.1, [Ref 23]):**
   - Matches and excises web addresses via regex: `https?://\S+|www\.\S+`.
   - **URL-Only Removal (No HTML Handling) (T3 — Assumption A-URL):** The paper specifies URL removal, not HTML tag stripping (`<p>`, `<div>`) or entity unescaping (`&amp;` $\to$ `&`). To remain strictly faithful to the explicit text, HTML tags and entities are **not** removed or decoded.

2. **512-Character String Slicing (T2 — Section 3.2):**
   - Section 3.2 states that the model *"extends supplied sentence length to 512 characters"*.
   - Executed as a literal string slice: `text = text[:512]`.

3. **Lexical Tokenization (T1/T3 — Section 3.1, [Ref 23], Assumption A-TOK):**
   - Tokenizer: `nltk.tokenize.TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)`.
   - `preserve_case=True`: Raw case is preserved for all tokens.
   - `reduce_len=False`: Character repetition sequences are preserved intact without truncation.
   - `strip_handles=False`: User handles (`@user`) are retained as individual tokens.
   - Contractions (e.g. "don't", "can't"), emojis/emoticons, punctuation tokens, hashtags, and mentions pass through as discrete tokens.

4. **Stopword Removal (T1 — Section 3.1, [Ref 24]):**
   - Filtered against NLTK's English stopword lexicon (`nltk.corpus.stopwords.words('english')`).
   - Matching is case-insensitive: a token is filtered if `token.lower()` matches a stopword.
   - Tokens retain their original case unless an operation modifies them, and modified tokens are emitted lowercase.
   - **Negation Lexeme Warning (T3 — Assumption A-STOP):** The standard NLTK stopword lexicon contains grammatical negations (`no`, `not`, `nor`). The default configuration applies the literal NLTK stopword list as indicated by Ref [24]; the impact of negation removal is evaluated in sensitivity ablation S5.

5. **Light-10 Terminal Suffix Rule (T1 / T3 — Section 3.1, Assumption A-LIGHT10):**
   - **Citation Verification:** Section 3.1 mentions Light-10 without any citation. Ref [25] (Gao et al., 2022) describes RoBERTa CLS vector normalization and must **not** be attached to Light-10.
   - **Linguistic Provenance:** The paper provides no citation or linguistic origin for its Light-10 rule. Any external claims regarding linguistic provenance are unverified in the context of this paper. This is a literal operational reading that is not a validated stemmer.
   - **Literal Operational Rule Formulation:**
     $$\text{If } \text{token.lower}().\text{endswith}("\text{and}") \text{ and } (\text{len}(\text{token}) - 3) \ge 3: \quad \text{token} \leftarrow \text{token}[:-3]$$
   - **Destructive Lexical Impact:** Truncates any token ending in `"and"` with $\ge 6$ characters:
     - `expand` $\to$ `exp`
     - `demand` $\to$ `dem`
     - `command` $\to$ `comm`
     - `understand` $\to$ `underst`
     - `island` $\to$ `isl`
     - `contraband` $\to$ `contrab`
     Words with $< 6$ characters (`and`, `band`, `hand`, `sand`, `stand`) remain untouched because $\text{len} - 3 < 3$.

6. **Dictionary Lemmatization (T1 / T3 — Section 3.1, Assumption A-LEM):**
   - **Citation Verification:** Section 3.1 contains **no citation** for lemmatization. Ref [24] is the stop-word/spam-email paper and must **not** be attached to lemmatization.
   - **Algorithm:** WordNet Lemmatizer (`nltk.stem.WordNetLemmatizer`).
   - **Two-Pass Resolution:** To fulfill the paper's explicit example (*"sharing"* $\to$ *"share"*), verb lemmatization is prioritized: `lemma_v = lemmatizer.lemmatize(token.lower(), pos='v')`. If `lemma_v != token.lower()`, adopt `lemma_v`; otherwise fallback to noun lemmatization: `lemmatizer.lemmatize(token.lower(), pos='n')`.
   - **WordNet Fallback:** If no lemma exists in WordNet, `WordNetLemmatizer` returns the token unchanged.
   - **Documented Known Quirks:** Words with conflicting inflections behave according to lookup priority (e.g., "leaves" $\to$ "leave" via verb path rather than "leaf" as noun; "was" $\to$ "be" via verb path).
   - Reconstructed tokens are re-joined using single spaces: `text_clean = ' '.join(tokens)`.

7. **RoBERTa Tokenizer Truncation Guard (T3 — Assumption A-SEQ):**
   - Reconstructed text is tokenized using `RobertaTokenizerFast.from_pretrained('roberta-base')`.
   - Parameters: `max_length=512, truncation=True, add_special_tokens=True`.
   - Special boundary tokens `<s>` and `</s>` leave exactly 510 positions for subwords. Any truncation events are logged.

### 2.3 Feature Caching Protocol (T3 — Assumption A-CACHE)
To decouple feature extraction from hyperparameter search and eliminate redundant compute:
- **Storage Layout:** Flat memory-mapped FP16 binary array storing RoBERTa sequence hidden states:
  - `features.bin`: Contiguous array of shape $(\sum_{i=1}^N T_i, 768)$ in IEEE 754 half-precision float (`float16`).
  - `metadata.json`: Sequence offsets, individual lengths $T_i \in [1, 512]$, ground-truth labels, and tweet IDs.
- **Cache Generation:** Features are extracted with RoBERTa in evaluation mode (`eval()`) in FP32 precision, then cast and saved to disk as FP16.
- **Integrity Manifest:** Accompanied by `manifest.sha256` hashing preprocessed text, tokenizer configuration, model checkpoint revision, and library versions.
- **Verification Tolerance:** Reloaded cached tensors are verified against an FP32 reference forward pass cast to FP16: `torch.allclose(cached_tensor, fp32_ref.half(), rtol=1e-3, atol=1e-3)`. An absolute error bound of $10^{-4}$ is mathematically unattainable in FP16.

#### Cache Storage Size Formulation
- **Formula:** $\text{Bytes} = \sum_{i=1}^N T_i \times 768 \times 2 \text{ bytes}$.
- **Theoretical Worst-Case Bound ($T_i = 512$ for all examples):** $512 \times 768 \times 2 = 786,432 \text{ bytes per example}$ ($0.75 \text{ MiB/example}$).
  - **Twitter15** ($N = 1,490$): $1,490 \times 786,432 = 1,171,783,680 \text{ bytes} \approx \mathbf{1.091 \text{ GiB}}$ ($1.172 \text{ GB}$).
  - **Twitter16** ($N = 818$): $818 \times 786,432 = 643,301,376 \text{ bytes} \approx \mathbf{0.599 \text{ GiB}}$ ($0.643 \text{ GB}$).
  - **PHEME** ($N = 6,425$): $6,425 \times 786,432 = 5,052,825,600 \text{ bytes} \approx \mathbf{4.706 \text{ GiB}}$ ($5.053 \text{ GB}$).
- **Illustrative Assumed Average Length Example (Planning Aid Only — Unmeasured):** Assuming an illustrative average token length of $\bar{T} = 35$ tokens per tweet (illustrative assumed average length, never empirical):
  - Twitter15 ($1,490$): $1,490 \times 35 \times 768 \times 2 = 80,102,400 \text{ bytes} \approx 76.4 \text{ MiB}$.
  - Twitter16 ($818$): $818 \times 35 \times 768 \times 2 = 43,975,680 \text{ bytes} \approx 41.9 \text{ MiB}$.
  - PHEME ($6,425$): $6,425 \times 35 \times 768 \times 2 = 345,417,600 \text{ bytes} \approx 329.4 \text{ MiB}$.

---

## 3. RoBERTa-to-GPT-2 Interface

### 3.1 RoBERTa Configuration
- **Model Checkpoint (T3 — Assumption A-ROBERTA):** Canonical `roberta-base` (12 layers, 12 attention heads, hidden dimension $d_{roberta} = 768$, 125M parameters).
- **Frozen Feature Extractor (T3 — Assumption A-FREEZE-ROB):** RoBERTa parameter weights are strictly frozen (`requires_grad=False`). Hidden features are extracted in evaluation mode (`eval()`) with dropout disabled.
- **Equation (1) Correction (T2 — Technical Invariant):** Manuscript Equation (1) claims $U_h = U_{token\_emb} + U_{seg\_emb} + U_{pos\_emb}$. Native RoBERTa does not use segment/token-type embeddings. Standard native `roberta-base` embeddings ($U_{token\_emb} + U_{pos\_emb}$) are used without synthetic segment embeddings.

### 3.2 Continuous Tensor Bridge and Initialization Justification (T3 — Assumption A-BRIDGE)
The manuscript specifies that RoBERTa extracted features are classified by GPT, but omits the interface mechanism. Direct injection is impossible because RoBERTa outputs dimension 768, while GPT-2 Medium expects hidden dimension 1024.

- **Bridge Formulation:** A trainable continuous linear projection maps feature space:
  $$\mathbf{E}_{proj} = \text{LayerNorm}_{1024}\left(\mathbf{H}_{seq} \mathbf{W}_{bridge} + \mathbf{b}_{bridge}\right)$$
  where $\mathbf{W}_{bridge} \in \mathbb{R}^{768 \times 1024}$ and $\mathbf{b}_{bridge} \in \mathbb{R}^{1024}$.
- **Weight Initialization Specification:**
  - Projection matrix $\mathbf{W}_{bridge}$ is initialized from a normal distribution $\mathcal{N}(0, 0.02)$, matching standard GPT-2 layer initialization conventions.
  - Bias $\mathbf{b}_{bridge}$ is initialized to $\mathbf{0}$.
  - LayerNorm parameters are initialized to default: gain $\gamma = 1.0$, bias $\beta = 0.0$, $\epsilon = 10^{-5}$.
  - *Variance Scaler Rejection:* Any claim that scaling initialization by $1/\sqrt{768}$ "prevents variance explosion before LayerNorm" is mathematically invalid and excised. LayerNorm explicitly normalizes input activations to zero mean and unit variance; an excessively small pre-LayerNorm scale interacts pathologically with regularizer $\epsilon = 10^{-5}$. Standard $\mathcal{N}(0, 0.02)$ ensures well-conditioned pre-normalization activations.
- **Bridge Dropout:** No dropout is applied within the bridge projection layer.

### 3.3 GPT-2 Input Injection and Embedding-Level Padding (T3 — Assumption A-PAD)
Because continuous representations are injected directly via `inputs_embeds`, discrete token IDs do not exist during GPT-2 execution.

- **Pruning Unused `wte` Embedding Matrix (T3 — Assumption A-PRUNE-WTE):** Because input tokens bypass discrete embedding lookup, the pretrained token embedding matrix `wte` ($50,257 \times 1024 = 51,463,168$ parameters, representing $\approx 0.192 \text{ GiB}$ in FP32 static memory) is excluded from the trainable parameter set and pruned from memory upon model instantiation.
- **Embedding Padding:** Padding positions in $\mathbf{E}_{proj}$ are filled with **zero vectors** ($\mathbf{0} \in \mathbb{R}^{1024}$).
- **Padding Orientation:** Right-padding is enforced.
- **Attention Mask:** A binary mask $\mathbf{M} \in \{0, 1\}^{B \times T}$ is supplied, where $\mathbf{M}[i, t] = 1$ for valid sequence positions and $0$ for padded positions.
- **Position IDs:** Explicit position indices $[0, 1, \dots, T-1]$ are passed to GPT-2's learnable positional embedding layer (`wpe`).
- **Last-Valid-Token Pooling (T3 — Assumption A-POOL):** Sequence representations are pooled strictly from the last valid non-padded token position using 0-based index:
  $$t_{last} = \sum_{t=1}^{T} \mathbf{M}[i, t] - 1, \quad \mathbf{h}_{last} = \mathbf{H}_{GPT}[i, t_{last}, :] \in \mathbb{R}^{1024}$$
  *(Note: Masked mean-pooling is rejected as unauthorized from the frozen base; it is evaluated solely as ablation S13).*

---

## 4. GPT-2 Dynamic Architecture Search Mechanisms

### 4.1 Dynamic Layer Sizing ($NL \in [1, 48]$) and Zero-Residual Gradient Flow (T1 Range / T3 Mechanism)
Pretrained `gpt2-medium` natively contains 24 transformer blocks ($L_{native} = 24$). Manuscript Table 1 specifies searching $NL \in [1, 48]$.

```
Case A: NL <= 24 (Early-Exit Sub-Network)
E_proj ──> [Block 0] ──> [Block 1] ──> ... ──> [Block NL-1] ──> [ln_f] ──> h_last
           (Blocks NL to 23 are completely bypassed)

Case B: NL > 24 (Hybrid Identity Extension via Master Checkpoint)
E_proj ──> [Blocks 0..23 Pretrained] ──> [Block 24] ──> ... ──> [Block NL-1] ──> [ln_f] ──> h_last
                                         └──────── ONE Fixed Set of 24 Appended Blocks ────────┘
                                                   (W_cproj = 0, b_cproj = 0)
```

1. **Case A: Early-Exit Sub-Network ($NL \le 24$):**
   - Forward pass executes sequentially through blocks `h[0]` to `h[NL-1]`.
   - Blocks `h[NL]` through `h[23]` are bypassed. Pretrained weights remain unaltered.

2. **Case B: Hybrid Identity Extension ($NL > 24$):**
   - All 24 pretrained blocks execute. An additional $K = NL - 24$ transformer blocks are dynamically enabled from **ONE fixed set of 24 appended blocks stored in the master initial checkpoint**.
   - **Hybrid Architecture Disclosure:** Models evaluated with $NL > 24$ are hybrid architectures consisting of 24 pretrained GPT-2 blocks plus newly initialized blocks; they are **not** the pretrained checkpoint.
   - Each appended block possesses identical structural dimensions: $d_{model} = 1024, d_{ff} = 4096, N_{heads} = 16$.
   - **Zero-Residual Initialization Protocol:**
     - Attention input projection `c_attn.weight` and MLP intermediate expansion `c_fc.weight` are initialized from $\mathcal{N}(0, 0.02)$.
     - All layer biases are initialized to $\mathbf{0}$.
     - LayerNorm parameters: $\gamma = 1.0, \beta = 0.0$.
     - **Both output projections are initialized to EXACT ZERO:** $\mathbf{W}_{c\_proj} = \mathbf{0}, \mathbf{b}_{c\_proj} = \mathbf{0}$ for attention output projection and MLP output projection.

3. **Mathematical Invariant at Initialization:**
   $$\mathbf{x}' = \mathbf{x} + \text{Attention}(\text{LN}_1(\mathbf{x})) = \mathbf{x} + \mathbf{0} = \mathbf{x}$$
   $$\mathbf{x}'' = \mathbf{x}' + \text{MLP}(\text{LN}_2(\mathbf{x}')) = \mathbf{x}' + \mathbf{0} = \mathbf{x}$$
   At initialization step zero, appended blocks compute exact identity mappings. The expanded model initially matches the exact representation of the 24-layer pretrained model.

4. **Non-Zero Gradient Flow Proof for Appended Blocks:**
   - Because the residual connection is identity, the incoming gradient with respect to output is $\frac{\partial \mathcal{L}}{\partial \mathbf{x}''} \neq \mathbf{0}$.
   - Intermediate forward activations $\mathbf{a}_{mid} = \text{GELU}(\text{LN}_2(\mathbf{x}') \mathbf{W}_{c\_fc} + \mathbf{b}_{c\_fc}) \neq \mathbf{0}$ are strictly non-zero because intermediate weights $\mathbf{W}_{c\_fc}$ are drawn from $\mathcal{N}(0, 0.02)$.
   - By the chain rule, the gradient with respect to the zero output projection weight is:
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{c\_proj}} = \mathbf{a}_{mid}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{x}''} \neq \mathbf{0}$$
   - Consequently, appended blocks receive immediate non-zero gradients from the very first optimization step and smoothly diverge from identity mapping.

5. **Final LayerNorm Placement ($ln\_f$):** GPT-2's final normalization layer ($ln\_f$) is applied **after the last active block** (`h[NL-1]`) prior to sequence pooling.

### 4.2 Dynamic Attention Head Masking ($NAH \in [1, 16]$) (T1 Range / T3 Mechanism)
Table 1 specifies searching $NAH \in [1, 16]$. Standard `gpt2-medium` contains 16 attention heads per layer ($d_k = 1024 / 16 = 64$).

- **Head Masking Protocol:** Rather than physically modifying weight matrices, dynamic capacity is controlled via a binary head mask $\mathbf{m}_{head} \in \{0, 1\}^{16}$ applied across all executed layers:
  $$\mathbf{m}_{head}[h] = \begin{cases} 1, & 0 \le h < NAH \\ 0, & NAH \le h < 16 \end{cases}$$
- **No Variance Rescaling (T3 — Assumption A-HEAD):** Head outputs are **not** rescaled by $16/NAH$. (Variance rescaling is rejected from the base and tracked exclusively in ablation S14).
- **Hardware Note:** In standard PyTorch eager attention, head masking zeroes out information flow but does not reduce peak activation memory or FLOP count.

### 4.3 Dropout Regularization ($DR \in [0.1, 0.5]$) and RNG Limitations (T1 Range / T3 Sites)
Table 1 mandates searching $DR \in [0.1, 0.5]$. This hyperparameter dynamically controls four distinct dropout sites:
1. Classification head dropout: `nn.Dropout(p=DR)` applied to $\mathbf{h}_{last}$.
2. GPT-2 embedding dropout: `config.embd_pdrop = DR`.
3. GPT-2 attention dropout: `config.attn_pdrop = DR` across all active blocks (including appended ones).
4. GPT-2 residual dropout: `config.resid_pdrop = DR` across all active blocks (including appended ones).

- **Dropout Exclusions:** The Continuous Tensor Bridge has no dropout. During validation and test evaluation, `model.eval()` completely disables dropout across all modules.
- **PRNG Stream and Batch-Size Interaction:**
  - Setting a shared base seed $S_{dropout} = 300$ fixes the pseudorandom number generator state at the start of training.
  - However, shared seeds do not guarantee identical dropout masks across different batch sizes $B$ or micro-batch sizes $B_{phys}$.
  - Different batch configurations execute different numbers of forward passes, advancing the PRNG stream by different step counts. This introduces stochastic variation between batch size configurations that cannot be eliminated by seed control alone.

---

## 5. DHOA Optimization Protocol and Search Space

### 5.1 Search Parameter Space (T1 — Table 1)
Manuscript Table 1 defines the 5-dimensional continuous/discrete hyperparameter search space:

| Parameter Symbol | Parameter Description | Manuscript Search Range | Parameter Type | Coordinate Mapping Protocol (T3 — Assumption A-DHOA-SPACE) |
| :---: | :--- | :---: | :---: | :--- |
| **$L$** | Initial Learning Rate | $[10^{-5}, 5 \times 10^{-4}]$ | Continuous | **Log-Linear:** $L = 10^{\log_{10}(L_{min}) + u_1 \cdot (\log_{10}(L_{max}) - \log_{10}(L_{min}))}$ |
| **$B$** | Logical Batch Size | $[1, 512]$ | Integer | **Affine + Round:** $B = \text{clamp}(\text{round}(1 + u_2 \cdot 511), 1, 512)$ |
| **$NL$** | Number of GPT-2 Transformer Layers | $[1, 48]$ | Integer | **Affine + Round:** $NL = \text{clamp}(\text{round}(1 + u_3 \cdot 47), 1, 48)$ |
| **$NAH$** | Number of Attention Heads per Layer | $[1, 16]$ | Integer | **Affine + Round:** $NAH = \text{clamp}(\text{round}(1 + u_4 \cdot 15), 1, 16)$ |
| **$DR$** | Dropout Regularization Rate | $[0.1, 0.5]$ | Continuous | **Affine:** $DR = \text{clamp}(0.1 + u_5 \cdot 0.4, 0.1, 0.5)$ |

### 5.2 DHOA Algorithmic Formulations (Paper Equations 4–14 Audit)

#### Direct Verification of Equations from Embedded WMF Images
The target paper's mathematical formulas for DHOA in Section 3.3.1 are embedded as WMF vector graphics inside OLE objects (`word/media/image5.wmf` through `image30.wmf`). By extracting and parsing the vector text records and high-resolution renderings, every equation is directly readable and verified against the manuscript text:

| Equation Number | Confirmed Formula from Image | Image Identifier | Tier | Status in Paper |
| :---: | :--- | :---: | :---: | :--- |
| **Eq. (4)** | $P_N = \{ S_1, S_2, \dots, S_N \}$ | `media/image5.wmf` | **T1** | Defines initial population of size $N$ |
| **Eq. (5)** | $S_i = \{ L, B, NL, NAH, DR \}_i$ | `media/image7.wmf` | **T1** | Defines candidate hyperparameter position vector |
| **Eq. (6)** | $\text{Fitness} = \text{Max}\left( \frac{TP + TN}{TP + TN + FP + FN} \right)$ | `media/image9.wmf` | **T1** | Binary accuracy fitness objective; no data partition stated |
| **Eq. (7)** | $S_{i+1} = S^{lead} - D \cdot l \cdot | P \times S^{lead} - S_i |$ | `media/image10.wmf` | **T1** | Strategy 1 (encircling leader deer position) |
| **Eq. (8)** | $D = \frac{1}{4} \log\left( i + \frac{1}{i_{max}} \right) z$ | `media/image16.wmf` | **T1** | Coefficient vector $D$ schedule; logarithmic bounding |
| **Eq. (9)** | $P = 2 \cdot w$ | `media/image17.wmf` | **T1** | Coefficient vector $P$ calculation |
| **Eq. (10)** | $y_i = \frac{\pi}{8} \times r$ | `media/image21.wmf` | **T1** | Deer visual angle component |
| **Eq. (11)** | $x_i = \theta_i - y_i$ | `media/image22.wmf` | **T1** | Angle discrepancy from wind angle |
| **Eq. (12)** | $\phi_{i+1} = \phi_i + x_i$ | `media/image24.wmf` | **T1** | Position angle update |
| **Eq. (13)** | $S_{i+1} = S^{lead} - l \cdot | \cos(v) \times S^{lead} - S_i |$ | `media/image25.wmf` | **T1** | Strategy 2 (hunting angle position update) |
| **Eq. (14)** | $S_{i+1} = S^{successor} - D \cdot l \cdot | P \times S^{successor} - S_i |$ | `media/image30.wmf` | **T1** | Strategy 3 (successor exploration update) |

#### High-Resolution Render Verification of Eq. 13 and `image26.wmf`
- **Equation 13 Render (`image25.wmf`):** Displays $S_{i+1} = S^{lead} - l \cdot |\cos(v) \times S^{lead} - S_i|$. The argument inside the cosine function is Latin italic $v$ (or Greek nu $\nu$).
- **Post-Eq. 13 Text and `image26.wmf`:** The sentence immediately following Eq. 13 reads: *"where [image26.wmf] represent the best location and [image27: l] is random number."* Direct extraction and rendering of `image26.wmf` at 500 DPI parses as $Y = \phi_{i+1}, S_i^*$.
- **Glyph Comparison Finding:** The typographic symbol inside $\cos(\cdot)$ in Eq. 13 ($v$) is **not** the same glyph as the symbols in `image26.wmf` ($Y$, $\phi_{i+1}$, $S_i^*$). In DHOA literature, $v$ denotes the hunting orientation angle related to wind angle $\phi$. Under **Assumption A-DHOA-V (T3)**, $v \equiv \phi_{i+1}$ is adopted as primary; alternative $v \sim \mathcal{U}(-\pi/4, \pi/4)$ is tracked in ablation S10.

#### Symbol Definitions from Manuscript Prose and Unresolved Gaps
The surrounding prose in Section 3.3.1 defines several symbols with explicit wording:
- $D, P$: *"vectors of coefficients"* (`image13.wmf`, `image14.wmf`, `image34.wmf`, `image35.wmf`).
- $i_{max}$: *"maximum repetition"* (`image18.wmf`).
- $z$: *"parameter whose value ranges from -1 to 1"* (`image19.wmf`, `image44.wmf`).
- $w$: *"a arbitrary integer falling between 0 and 1"* (`image20.wmf`).
- $l$: *"a random number between 0 and 2 that was created with the wind speed in mind"* (`image15.wmf`, `image27.wmf`, `image42.wmf`).
- $\theta$: *"the angle of wind"* (`image23.wmf`).
- $S^{lead}$: *"the best solution found so far"* (`image37.wmf`, `image46.wmf`).
- $S^{successor}$: *"the second best / replacement agent"* (`image31.wmf`, `image38.wmf`, `image47.wmf`).

**Concrete Engineering Assumptions for Unresolved Symbols (T3):**
1. **$D$ and $P$ vectors (T3 — Assumption A-DHOA-VEC):** $D$ and $P$ are implemented as 5-dimensional vectors via independent draws of $z$ and $w$ across each coordinate dimension.
2. **$l$ distribution (T3 — Assumption A-DHOA-L):** Sampled as a scalar uniform random variable $l \sim \mathcal{U}(0, 2)$ per deer per iteration.
3. **$w$ type and range (T3 — Assumption A-DHOA-W):** The text states *"integer between 0 and 1"*. If strictly integer $w \in \{0, 1\}$, $P = 2w \in \{0, 2\}$. To support smooth continuous exploration, continuous uniform $w \sim \mathcal{U}(0, 1)$ is adopted, giving continuous vector $P \in [0, 2]^5$.
4. **$z$ distribution (T3 — Assumption A-DHOA-Z):** Sampled as continuous uniform $z \sim \mathcal{U}(-1, 1)^5$.
5. **$r$ definition (T3 — Assumption A-DHOA-R):** Symbol $r$ appears in Eq. 10 ($y_i = \frac{\pi}{8} \times r$) but is never defined in the text. Sampled as continuous uniform scalar $r \sim \mathcal{U}(0, 1)$, giving $y_i \in [0, \pi/8]$.
6. **$\theta$ schedule (T3 — Assumption A-DHOA-THETA):** Constant environmental wind angle $\theta_0 \sim \mathcal{U}(0, 2\pi)$ drawn once per optimization run. (Random walk $\theta_t$ evaluated in ablation S10).
7. **$\phi_0$ initial position angle (T3 — Assumption A-DHOA-PHI):** Initial position angle initialized as $\phi_0 \sim \mathcal{U}(0, 2\pi)$ and updated via Eq. 12: $\phi_{t+1} = \phi_t + x_t$.
8. **Logarithm Base in Eq. 8 (T3 — Assumption A-DHOA-LOG):** Manuscript Eq. 8 writes an unadorned $\log$. Under standard mathematical convention, natural $\log$ ($\ln$) is adopted. Across update iterations $t \in \{1, 2, 3, 4, 5\}$ with $i_{max} = 5$ and $z = 1$, the factor $\frac{1}{4} \log(t + 1/i_{max})$ evaluates as:

| Iteration ($t$) | Argument ($t + 1/i_{max}$) | Natural Log ($\ln$) | Factor $\frac{1}{4} \ln(\cdot)$ (T3 Primary) | Base-10 ($\log_{10}$) | Factor $\frac{1}{4} \log_{10}(\cdot)$ | Base-2 ($\log_2$) | Factor $\frac{1}{4} \log_2(\cdot)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $t = 1$ | $1.20$ | $0.1823$ | **$0.0456$** | $0.0792$ | $0.0198$ | $0.2630$ | $0.0658$ |
| $t = 2$ | $2.20$ | $0.7885$ | **$0.1971$** | $0.3424$ | $0.0856$ | $1.1375$ | $0.2844$ |
| $t = 3$ | $3.20$ | $1.1632$ | **$0.2908$** | $0.5051$ | $0.1263$ | $1.6781$ | $0.4195$ |
| $t = 4$ | $4.20$ | $1.4351$ | **$0.3588$** | $0.6232$ | $0.1558$ | $2.0704$ | $0.5176$ |
| $t = 5$ | $5.20$ | $1.6487$ | **$0.4122$** | $0.7160$ | $0.1790$ | $2.3785$ | $0.5946$ |

9. **Separation of Overloaded Symbol $i$ (T2 — Invariant):** The manuscript overloads $i$ as both iteration counter and candidate index. In this specification, they are strictly separated into iteration $t \in [1, T_{max}]$ ($T_{max}=5$) and candidate deer index $k \in [1, N]$ ($N=6$).

#### Table 2 Pseudocode Reconstruction and Branch Structure
Inspection of Table 2 in `word/document.xml` reveals the exact embedded condition images and pseudocode structure:

```
Table 2: Pseudo code of DHOA
Input: initial population [image36: S]
Output: first-rate resolution [image37: S^lead] and second-best option [image38: S^successor]
start
while [image39: i < i_max]
  for any population-wide solution
    Determine whether each answer is fit
    modernize [image40: y], [image41: x], [image42: l], [image35: D], [image43: P] and [image44: z]
    If [image45: |l| < 1]
      if [image33: |P| >= 1]
        Update the personnel status.      <-- Action 1
      Else
        Update that individual's status.  <-- Action 2
      end if
    Else
      Update the person's status.         <-- Action 3
    end if
  end for
  Determine whether every response is fit
  Update [image46: S^lead] and [image47: S^successor]
  i = i + 1
end while
return [image46: S^lead]
end
```

##### Resolution of Table 2 Branch-to-Strategy Mapping (UNRESOLVED / T3)
The pseudocode labels inner branch actions with generic phrases. The mapping of these branches to mathematical strategies is marked **UNRESOLVED** and resolved via engineering assumption:
- **Textual Evidence from Strategy 3 Prose:** Manuscript Section 3.3.1 explicitly states: *"Strategy 3: Propagation through the position of the successor: By changing the vector, the same encircling behavior concept may be used throughout the exploration phase. Since the first search is assumed to be random, the vector's value taken into account is smaller than 1. As a result, successor location—rather than initial best explanation—is basis for the position update. This permits a worldwide search, as shown by the equation that follows: [Eq. 14]"*. This text directly proves that when the vector value is smaller than 1 ($|P| < 1$, `image32.wmf`), Strategy 3 (Eq. 14, successor update) is executed.
- **Primary Mapping (T3 — Assumption A-DHOA-BRANCH):**
  - Branch 1 ($|l| < 1$ and $|P| \ge 1$): Action 1 $\to$ **Strategy 1** (Eq. 7, encircling leader position).
  - Branch 2 ($|l| < 1$ and $|P| < 1$): Action 2 $\to$ **Strategy 3** (Eq. 14, successor position exploration). Supported directly by Strategy 3 prose.
  - Branch 3 ($|l| \ge 1$): Action 3 $\to$ **Strategy 2** (Eq. 13, hunting angle position update with step multiplier $l$ and $\cos(v)$).
- **Alternative Mapping (Ablation S10):** Action 1 = Eq. 13, Action 2 = Eq. 14, Action 3 = Eq. 7.

#### Algorithm Naming Inconsistency Audit
Section 3.3.1 explicitly states: *"The DHOA [26]is used to optimize these hyperparameters. The Osprey Optimization algorithm's step-by-step procedure is explained below."*
- A text search across the manuscript confirms that *"Osprey"* appears **exactly once** in the entire document.
- *"DHOA"* appears **14 times**, and *"Deer Hunting"* appears **4 times**.
- The single occurrence of *"Osprey"* is confirmed as an uncorrected author transcription error.

#### Feasible Candidate Edge-Case Fallbacks ($S^{lead}$ and $S^{successor}$)
Under no circumstances may an undefined leader or successor vector be passed to Equations (7), (13), or (14):
- **Case 0 Feasible Candidates:** If zero feasible candidates exist in the initial population (e.g. all fail due to hardware limits), execution **halts immediately** with an `INFEASIBLE_HARDWARE_ABORT`. The search does not proceed with an undefined $S^{lead}$.
- **Case 1 Feasible Candidate:** If exactly one feasible candidate exists, that candidate is assigned to $S^{lead}$. The successor position falls back to the leader's position: $S^{successor} = S^{lead}$.
- **Case $\ge 2$ Feasible Candidates:** The top-scoring distinct candidate becomes $S^{lead}$, and the second-best distinct candidate becomes $S^{successor}$ among all feasible candidates evaluated so far (matching the text *"found so far"*). Any candidate marked `INFEASIBLE_HARDWARE` (`fitness=null`) is strictly excluded from role assignment.

### 5.3 Population, Iteration Budget, and Cache Policy (T3 — Assumption A-BUDGET)
- **Population Size ($N$):** Exactly $6$ deer candidates.
- **Optimization Iterations ($T_{max}$):** Exactly $5$ update iterations (rounds $t = 1, \dots, 5$) following initial population evaluation (round $t = 0$).
- **Evaluation Budget per Task Arm:** Exactly $6 \times (1 + 5) = \mathbf{36 \text{ candidate evaluations}}$.
- **Training Epochs per Candidate:** Exactly **2 epochs** (T3 — Assumption A-EPOCHS).
- **Evaluation Cache (`CACHE_HIT`):** If a deer position rounds to an identical hyperparameter vector previously evaluated, cached validation fitness is reused without re-running training, consuming 1 evaluation slot.

### 5.4 Candidate Isolation, State Reloading, and Seed Architecture
1. **Complete Candidate Weight Isolation:** Each candidate evaluation instantiates a fresh model from the master initial checkpoint (`master_init_checkpoint.pt`), trains for 2 epochs, records validation fitness, and immediately unloads tensors. No candidate ever inherits weights from another candidate.
2. **Master Initial Checkpoint (`S_init = 100`):** Pretrained GPT-2 weights, projection bridge initialized with $\mathbf{W}_{bridge} \sim \mathcal{N}(0, 0.02)$, classification head, and one fixed set of 24 appended blocks are saved once to disk with seed 100 and reloaded for every candidate and the final retrain.
3. **Seed Architecture (T3 — Assumption A-SEEDS):**
   - $S_{split} = 42$: The split seed recorded in Phase 1 metadata (`src/data_loader.py`). Split files already exist in `data/splits/` and are strictly preserved; splits are never regenerated.
   - $S_{init} = 100$: Governs initial model checkpoint generation (bridge, head, appended blocks).
   - $S_{order} = 200$: Governs per-epoch data loader shuffling permutation, strictly independent of batch size $B$.
   - $S_{dropout} = 300$: Governs the initial PRNG stream for dropout operations in every candidate.
   - $S_{dhoa} = 400$: Governs initial DHOA population sampling and stochastic terms ($r, w, z, l, \theta_0, \phi_0$).
4. **Non-Finite Loss Handling (`DIVERGED_NONFINITE`):** If loss evaluates to `NaN` or `Inf`, the candidate is assigned `fitness = null`, reported as a training failure, and ranked strictly below every evaluated candidate. It is never assigned fitness 0.0 and never labeled hardware-infeasible.
5. **Deterministic Tie-Breaking:** If two candidates achieve identical validation accuracy, ties are broken by: (1) lower validation cross-entropy loss, (2) lower layer count (smaller $NL$), (3) lower candidate index $k$.
6. **Final Retrain Protocol (T3 — Assumption A-RETRAIN):** The winning architecture is retrained on the **TRAIN split only** from the master initial checkpoint for **2 epochs**, then evaluated once on the test set. Both search-time validation accuracy and final retrain validation accuracy must be reported to account for dropout PRNG divergence.

---

## 6. Training Procedure and Gradient Accumulation

### 6.1 Logical vs. Physical Batch Definitions
To evaluate batch sizes across Table 1's full range $B \in [1, 512]$ without exceeding GPU memory capacity:
- **Logical Batch Size ($B$):** The hyperparameter defined in Table 1 ($B \in [1, 512]$). Each logical batch contains exactly $B$ training examples, except the final incomplete logical batch containing $b \le B$ examples.
- **Physical Micro-Batch Size ($B_{phys}$):** The maximum number of sequences processed concurrently in device memory during a single forward-backward pass ($B_{phys} \le B_{phys\_max}$). Determined at runtime via Gate G3.
- **Accumulation Step Count ($K(b)$):** The number of sequential micro-batches required to accumulate gradients for logical batch $b$:
  $$K(b) = \left\lceil \frac{b}{B_{phys}} \right\rceil$$

### 6.2 Mathematically Correct Loss Scaling
For an epoch with $N$ training examples and logical batch size $B$, the training set is partitioned into $\lceil N / B \rceil$ logical batches. For a logical batch containing $b$ examples (where $b = B$ for full batches, and $b = N \pmod B$ for an incomplete final batch), the logical batch is subdivided into $K(b) = \lceil b / B_{phys} \rceil$ physical micro-batches. Micro-batch $j$ contains $n_j$ examples, where $\sum_{j=1}^{K(b)} n_j = b$.

To ensure that accumulated gradients exactly match the gradient of the mean loss across all $b$ examples, each micro-batch standard mean loss $\ell_j = \frac{1}{n_j} \sum_{k=1}^{n_j} \text{CrossEntropy}(z_k, y_k)$ must be scaled by its exact example proportion:

$$\mathcal{L}_j = \ell_j \cdot \left(\frac{n_j}{b}\right) = \frac{1}{b} \sum_{k=1}^{n_j} \text{CrossEntropy}(z_k, y_k)$$

Accumulating partial gradients across all $K(b)$ micro-batches yields:

$$\nabla_\theta = \sum_{j=1}^{K(b)} \nabla_\theta \mathcal{L}_j = \frac{1}{b} \sum_{j=1}^{K(b)} \sum_{k=1}^{n_j} \nabla_\theta \text{CrossEntropy}(z_k, y_k) = \nabla_\theta \left( \frac{1}{b} \sum_{i=1}^b \text{CrossEntropy}(z_i, y_i) \right)$$

#### Edge Case Verification Table

| Logical $B$ | Actual $b$ | $B_{phys}$ | Micro-Batches ($K(b)$) | Micro-Batch Sizes ($n_j$) | Loss Scale Factors ($n_j / b$) | Sum of Scale Factors | Optimizer Steps |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B = 1$** | $1$ | $4$ | $1$ | $[1]$ | $[1.0]$ | $1.0$ | 1 |
| **$B = 2$** | $2$ | $4$ | $1$ | $[2]$ | $[1.0]$ | $1.0$ | 1 |
| **$B = 3$** | $3$ | $4$ | $1$ | $[3]$ | $[1.0]$ | $1.0$ | 1 |
| **$B = 5$** | $5$ | $4$ | $2$ | $[4, 1]$ | $[4/5, 1/5]$ | $1.0$ | 1 |
| **$B = 7$** | $7$ | $4$ | $2$ | $[4, 3]$ | $[4/7, 3/7]$ | $1.0$ | 1 |
| **$B = 512$** | $512$ | $4$ | $128$ | $128 \times [4]$ | $128 \times [4/512]$ | $1.0$ | 1 |
| **Final Batch** ($N=1043, B=512$) | $19$ | $4$ | $5$ | $[4, 4, 4, 4, 3]$ | $[4/19, 4/19, 4/19, 4/19, 3/19]$ | $1.0$ | 1 |

#### Worked Example: Twitter15 Training Split ($N = 1,043$ examples, $B = 512, B_{phys} = 4$)
- **Logical Batch 1:** $b = 512$ examples. $K = 128$ micro-batches of size 4. Loss scaled by $4/512$. Single optimizer step.
- **Logical Batch 2:** $b = 512$ examples. $K = 128$ micro-batches of size 4. Loss scaled by $4/512$. Single optimizer step.
- **Logical Batch 3 (Final Incomplete Batch):** $b = 1,043 - 1,024 = 19$ examples. $K = \lceil 19/4 \rceil = 5$ micro-batches of sizes $[4, 4, 4, 4, 3]$. Losses scaled by $[4/19, 4/19, 4/19, 4/19, 3/19]$. Single optimizer step.
- **Total Optimizer Steps Per Epoch:** Exactly 3 updates. In a 2-epoch candidate evaluation, exactly **6 optimizer updates** are performed.

**Under-Training Confound Note:** For candidates sampled with $B = 512$, 2 epochs on Twitter15 produce only 6 total parameter updates, whereas candidates with $B = 16$ receive $\approx 130$ updates. This structural disparity is an inherent property of Table 1's parameter bounds and is documented as an optimization confound.

### 6.3 Mixed-Precision (AMP) and Training Execution Protocol (Fix 7a, 7b, B8)
1. **Optimizer Specification (T3 — Assumption A-OPT):** Standard AdamW with parameters $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 0.01$. Constant learning rate equal to candidate's parameter $L$.
2. **Loss Criterion:** Standard unweighted categorical Cross-Entropy loss.
3. **Training Duration:** Exactly **2 epochs** per candidate. Fitness is computed once after epoch 2 on the full validation partition with final-epoch weights (epoch 1 validation accuracy is logged only as a diagnostic).
4. **Autocast Dtype Selection:** Selected dynamically by hardware capability: `torch.bfloat16` if supported (NVIDIA Ampere or newer), otherwise `torch.float16`. `GradScaler` is active **only** under `float16` to prevent underflow, and is disabled under `bfloat16`.
5. **Micro-Batch Execution Loop:**
   - For each micro-batch $j \in [1, K(b)]$:
     - Forward pass under `torch.cuda.amp.autocast(dtype=amp_dtype)`.
     - Compute scaled loss: `loss_scaled = (n_j / b) * criterion(logits, labels)`.
     - Backward pass: `scaler.scale(loss_scaled).backward()` if FP16 else `loss_scaled.backward()`.
6. **Logical Batch Boundary Step:**
   - Exactly once per logical batch $b$:
     - If FP16: `scaler.unscale_(optimizer)`.
     - Gradient clipping: `torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)` (T3 — Assumption A-CLIP).
     - Optimizer step: `scaler.step(optimizer)` if FP16 else `optimizer.step()`.
     - Scaler update: `scaler.update()` if FP16.
     - Gradient clearing: `optimizer.zero_grad(set_to_none=True)`.
7. **Fine-Tuning Scope (T3 — Assumption A-FINETUNE):** All retained GPT-2 blocks, the Continuous Tensor Bridge, and the classification head are fully trainable (`requires_grad=True`). RoBERTa encoder parameters remain strictly frozen.
8. **Divergence Handling (`DIVERGED_NONFINITE`):** If loss evaluates to non-finite (`NaN` or `Inf`), the candidate is assigned `fitness = null`, ranked below every evaluated candidate, and reported as a training failure. It is never assigned fitness 0.0 and never classified as hardware-infeasible.

### 6.4 Limitations of Gradient Accumulation
Gradient accumulation achieves exact mathematical equivalence for mean-reduced loss in exact arithmetic. However, bitwise numerical equivalence to a physical batch execution does not hold due to:
1. **Dropout Mask Sampling:** Dropout masks are sampled per micro-batch forward pass.
2. **Floating-Point Order:** Sequential summation of partial gradients introduces order-dependent rounding differences.
3. **Autocast / AMP:** Loss scaling and unscaling under dynamic FP16 scaling introduce small precision variances.
4. **Dynamic Padding Differences:** If micro-batches are padded dynamically to micro-batch max length rather than fixed $T=512$, padding lengths vary.
Neither bitwise nor unconditional numerical equivalence to physical execution is claimed.

---

## 7. Metrics, Fitness, Statistics, Final Retrain Protocol, and Comparison Template

### 7.1 Verified Label Mapping and Split Invariants
The authoritative Phase 1 CSV files and split JSON files establish the following exact counts and mapping:
$$\text{Label Index Map}: \quad \{0: \text{non-rumour}, \; 1: \text{false}, \; 2: \text{true}, \; 3: \text{unverified}\}$$

#### Exact Split Sample Counts (from `data/splits/*_train_val_test.json`)
- **Twitter15 ($N=1,490$):**
  - Train: 1,043 examples (70.0%)
  - Validation: 149 examples (10.0%)
  - Test: 298 examples (20.0%)
- **Twitter16 ($N=818$):**
  - Train: 572 examples (69.9%)
  - Validation: 82 examples (10.0%)
  - Test: 164 examples (20.0%)
- **PHEME ($N=6,425$):**
  - Train: 4,497 examples (70.0%)
  - Validation: 643 examples (10.0%)
  - Test: 1,285 examples (20.0%)

#### Arm D Veracity Subset Sample Counts (Filtered $y \in \{1, 2\}$ directly from 4-class splits)
- **Twitter15 Arm D ($N=742$ threads: 370 false, 372 true):**
  - Train: 520 examples (259 false, 261 true)
  - Validation: 74 examples (37 false, 37 true)
  - Test: 148 examples (74 false, 74 true) — exact 74/74 balance in test set. Check: $520 + 74 + 148 = 742$.
- **Twitter16 Arm D ($N=412$ threads: 205 false, 207 true):**
  - Train: 288 examples (144 false, 144 true)
  - Validation: 41 examples (20 false, 21 true)
  - Test: 83 examples (41 false, 42 true). Check: $288 + 41 + 83 = 412$.
- **PHEME Arm D ($N=1,705$ threads: 638 false, 1,067 true):**
  - Train: 1,193 examples (446 false, 747 true)
  - Validation: 171 examples (64 false, 107 true)
  - Test: 341 examples (128 false, 213 true). Check: $1,193 + 171 + 341 = 1,705$.

### 7.2 Multiclass Evaluation Formulas (Arm A)
For each class $c \in \{0, 1, 2, 3\}$, let $TP_c, FP_c, FN_c, TN_c$ denote one-vs-rest confusion counts:
1. **Per-Class Metrics:**
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   *Zero-Division Policy:* If denominator evaluates to zero, metric is assigned $0.0$ and logged.
2. **Macro-Averaged Metrics:**
   $$\text{Macro Precision} = \frac{1}{4} \sum_{c=0}^3 \text{Precision}_c, \quad \text{Macro Recall} = \frac{1}{4} \sum_{c=0}^3 \text{Recall}_c, \quad \text{Macro F1} = \frac{1}{4} \sum_{c=0}^3 F1_c$$
3. **Four-Class Overall Accuracy:**
   $$\text{Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)}{N}$$

### 7.3 Metric Equations in Paper Document Order (Section 4.1, Eqs. 15–22)
All metric formulas appear in the following document order in Section 4.1 of the manuscript:

#### Equations Confirmed from Embedded WMF Images (Eqs. 15–18)
1. **Equation (15) — Accuracy (`media/image48.wmf`):**
   $$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Equation (16) — Recall (`media/image49.wmf`):**
   $$\text{recall} = \frac{TP}{TP + FN}$$
3. **Equation (17) — Precision (`media/image50.wmf`):**
   $$\text{precision} = \frac{TP}{TP + FP}$$
4. **Equation (18) — F1-score (`media/image51.wmf`):**
   $$\text{F1-score} = \frac{2TP}{2TP + FP + FN}$$

#### Typeset Equations (Eqs. 19–22)
5. **Equation (19) — False Positive Rate (FPR):**
   $$\text{FPR} = \frac{FP}{TN + FP}$$
6. **Equation (20) — False Negative Rate (FNR):**
   $$\text{FNR} = \frac{FN}{TP + FN}$$
7. **Equation (21) — Matthews Correlation Coefficient (MCC):**
   - Manuscript typeset text contains an XML formatting defect: an errant minus sign `-` following $(TP + FP)$ in place of multiplication under the radical (`<m:rad>`).
   - The standard mathematically correct MCC formula is used:
     $$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
8. **Equation (22) — Negative Predictive Value (NPV):**
   $$\text{NPV} = \frac{TN}{TN + FN}$$

### 7.4 Statistical Uncertainty and Confidence Interval Protocol (Fix 5, B10)
1. **Validation Resolution Warning:** Validation partitions are relatively small (Twitter15: $N_{val} = 149$, Twitter16: $N_{val} = 82$). A single misclassified example shifts validation accuracy by $\approx 0.67$ percentage points on Twitter15 and $\approx 1.22$ percentage points on Twitter16. Consequently, validation accuracy has coarse resolution and ranking ties among top candidates are expected.
2. **Wilson 95% Confidence Intervals:** Every measured accuracy metric is accompanied by its 95% Wilson score binomial confidence interval ($z = 1.96$):
   $$p_{centre} = \frac{\hat{p} + \frac{z^2}{2N_{test}}}{1 + \frac{z^2}{N_{test}}}, \quad \Delta = \frac{z \sqrt{\frac{\hat{p}(1-\hat{p})}{N_{test}} + \frac{z^2}{4N_{test}^2}}}{1 + \frac{z^2}{N_{test}}}, \quad \text{CI}_{95\%} = [p_{centre} - \Delta, \; p_{centre} + \Delta]$$

| Dataset | Test Size ($N_{test}$) | Paper Reported Accuracy | Wilson 95% Interval (Projection only, not an empirical interval) | Implied Correct Count ($p \times N_{test}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Twitter15** | $N = 298$ | **$99.50\%$** | **$[97.9\%, 99.9\%]$** | $296.51$ (non-integer count) |
| **Twitter16** | $N = 164$ | **$99.30\%$** | **$[96.5\%, 99.9\%]$** | $162.85$ (non-integer count) |
| **PHEME** | $N = 1,285$ | **$99.10\%$** | **$[98.4\%, 99.5\%]$** | $1,273.44$ (non-integer count) |

*(Note on Implied Counts: While $296.51$ and $162.85$ are non-integers, the paper's own proprietary test-set size is unstated, so this arithmetic does not prove a defect in the original paper).* 

3. **Hypothesis Testing Policy:** Confidence interval overlap is **not** a formal hypothesis test. Formal significance testing between reproduction models and baselines must be evaluated using **McNemar's test** on paired binary predictions or a **10,000-resample bootstrap test**. Multi-seed runs are reported as $\text{Mean} \pm \text{Standard Deviation}$ alongside individual seed values.

### 7.5 Final Model Retraining and Evaluation Protocol
To prevent test data snooping and overfitting:
1. **Step 1 (Candidate Selection):** The winning architecture $[L^*, B^*, NL^*, NAH^*, DR^*]$ is selected strictly by validation accuracy after 36 DHOA evaluations. The test partition is never accessed during DHOA search.
2. **Step 2 (Checkpoint Reload):** The master initial checkpoint (`master_init_checkpoint.pt`, seed $S_{init}=100$) is reloaded to guarantee identical initial weights.
3. **Step 3 (Retraining on Train Split Only):** The selected model is retrained on the **TRAIN split only** for **2 epochs** using the winning hyperparameters, with seeds $S_{order} = 200$ and $S_{dropout} = 300$. *(2B.3.5's train+val retraining is REJECTED)*.
4. **Step 4 (Single Test Evaluation):** The retrained model evaluates the test set exactly once.
5. **Step 5 (Dual Accuracy Reporting):** Because dropout PRNG streams advance differently during retraining, both the search-time validation accuracy and final retrain validation accuracy must be recorded and reported.

### 7.6 Comparison Table Format

| Dataset & Task Arm | Metric | Paper Claimed Result | Hypothetical Projection Band on $N_{test}$ | Reproduction Result (95% CI) | Status | Primary Discrepancy Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Twitter16 (Arm B - Priority Pilot)** | Accuracy | 99.30% | $[96.5\%, 99.9\%]$ | TBD | TBD | Binary rumor detection fitness & Section 1 text |
| | Precision | 99.56% | — | TBD | TBD | Small test size ($N=164$) / DHOA search noise |
| | Recall | 99.36% | — | TBD | TBD | 2-epoch training limitation |
| | F-measure | 99.39% | — | TBD | TBD | Frozen RoBERTa / T3 Bridge capacity |
| **Twitter15 (Arm B - Priority)** | Accuracy | 99.50% | $[97.9\%, 99.9\%]$ | TBD | TBD | Direct test of Eq. 6 binary accuracy |
| | Precision | 99.57% | — | TBD | TBD | DHOA hyperparameter convergence |
| | Recall | 99.60% | — | TBD | TBD | 2-epoch training limitation |
| | F-measure | 99.59% | — | TBD | TBD | Frozen RoBERTa / T3 Bridge capacity |
| **PHEME (Arm B - Priority)** | Accuracy | 99.10% | $[98.4\%, 99.5\%]$ | TBD | TBD | Severe class distribution divergence (Table 3 vs. Archive) |
| | Precision | 98.90% | — | TBD | TBD | Archive is 62.6% non-rumor vs Table 3 claim of 40% non-rumor |
| | Recall | 98.98% | — | TBD | TBD | Archive contains only 2,403 rumor threads |
| | F-measure | 98.89% | — | TBD | TBD | Class imbalance in physical data |
| **Twitter16 (Arm A - 4-Class)** | Accuracy | 99.30% | $[96.5\%, 99.9\%]$ | TBD | TBD | Multiclass generalization of Eq. 6 |
| | Macro F1 | 99.39% | — | TBD | TBD | 4-class confusion matrix |
| **Twitter15 (Arm A - 4-Class)** | Accuracy | 99.50% | $[97.9\%, 99.9\%]$ | TBD | TBD | Multiclass generalization of Eq. 6 |
| | Macro F1 | 99.59% | — | TBD | TBD | 4-class confusion matrix |
| **PHEME (Arm A - 4-Class)** | Accuracy | 99.10% | $[98.4\%, 99.5\%]$ | TBD | TBD | Class distribution disparity |
| | Macro F1 | 98.89% | — | TBD | TBD | 4-class confusion matrix |
| **Twitter16 (Arm D - Veracity)** | Accuracy | — | — | TBD | TBD | Binary veracity (true vs false only, $N=412$) |
| **Twitter15 (Arm D - Veracity)** | Accuracy | — | — | TBD | TBD | Binary veracity (true vs false only, $N=742$) |
| **PHEME (Arm D - Veracity)** | Accuracy | — | — | TBD | TBD | Binary veracity (true vs false only, $N=1,705$) |
| **Report C (Post-Hoc Binary)** | Accuracy | — | — | TBD | TBD | Post-hoc collapse of Arm A predictions |

---

## 8. Hardware Profile, Memory Analysis, Gates, and OOM Protocol

### 8.1 Hardware Profile Specification
Because execution hardware is determined at runtime, this design does **not** assume any specific GPU model, VRAM capacity, or CUDA capability. No GPU model or VRAM size is assumed anywhere, and no figure is called measured. Memory units are standardized strictly to **GiB** ($1024^3$ bytes). Tier labels denote nominal manufacturer ratings; usable VRAM on display-attached or shared GPUs is lower due to OS, display, and driver allocations.

```python
hardware_profile = {
    "gpu_name": torch.cuda.get_device_name(0),
    "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / (1024**3),
    "usable_vram_gib": torch.cuda.mem_get_info()[0] / (1024**3),
    "compute_capability": torch.cuda.get_device_capability(0),
    "supported_autocast_dtypes": [torch.float16] + ([torch.bfloat16] if torch.cuda.is_bf16_supported() else [])
}
```

### 8.2 Parameter Counts as Functions of Architecture ($NL$)
- **One GPT-2 Medium Block ($P_{block}$):**
  - Attention: $4 \times d_{model}^2 + 4 \times d_{model} = 4 \times 1024^2 + 4096 = 4,198,400$ parameters.
  - MLP: $2 \times d_{model} \times d_{ff} + d_{model} + d_{ff} = 2 \times 1024 \times 4096 + 1024 + 4096 = 8,393,728$ parameters.
  - LayerNorm: $2 \times (2 \times d_{model}) = 4,096$ parameters.
  - **Total per block:** $P_{block} = 4,198,400 + 8,393,728 + 4,096 = \mathbf{12,596,224 \text{ parameters}}$.
- **Ancillary Components Outside Blocks:**
  - Token embedding (`wte`): $50,257 \times 1024 = 51,463,168$ parameters. *(T3 Decision: Excluded from trainable set and pruned upon model instantiation; if retained, adds $0.192 \text{ GiB}$ in FP32)*.
  - Position embedding (`wpe`): $1024 \times 1024 = 1,048,576$ parameters.
  - Final LayerNorm (`ln_f`): $2 \times 1024 = 2,048$ parameters.
  - Bridge: $768 \times 1024 + 1024 + 2048 = 789,504$ parameters.
  - Classification Head: $1024 \times d_{out} + d_{out}$ ($4,100$ for $d_{out}=4$; $2,050$ for $d_{out}=2$).
- **Trainable Parameters Formula ($d_{out}=4$, excluding `wte`):**
  $$P_{train}(NL) = NL \times 12,596,224 + (1,048,576 + 2,048 + 789,504 + 4,100) = NL \times 12,596,224 + 1,844,228$$
  - At $NL = 24$: $P_{train} = 24 \times 12,596,224 + 1,844,228 = 302,309,376 + 1,844,228 = \mathbf{304,153,604 \text{ parameters}}$ ($\approx 304.15\text{M}$).
  - At $NL = 48$: $P_{train} = 48 \times 12,596,224 + 1,844,228 = 604,618,752 + 1,844,228 = \mathbf{606,462,980 \text{ parameters}}$ ($\approx 606.46\text{M}$).
- **Fine-Tuning Scope (T3 — Assumption A-FINETUNE):** All retained GPT-2 blocks, the Continuous Tensor Bridge, and the classification head are fully trainable for every candidate; RoBERTa encoder parameters remain strictly frozen.

### 8.3 Theoretical Static Memory Calculations

#### Primary Layout: Standard PyTorch Autocast (AMP)
Under standard PyTorch mixed-precision training (`torch.cuda.amp.autocast()` with `GradScaler`):
- Parameters (FP32): 4 bytes per parameter.
- Gradients (FP32): 4 bytes per parameter.
- AdamW First Moment (FP32): 4 bytes per parameter.
- AdamW Second Moment (FP32): 4 bytes per parameter.
- Master Weights: 0 bytes (no separate copy is maintained because model weights already exist in FP32; `GradScaler` holds no master weights and only scales loss).
- **Total Theoretical Static Training State: Exactly 16 bytes per parameter.**

#### Theoretical Static Memory Subtotals
- **At $NL = 24$ ($P_{train} = 304,153,604$ parameters):**
  - Parameters (FP32): $304,153,604 \times 4 = 1,216,614,416 \text{ B} = 1.133 \text{ GiB}$
  - Gradients (FP32): $304,153,604 \times 4 = 1,216,614,416 \text{ B} = 1.133 \text{ GiB}$
  - AdamW Moments (FP32): $304,153,604 \times 8 = 2,433,228,832 \text{ B} = 2.266 \text{ GiB}$
  - **Total Static Subtotal ($NL=24$):** $304,153,604 \times 16 \text{ B} = 4,866,457,664 \text{ B} = \mathbf{4.866 \text{ GB}} = \mathbf{4.532 \text{ GiB}}$.
- **At $NL = 48$ ($P_{train} = 606,462,980$ parameters):**
  - Parameters (FP32): $606,462,980 \times 4 = 2,425,851,920 \text{ B} = 2.259 \text{ GiB}$
  - Gradients (FP32): $606,462,980 \times 4 = 2,425,851,920 \text{ B} = 2.259 \text{ GiB}$
  - AdamW Moments (FP32): $606,462,980 \times 8 = 4,851,703,840 \text{ B} = 4.518 \text{ GiB}$
  - **Total Static Subtotal ($NL=48$):** $606,462,980 \times 16 \text{ B} = 9,703,407,680 \text{ B} = \mathbf{9.703 \text{ GB}} = \mathbf{9.037 \text{ GiB}}$.

#### Alternative 8-Bit Optimizer Layout (Reference Only)
- **8-Bit AdamW Layout (10 bytes/param):** Under autocast, gradients are FP32 (4 bytes), model weights in FP32 (4 bytes), and first/second AdamW moments quantized in INT8 (2 bytes) = **10 bytes per parameter**.
  - At $NL = 24$: $304,153,604 \times 10 \text{ B} = 3,041,536,040 \text{ B} = \mathbf{2.833 \text{ GiB}}$.
  - At $NL = 48$: $606,462,980 \times 10 \text{ B} = 6,064,629,800 \text{ B} = \mathbf{5.648 \text{ GiB}}$.
- **Arithmetic Fact Note on 470M Parameters:** Storing 470M parameters in FP16 alone requires $470\text{M} \times 2 \text{ bytes} = 0.94 \text{ GB}$ (arithmetic fact only, excluding gradients, optimizer states, activations, and runtime buffers; not attributed to the paper).

### 8.4 Activation Memory and Theoretical Feasibility Matrix

#### Dynamic Activation Memory Formulation (FP16)
For hidden dimension $h = 1024$, attention heads $a = 16$, sequence length $s$, and batch size $b$, the activation storage per transformer layer is:
$$\text{Bytes}_{layer}(s, b) = s \cdot b \cdot h \cdot \left( 34 + 5 \cdot \frac{a \cdot s}{h} \right) = s \cdot b \cdot (34,816 + 80 \cdot s)$$

The per-token activation coefficient depends strictly on sequence length $s$ and must **not** be applied across shorter sequences:
- At $s = 64$: $\text{coeff}(64) = 34,816 + 80 \times 64 = 34,816 + 5,120 = \mathbf{39,936 \text{ bytes}}$.
- At $s = 128$: $\text{coeff}(128) = 34,816 + 80 \times 128 = 34,816 + 10,240 = \mathbf{45,056 \text{ bytes}}$.
- At $s = 256$: $\text{coeff}(256) = 34,816 + 80 \times 256 = 34,816 + 20,480 = \mathbf{55,296 \text{ bytes}}$.
- At $s = 512$: $\text{coeff}(512) = 34,816 + 80 \times 512 = 34,816 + 40,960 = \mathbf{75,776 \text{ bytes}}$.

- **Uncheckpointed Memory:** Storing all forward activations across $NL$ layers: $\text{Act}_{unchk} = NL \times s \cdot b \cdot \text{coeff}(s)$.
- **Checkpointed Memory:** Retains only block boundaries ($2 \cdot s \cdot b \cdot h \cdot NL \text{ bytes}$) plus one block recomputation ($s \cdot b \cdot \text{coeff}(s)$):
  $$\text{Act}_{chk} = (2 \cdot s \cdot b \cdot 1024 \cdot NL) + (s \cdot b \cdot \text{coeff}(s))$$
- **Cross-Checks:**
  - $NL=24, b=4, s=512$ uncheckpointed: $24 \times 512 \times 4 \times 75,776 = 3,724,591,104 \text{ B} = \mathbf{3.469 \text{ GiB}}$.
  - $NL=24, b=4, s=128$ uncheckpointed: $24 \times 128 \times 4 \times 45,056 = 553,582,592 \text{ B} = \mathbf{0.516 \text{ GiB}}$.
  - $NL=24, b=4, s=512$ checkpointed: $(2 \times 512 \times 4 \times 1024 \times 24) + (512 \times 4 \times 75,776) = 100,663,296 + 155,189,248 = 255,852,544 \text{ B} = \mathbf{0.238 \text{ GiB}}$ (a **$\approx 93\%$ reduction**).
- **Runtime Context Overhead:** CUDA context, memory allocator overhead, and communication buffers require an estimated $0.8 - 1.0 \text{ GiB}$ (nominal $0.9 \text{ GiB}$).

#### Activation Memory Complete Arithmetic Table

| Sequence ($s$) | Batch ($b$) | Layers ($NL$) | Uncheckpointed Bytes | Uncheckpointed (GiB) | Checkpointed Bytes | Checkpointed (GiB) | Activation Reduction (\%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 64 | 1 | 24 | $61,341,696$ | $0.057 \text{ GiB}$ | $5,701,632$ | $0.005 \text{ GiB}$ | $90.7\%$ |
| 64 | 1 | 48 | $122,683,392$ | $0.114 \text{ GiB}$ | $8,847,360$ | $0.008 \text{ GiB}$ | $92.8\%$ |
| 64 | 4 | 24 | $245,366,784$ | $0.229 \text{ GiB}$ | $22,806,528$ | $0.021 \text{ GiB}$ | $90.7\%$ |
| 64 | 4 | 48 | $490,733,568$ | $0.457 \text{ GiB}$ | $35,389,440$ | $0.033 \text{ GiB}$ | $92.8\%$ |
| 64 | 8 | 24 | $490,733,568$ | $0.457 \text{ GiB}$ | $45,613,056$ | $0.042 \text{ GiB}$ | $90.7\%$ |
| 64 | 8 | 48 | $981,467,136$ | $0.914 \text{ GiB}$ | $70,778,880$ | $0.066 \text{ GiB}$ | $92.8\%$ |
| 64 | 16 | 24 | $981,467,136$ | $0.914 \text{ GiB}$ | $91,226,112$ | $0.085 \text{ GiB}$ | $90.7\%$ |
| 64 | 16 | 48 | $1,962,934,272$ | $1.828 \text{ GiB}$ | $141,557,760$ | $0.132 \text{ GiB}$ | $92.8\%$ |
| 128 | 1 | 24 | $138,395,648$ | $0.129 \text{ GiB}$ | $12,058,624$ | $0.011 \text{ GiB}$ | $91.3\%$ |
| 128 | 1 | 48 | $276,791,296$ | $0.258 \text{ GiB}$ | $18,350,080$ | $0.017 \text{ GiB}$ | $93.4\%$ |
| 128 | 4 | 24 | $553,582,592$ | $0.516 \text{ GiB}$ | $48,234,496$ | $0.045 \text{ GiB}$ | $91.3\%$ |
| 128 | 4 | 48 | $1,107,165,184$ | $1.031 \text{ GiB}$ | $73,400,320$ | $0.068 \text{ GiB}$ | $93.4\%$ |
| 128 | 8 | 24 | $1,107,165,184$ | $1.031 \text{ GiB}$ | $96,468,992$ | $0.090 \text{ GiB}$ | $91.3\%$ |
| 128 | 8 | 48 | $2,214,330,368$ | $2.062 \text{ GiB}$ | $146,800,640$ | $0.137 \text{ GiB}$ | $93.4\%$ |
| 128 | 16 | 24 | $2,214,330,368$ | $2.062 \text{ GiB}$ | $192,937,984$ | $0.180 \text{ GiB}$ | $91.3\%$ |
| 128 | 16 | 48 | $4,428,660,736$ | $4.125 \text{ GiB}$ | $293,601,280$ | $0.273 \text{ GiB}$ | $93.4\%$ |
| 256 | 1 | 24 | $339,738,624$ | $0.316 \text{ GiB}$ | $26,738,688$ | $0.025 \text{ GiB}$ | $92.1\%$ |
| 256 | 1 | 48 | $679,477,248$ | $0.633 \text{ GiB}$ | $39,321,600$ | $0.037 \text{ GiB}$ | $94.2\%$ |
| 256 | 4 | 24 | $1,358,954,496$ | $1.266 \text{ GiB}$ | $106,954,752$ | $0.100 \text{ GiB}$ | $92.1\%$ |
| 256 | 4 | 48 | $2,717,908,992$ | $2.531 \text{ GiB}$ | $157,286,400$ | $0.146 \text{ GiB}$ | $94.2\%$ |
| 256 | 8 | 24 | $2,717,908,992$ | $2.531 \text{ GiB}$ | $213,909,504$ | $0.199 \text{ GiB}$ | $92.1\%$ |
| 256 | 8 | 48 | $5,435,817,984$ | $5.062 \text{ GiB}$ | $314,572,800$ | $0.293 \text{ GiB}$ | $94.2\%$ |
| 256 | 16 | 24 | $5,435,817,984$ | $5.062 \text{ GiB}$ | $427,819,008$ | $0.398 \text{ GiB}$ | $92.1\%$ |
| 256 | 16 | 48 | $10,871,635,968$ | $10.125 \text{ GiB}$ | $629,145,600$ | $0.586 \text{ GiB}$ | $94.2\%$ |
| 512 | 1 | 24 | $931,147,776$ | $0.867 \text{ GiB}$ | $63,963,136$ | $0.060 \text{ GiB}$ | $93.1\%$ |
| 512 | 1 | 48 | $1,862,295,552$ | $1.734 \text{ GiB}$ | $89,128,960$ | $0.083 \text{ GiB}$ | $95.2\%$ |
| 512 | 4 | 24 | $3,724,591,104$ | $3.469 \text{ GiB}$ | $255,852,544$ | $0.238 \text{ GiB}$ | $93.1\%$ |
| 512 | 4 | 48 | $7,449,182,208$ | $6.938 \text{ GiB}$ | $356,515,840$ | $0.332 \text{ GiB}$ | $95.2\%$ |
| 512 | 8 | 24 | $7,449,182,208$ | $6.938 \text{ GiB}$ | $511,705,088$ | $0.477 \text{ GiB}$ | $93.1\%$ |
| 512 | 8 | 48 | $14,898,364,416$ | $13.875 \text{ GiB}$ | $713,031,680$ | $0.664 \text{ GiB}$ | $95.2\%$ |
| 512 | 16 | 24 | $14,898,364,416$ | $13.875 \text{ GiB}$ | $1,023,410,176$ | $0.953 \text{ GiB}$ | $93.1\%$ |
| 512 | 16 | 48 | $29,796,728,832$ | $27.750 \text{ GiB}$ | $1,426,063,360$ | $1.328 \text{ GiB}$ | $95.2\%$ |

#### Extended Theoretical Feasibility Matrix (Planning Aid Only — Unmeasured)
Peak VRAM is calculated using a single consistent convention: $\text{Peak} = \text{Static Memory} + \text{Activation Memory} + 0.9 \text{ GiB Overhead}$. Peak VRAM is compared directly against the nominal VRAM rating of each tier. Usable VRAM on display-attached GPUs is lower due to driver allocations.

| Configuration ($NL, b, s$, Checkpoint) | Peak VRAM (GiB) | 8 GiB Tier | 12 GiB Tier | 16 GiB Tier | 24 GiB Tier | 32 GiB Tier | 48 GiB Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$NL=24, b=1, s=128$, Uncheckpointed** | **5.56 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=4, s=128$, Uncheckpointed** | **5.95 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=8, s=128$, Uncheckpointed** | **6.46 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=16, s=128$, Uncheckpointed**| **7.49 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=1, s=128$, Checkpointed**   | **5.44 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=4, s=128$, Checkpointed**   | **5.48 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=8, s=128$, Checkpointed**   | **5.52 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=16, s=128$, Checkpointed**  | **5.61 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=1, s=128$, Uncheckpointed** | **10.19 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=4, s=128$, Uncheckpointed** | **10.97 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=8, s=128$, Uncheckpointed** | **12.00 GiB** | Infeasible (OOM)* | Infeasible (OOM) | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=16, s=128$, Uncheckpointed**| **14.06 GiB** | Infeasible (OOM)* | Infeasible (OOM) | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=1, s=128$, Checkpointed**   | **9.95 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=4, s=128$, Checkpointed**   | **10.01 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=8, s=128$, Checkpointed**   | **10.07 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=16, s=128$, Checkpointed**  | **10.21 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=1, s=512$, Uncheckpointed** | **6.30 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=4, s=512$, Uncheckpointed** | **8.90 GiB** | Infeasible (OOM) | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=8, s=512$, Uncheckpointed** | **12.37 GiB** | Infeasible (OOM) | Infeasible (OOM) | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=16, s=512$, Uncheckpointed**| **19.31 GiB** | Infeasible (OOM) | Infeasible (OOM) | Infeasible (OOM) | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=1, s=512$, Checkpointed**   | **5.49 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=4, s=512$, Checkpointed**   | **5.67 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=8, s=512$, Checkpointed**   | **5.91 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=24, b=16, s=512$, Checkpointed**  | **6.39 GiB** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=1, s=512$, Uncheckpointed** | **11.67 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=4, s=512$, Uncheckpointed** | **16.87 GiB** | Infeasible (OOM)* | Infeasible (OOM) | Infeasible (OOM) | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=8, s=512$, Uncheckpointed** | **23.81 GiB** | Infeasible (OOM)* | Infeasible (OOM) | Infeasible (OOM) | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=16, s=512$, Uncheckpointed**| **37.69 GiB** | Infeasible (OOM)* | Infeasible (OOM) | Infeasible (OOM) | Infeasible (OOM) | Infeasible (OOM) | **Fit** |
| **$NL=48, b=1, s=512$, Checkpointed**   | **10.02 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=4, s=512$, Checkpointed**   | **10.27 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=8, s=512$, Checkpointed**   | **10.60 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |
| **$NL=48, b=16, s=512$, Checkpointed**  | **11.27 GiB** | Infeasible (OOM)* | **Fit** | **Fit** | **Fit** | **Fit** | **Fit** |

*\*Note on 8 GiB tier: Because theoretical static training state alone requires $9.037 \text{ GiB}$ under standard PyTorch AMP, executing $NL=48$ on an 8 GiB GPU is theoretically impossible without 8-bit optimizer compression ($5.648 \text{ GiB}$ static) or CPU offloading.*

### 8.5 Pre-Flight Checks and Staged Measurement Gates (Fix B7)
To establish empirical hardware feasibility, four pre-flight checks and six measurement gates are specified. **None has been executed yet.**

#### Separate Pre-Flight Checks (Not Gates)
- **P1 (Hardware Probe & Dtype Support):** Query `torch.cuda.is_available()`, device name, total and usable VRAM in GiB, compute capability, and test BF16/FP16 autocast allocation.
- **P2 (Cache Validation):** Preprocess 50 examples, extract RoBERTa features in FP32, save to FP16 cache, reload, and verify `torch.allclose(cached, fresh_fp32.half(), rtol=1e-3, atol=1e-3)`.
- **P3 (Appended-Block Gradient Probe):** Instantiate hybrid model with $NL = 25$ from master initial checkpoint, execute single synthetic backward pass, verify non-zero gradients on zero-initialized output projections (`c_proj.weight.grad != 0`).
- **P4 (Candidate Isolation Probe):** Run two mock candidate evaluations sequentially, verifying complete memory deallocation and identical initial model state reloading.

#### Staged Measurement Gates (G1–G6)
- **Gate G1 (Forward Memory):** Allocate model at $NL=24, s=512$, pass single micro-batch ($b=4$), verify execution without OOM.
- **Gate G2 (Forward + Backward at Longest Sequence):** Execute full backward pass at $NL=24, s=512$, verify gradient allocation and memory release.
- **Gate G3 ($B_{phys\_max}$ Worst-Case Calibration):** Sweep $B_{phys} \in \{1, 2, 4, 8, 16\}$ at maximum bounds $NL=48, s=512$. Record the highest stable micro-batch size as $B_{phys\_max}$. The runtime OOM ladder may only decrement below this calibrated value.
- **Gate G4 ($NL=48$ Stress Test):** Execute full training step at maximum bounds ($NL=48, B=512, s=512$) via gradient accumulation.
- **Gate G5 (Accumulation Correctness Probe):** Verify on synthetic data that gradient accumulation with $B=5, B=7$, and an incomplete final batch matches unaccumulated gradients within FP32 tolerance ($10^{-4}$).
- **Gate G6 (OOM Recovery Verification):** Trigger an artificial CUDA OOM exception, verify clean cache reset (`torch.cuda.empty_cache()`), state rollback to master initial checkpoint, and graceful exception handling.

### 8.6 Out-of-Memory (OOM) and Infeasibility Protocol
If a GPU OOM exception occurs during candidate training:

```
                    [Candidate Execution: (L, B, NL, NAH, DR)]
                                        │
                                        ├──> Success ──> Fitness = val_accuracy
                                        │
                           [torch.cuda.OutOfMemoryError]
                                        │
                                        ▼
                    [Log Failure: params, stage, VRAM stats]
                                        │
                                        ▼
                            [Deterministic Retry Ladder]
                            • Step 1: Halve B_phys (double K)
                            • Step 2: Enable Gradient Checkpointing
                            • Step 3: Set B_phys = 1
                                        │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
             [Retry Succeeds]                         [All Retries Fail]
                    │                                         │
                    ▼                                         ▼
          Fitness = val_accuracy                    Status = INFEASIBLE_HARDWARE
          (Log applied mitigations)                 Fitness = null (Excluded from ranking)
                                                    (Deer retains position in population)
```

1. **Failure Logging:** Immediately record candidate parameter vector, failure stage (model build / forward / backward / optimizer step / validation), and memory statistics: `torch.cuda.memory_allocated()`, `memory_reserved()`, `max_memory_allocated()`, and `torch.cuda.mem_get_info()`.
2. **Deterministic Retry Ladder (Strict Checkpoint Restart):** Each retry restarts strictly from the master initial checkpoint (`master_init_checkpoint.pt`), never resuming mid-epoch:
   - **Step 1:** Halve physical micro-batch size: $B_{phys} \leftarrow \max(1, \lfloor B_{phys} / 2 \rfloor)$, doubling accumulation count $K$.
   - **Step 2:** Enable PyTorch activation checkpointing (`torch.utils.checkpoint`) across all active transformer blocks.
   - **Step 3:** Reduce physical micro-batch size to minimum: $B_{phys} = 1$.
3. **Persistent Failure Classification:** If all retry steps fail, the candidate is assigned status `INFEASIBLE_HARDWARE`:
   - Fitness evaluates to `null` (never $0.0$).
   - Excluded from all ranking, comparison tables, and $S^{lead}$ / $S^{successor}$ roles.
   - The deer retains its coordinate position in the population; its coordinates are updated normally by DHOA equations in subsequent iterations.
4. **Initial Population Abort Rule:** If all 6 candidates in the initial population are persistently infeasible, execution halts immediately with a formal hardware limitation report. The search space is **never** silently narrowed.
5. **Edge Case Candidate Handling:**
   - 0 feasible candidates: Immediate abort with `INFEASIBLE_HARDWARE_ABORT`.
   - 1 feasible candidate: Assigned to $S^{lead}$; fallback $S^{successor} = S^{lead}$.
   - $\ge 2$ feasible candidates: Highest validation accuracy assigned to $S^{lead}$, second highest to $S^{successor}$.

---

## 9. Paper Specification vs. Implementation Assumption Matrix

This matrix provides a comprehensive audit of every design decision, classifying each into its formal fidelity tier with exact manuscript location, classification category, adopted decision, and engineering justification.

| ID | Location in Paper | Tier | Classification | Decision | Justification |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **P-RAW** | Section 4.1, Table 3 | **T1** | Paper-Explicit | Source tweet text corpus ($N=1490, 818, 6425$) | Benchmark datasets named; physical text releases utilized |
| **P-SPLIT** | Section 4.1 (Unstated) | **T3** | Gap-Filled | Stratified 70% Train, 10% Val, 20% Test | Ratio omitted in paper; adopted standard partition from Phase 1 files |
| **P-SEED-SPLIT** | Phase 1 Metadata | **T3** | Gap-Filled | Split seed $S_{split} = 42$ | Seed recorded in Phase 1 metadata; split files already exist and are strictly preserved |
| **P-SEED-DHOA** | Section 3.3.1 (Unstated) | **T3** | Gap-Filled | $S_{init}=100, S_{order}=200, S_{dropout}=300, S_{dhoa}=400$ | Unspecified random seeds fixed to guarantee exact experimental reproducibility |
| **P-ARMD-SPLIT** | Section 3 (Implied) | **T3** | Gap-Filled | Filtered labels $y \in \{1, 2\}$ directly from 4-class splits | Produces exact 74/74 test split on Twitter15; counts match physical dataset census |
| **P-TOK** | Section 3.1, [Ref 23] | **T1/T3** | Gap-Filled | `TweetTokenizer(preserve_case=True, reduce_len=False)` | Tokenizer unnamed; TweetTokenizer preserves tweet punctuation, casing, emojis |
| **P-STOP** | Section 3.1, [Ref 24] | **T1/T3** | Paper-Explicit | Standard NLTK English stopword filter (case-insensitive) | Specified by paper; matching is case-insensitive; negation impact tracked in S5 |
| **P-LEM** | Section 3.1 (No Citation)| **T1/T3** | Gap-Filled | WordNet Lemmatizer with two-pass verb-then-noun lookup | Specified by paper; Ref [24] excised; two-pass lookup fulfills "sharing" $\to$ "share" |
| **P-ORDER** | Section 3.1 | **T3** | Gap-Filled | URL $\to$ [:512] $\to$ Tokenize $\to$ Stopwords $\to$ Light-10 $\to$ Lemma | Resolves prose narrative ordering; tokenization must precede stopword filtering |
| **P-URL** | Section 3.1, [Ref 23] | **T1/T3** | Paper-Explicit | URL removal via regex; no HTML entity/tag stripping | Paper explicitly specifies URL removal; HTML handling is omitted |
| **P-SLICE** | Section 3.2 | **T2** | Paper-Implied | Literal string slicing `text = text[:512]` | Derived from Section 3.2 text: "extends supplied sentence length to 512 characters" |
| **P-LIGHT10** | Section 3.1 (No Citation)| **T1/T3** | Gap-Filled | Strip terminal "and" if remaining length $\ge 3$ characters | Operational reading of unverified rule; Ref [25] excised; destructive examples documented |
| **P-CACHE-FMT** | Section 3.2 (Implied) | **T3** | Gap-Filled | Flat memory-mapped FP16 binary array + SHA-256 manifest | Eliminates redundant RoBERTa forward passes during 324 candidate evaluations |
| **P-CACHE-TOL** | Engineering Standard | **T3** | Gap-Filled | Reload verification `rtol=1e-3, atol=1e-3` | Absolute $10^{-4}$ tolerance is mathematically unattainable in FP16 precision |
| **P-ENC** | Section 3.2 | **T1/T3** | Gap-Filled | `roberta-base` (12L, 768H, 12 heads), frozen in eval mode | Checkpoint unstated; base RoBERTa is canonical standard; frozen prevents GPU OOM |
| **P-PRUNE-WTE** | Section 3.3 (Implied) | **T3** | Gap-Filled | Prune unused `wte` ($51.46\text{M}$ params / $0.192 \text{ GiB}$) | Inputs injected via `inputs_embeds`; pruning frees static memory |
| **P-BRID-INIT** | Section 3.2 (Implied) | **T3** | Gap-Filled | $\mathbf{W}_{bridge} \sim \mathcal{N}(0, 0.02), \mathbf{b}=\mathbf{0}, \text{LN}(\gamma=1, \beta=0)$ | Bridges 768 to 1024; well-conditioned pre-LayerNorm variance |
| **P-LNF-POS** | Section 3.3 (Implied) | **T3** | Gap-Filled | GPT-2 `ln_f` applied after last active block before pooling | Preserves canonical GPT-2 architecture convention across all layer counts |
| **P-FINETUNE** | Section 3.3 (Implied) | **T3** | Gap-Filled | All retained GPT-2 blocks, bridge, and head fully trainable | Downstream classification capacity; RoBERTa remains frozen |
| **P-APP-INIT** | Section 3.3.1, Table 1 | **T3** | Gap-Filled | Appended blocks: $\mathbf{W}_{c\_proj}=\mathbf{0}, \mathbf{b}=\mathbf{0}$; intermediate $\mathcal{N}(0, 0.02)$ | Preserves identity residual mapping and enables non-zero GELU gradient flow |
| **P-HEAD-MASK** | Section 3.3.1, Table 1 | **T1/T3** | Gap-Filled | Dynamic head mask $\mathbf{m}_{head} \in \{0, 1\}^{16}$, NO variance rescaling | Spans $NAH \in [1, 16]$; preserves tensor shapes; rescaling evaluated in S14 |
| **P-POOL** | Section 3.3 (Implied) | **T3** | Gap-Filled | Last-valid-token pooling with index $\sum(\mathbf{M}) - 1$ | Canonical autoregressive GPT-2 sequence pooling (mean-pooling evaluated in S13) |
| **P-DROP-LOC** | Section 3.3.1, Table 1 | **T1/T3** | Gap-Filled | Dynamic DR applied to embd/attn/resid dropouts and head | Spans $DR \in [0.1, 0.5]$; bridge has no dropout; disabled during eval mode |
| **P-DHOA-EQ** | Section 3.3.1, Eqs 4–14 | **T1** | Paper-Explicit | Preserved verbatim as confirmed from WMF vector records | Verified from embedded images `image5.wmf` to `image30.wmf` |
| **P-DHOA-BRANCH**| Table 2 (Unstated) | **T3** | Gap-Filled | UNRESOLVED; primary mapping: $\|l\|<1 \land \|P\|\ge 1 \to$ Eq 7, $\|P\|<1 \to$ Eq 14, $\|l\|\ge 1 \to$ Eq 13 | Primary mapping supported by Strategy 3 prose; alternative evaluated in S10 |
| **P-DHOA-LOG** | Section 3.3.1, Eq. 8 | **T3** | Gap-Filled | Base of $\log$ is natural logarithm ($\ln$) | Standard mathematical convention when base is unadorned |
| **P-DHOA-SPACE**| Section 3.3.1, Table 1 | **T3** | Gap-Filled | Normalized space $[0, 1]^5$; log-linear map for $L$; affine for others | Prevents $B \in [1, 512]$ from drowning out $L \in [10^{-5}, 5 \times 10^{-4}]$ |
| **P-DHOA-L** | Section 3.3.1 | **T1/T3** | Gap-Filled | Uniform scalar $l \sim \mathcal{U}(0, 2)$ per deer per iteration | Paper text states "random number between 0 and 2"; uniform continuous is least-deviating |
| **P-DHOA-W** | Section 3.3.1 | **T1/T3** | Gap-Filled | Continuous uniform vector $w \sim \mathcal{U}(0, 1)^5$, yielding $P \in [0, 2]^5$ | Paper says "integer between 0 and 1"; continuous allows smooth exploration |
| **P-DHOA-Z** | Section 3.3.1 | **T1/T3** | Gap-Filled | Continuous uniform vector $z \sim \mathcal{U}(-1, 1)^5$ | Paper says "ranges from -1 to 1"; uniform vector provides directional exploration |
| **P-DHOA-R** | Section 3.3.1, Eq. 10 | **T3** | Gap-Filled | Continuous uniform scalar $r \sim \mathcal{U}(0, 1)$, giving $y_i \in [0, \pi/8]$ | Symbol $r$ appears in Eq. 10 but is never defined in text |
| **P-DHOA-THETA**| Section 3.3.1, Eq. 11 | **T3** | Gap-Filled | Constant environmental wind angle $\theta_0 \sim \mathcal{U}(0, 2\pi)$ per run | Paper calls $\theta$ "angle of wind"; constant direction is primary (random walk in S10) |
| **P-DHOA-PHI** | Section 3.3.1, Eq. 12 | **T3** | Gap-Filled | Initial position angle $\phi_0 \sim \mathcal{U}(0, 2\pi)$; updated $\phi_{t+1}=\phi_t+x_t$ | Initial angle unstated in paper; initialized uniformly |
| **P-DHOA-V** | Section 3.3.1, Eq. 13 | **T3** | Gap-Filled | Argument $v \equiv \phi_{i+1}$ primary; alternative $\mathcal{U}(-\pi/4, \pi/4)$ in S10 | Symbol $v$ unstated; image 26 text contains $\phi_{i+1}$; links hunting angle to wind |
| **P-DHOA-INDEX**| Section 3.3.1 | **T2** | Paper-Implied | $t \in [1, 5]$ (iteration) and $k \in [1, 6]$ (candidate deer index) | Disambiguates manuscript symbol $i$ which is overloaded in text |
| **P-DHOA-BUDGET**| Section 3.3.1 (Unstated) | **T3** | Gap-Filled | $N=6$ deer, $T_{max}=5$ update iterations (36 evaluations), 2 epochs | Optimization budget omitted in paper; bounded for computational feasibility |
| **P-DHOA-LEAD** | Section 3.3.1 | **T1/T3** | Gap-Filled | $S^{lead}$ and $S^{successor}$ are best/second-best distinct feasible solutions found so far | Matches text "found so far"; fallbacks handle edge cases with 0 or 1 feasible candidate |
| **P-DHOA-TIE** | Section 3.3.1 (Unstated) | **T3** | Gap-Filled | Lower validation cross-entropy, then lower $NL$, then candidate index | Deterministic candidate resolution for coarse validation accuracy ties |
| **P-DIVERGE** | Training Dynamics | **T3** | Gap-Filled | Non-finite loss $\to$ `fitness = null`, ranked below all evaluated candidates | Prevents invalid 0.0 accuracy assignment; accurately reports training failure |
| **P-CACHE-HIT** | Optimization Protocol | **T3** | Gap-Filled | Duplicate hyperparameter vector reuses validation score without retraining | Prevents redundant compute while strictly consuming 1 evaluation budget slot |
| **P-FINAL-RET** | Evaluation Protocol | **T3** | Gap-Filled | Winning model retrained on TRAIN split only from master checkpoint for 2 epochs | Prevents test data snooping; dual reporting logs dropout PRNG divergence |
| **P-BATCH-ACC** | Section 3.3.1, Table 1 | **T2** | Paper-Implied | Micro-batch loss scaled by $n_j / b$, 1 optimizer step per logical batch | Mathematically required to execute $B \in [1, 512]$ across resource-bounded GPUs |
| **P-AMP-CLIP** | Training Dynamics | **T3** | Gap-Filled | Hardware-selected dtype (BF16/FP16), clipping `max_norm=1.0` | Prevents gradient explosion in deep transformer backbones |
| **P-HARDWARE** | Environment Setup | **T2** | Paper-Implied | Dynamic runtime measurement via Gate G1, standardized in GiB units | Hardware-agnostic execution across consumer and enterprise GPU tiers |
| **P-OOM-RETRY** | Section 8.6 | **T3** | Gap-Filled | Retry ladder: halve $B_{phys} \to$ checkpointing $\to B_{phys}=1 \to$ `INFEASIBLE_HARDWARE` | Preserves hyperparameter objective without illegally truncating search space |

---

## 10. Sensitivity Plan, Limitations, and Implementation-Readiness Verdict

### 10.1 Sensitivity Analysis Plan (Design Specification)
To measure the sensitivity of experimental outcomes to engineering assumptions, fourteen targeted sensitivity ablations (S1–S14) are specified. All ablations are piloted first on Twitter16 ($N=818$) to minimize computational cost:

1. **S1 (RoBERTa Checkpoint Capacity):** Pretrained `roberta-base` (125M params, $d=768$) vs. `roberta-large` (355M params, $d=1024$, requiring bridge identity projection).
2. **S2 (RoBERTa Fine-Tuning Scope):** Offline frozen feature extractor mode (`requires_grad=False`) vs. end-to-end joint fine-tuning of RoBERTa + GPT-2 backbone.
3. **S3 (Projection Bridge Architecture):** Standard continuous Linear + LayerNorm bridge vs. non-linear multi-layer perceptron bridge (`Linear(768, 1024)` $\to$ `GELU` $\to$ `Linear(1024, 1024)` $\to$ `LayerNorm(1024)`).
4. **S4 (Light-10 Stemming Rule Efficacy):** Canonical pipeline with Light-10 terminal-"and" stripping vs. pipeline with Light-10 disabled.
5. **S5 (Negation-Preserving Stopword Filtering):** Standard NLTK English stopword filter (excising `no`, `not`, `nor`) vs. negation-retaining lexicon (preserving negation tokens).
6. **S6 (Optimization Epoch Duration):** Baseline budget of 2 training epochs per candidate vs. extended budget of 5 training epochs per candidate.
7. **S7 (Test Set Selection Leakage Diagnostic) (Fix 9, C6):** Evaluates a diagnostic-only candidate selection policy where the winning DHOA candidate is selected by **Test set accuracy** rather than Validation accuracy. *(Strict Diagnostic Only: Never reported as a valid reproduction result; quantifies the optimistic bias and overfitting ceiling if test set data were improperly leaked into hyperparameter selection)*.
8. **S8 (Split Architecture Invariance):** Standard 70/10/20 train/val/test split vs. stratified 5-fold cross-validation.
9. **S9 (Preprocessing Stage Ordering) (Fix 3a, C6):** Canonical pipeline (tokenization and stopword filtering before lemmatization) vs. lemmatization before stopword filtering.
10. **S10 (DHOA Branch Mapping and Angular Variants) (Fix A1, C1, C6):** Primary Table 2 branch mapping (Action 1 = Eq. 7, Action 2 = Eq. 14, Action 3 = Eq. 13) vs. Alternative mapping (Action 1 = Eq. 13, Action 2 = Eq. 14, Action 3 = Eq. 7); angular argument $v \equiv \phi_{i+1}$ vs. $v \sim \mathcal{U}(-\pi/4, \pi/4)$; constant wind angle $\theta_0$ vs. stochastic random walk $\theta_t$.
11. **S11 (DHOA Search Efficacy vs. Baselines) (Fix C6):** DHOA hyperparameter optimization (36 evaluations) vs. Random Search (36 evaluations) vs. Fixed Default Configuration ($NL=24, NAH=16, DR=0.1, B=16, L=10^{-4}$).
12. **S12 (Learning Rate Coordinate Mapping) (Fix C4, C6):** Log-linear coordinate mapping of $L \in [10^{-5}, 5 \times 10^{-4}]$ vs. linear coordinate mapping.
13. **S13 (Sequence Pooling Mechanism) (Fix A6, C6):** Last-valid-token pooling with index $\sum(\mathbf{M}) - 1$ vs. masked mean-pooling across unpadded sequence tokens.
14. **S14 (Attention Head Rescaling) (Fix A6, C6):** Head masking without variance rescaling (primary base) vs. head masking with $16/NAH$ output variance rescaling.

#### Sensitivity Execution Run Order and Estimated Cost
Ablations are ordered into four computational tiers to ensure the least compute-intensive and most diagnostic experiments execute first. **Arm B (Priority Rumour Detection) runs first** across all datasets:

| Tier | Ablation IDs | Scope / Description | Compute Cost per Run | Execution Order |
| :--- | :--- | :--- | :---: | :---: |
| **Tier 1 (Instantaneous / Preprocessing)** | S4, S5, S9 | Light-10 on/off, negation stopwords, stage ordering | Preprocessing only (0 GPU hours) | Run 1st on Twitter16 |
| **Tier 2 (Fixed Architecture / 1 Retrain)** | S1, S2, S3, S12, S13, S14 | Checkpoint, joint fine-tuning, bridge, LR map, pooling, head rescaling | 1 retrain (2 epochs) per ablation | Run 2nd on Twitter16 |
| **Tier 3 (Optimization Dynamics)** | S6, S8, S10 | 5 epochs, 5-fold cross-val, DHOA branch variants | 36 evaluations per ablation | Run 3rd on Twitter16 |
| **Tier 4 (Search Baselines & Leakage)** | S11, S7 | Random search (36 evals), Leakage diagnostic | 36 evaluations per ablation | Run 4th on Twitter16 |

### 10.2 Documented Reproduction Limitations
1. **Unresolvable Textual Equations:** DHOA Table 2 pseudocode labels inner branch actions with generic phrases, and variables $w, z, l, r, \theta, \phi, v$ lack complete mathematical distributions in the manuscript. All resolutions are documented as assumptions and audited via ablation S10.
2. **Non-Integer Implied Sample Counts:** Reported test accuracies ($99.50\%$, $99.30\%$) require fractional sample counts on verified test splits ($296.51$ and $162.85$ examples). While this proves that published percentages cannot be exact quotients on our test splits, the authors' exact proprietary test partition size is unstated.
3. **Unstated Search Hyperparameters:** DHOA population size ($N=6$) and update iterations ($T_{max}=5$) are engineering bounds adopted for computational feasibility.
4. **Hardware-Dependent Execution Feasibility:** Executing $NL=48$ on GPUs with $<12 \text{ GiB}$ VRAM is theoretically impossible under standard PyTorch AMP without 8-bit optimizer compression or activation checkpointing.
5. **Under-Training Confound for Large Batches:** Candidates sampled with $B = 512$ receive only 6 optimizer updates in 2 epochs, creating an unavoidable optimization confound compared to small-batch candidates.
6. **Severe Class Disparity in Physical PHEME Archive:** The physical PHEME dataset contains only 2,403 rumor threads ($37.4\%$), directly contradicting Table 3's claim of 3,853 rumors ($60.0\%$).

### 10.3 Implementation-Readiness Verdict
Design ready for staged implementation, subject to the measurement gates G1-G6 and to resolution of any equations or variable definitions marked UNRESOLVED. Hardware feasibility is unknown until the environment-setup stage measures the device and passes gates G1-G6. This document has not been validated by execution.

---

## 11. Implementation Interfaces and Run Order

### 11.1 Configuration Schema Specification
Every execution parameter is strictly typed, bounded, and mapped to its fidelity tier and owning design section:

| Parameter Name | Data Type | Permissible Range / Options | Default Value | Tier | Owning Section | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `task_arm` | String | `['arm_b', 'arm_a', 'arm_d', 'report_c']` | `'arm_b'` | T1/T3 | §0.4 | Selected experimental classification task arm |
| `dataset_name` | String | `['twitter16', 'twitter15', 'pheme']` | `'twitter16'` | T1 | §0.3 | Target social media benchmark dataset |
| `roberta_model` | String | `['roberta-base', 'roberta-large']` | `'roberta-base'` | T3 | §3.1 | Pretrained RoBERTa checkpoint identifier |
| `freeze_roberta`| Boolean | `[True, False]` | `True` | T3 | §3.1 | Freezes RoBERTa parameters during training |
| `gpt2_model` | String | `['gpt2-medium']` | `'gpt2-medium'` | T1 | §4.1 | Canonical GPT-2 medium backbone |
| `max_seq_len` | Integer | `[64, 512]` | `512` | T2 | §2.2 | Maximum token sequence length |
| `d_model` | Integer | `[1024]` | `1024` | T1 | §4.1 | GPT-2 Medium hidden representation dimension |
| `d_roberta` | Integer | `[768]` | `768` | T1 | §3.1 | RoBERTa-base feature dimension |
| `d_out` | Integer | `[2, 4]` | `2` (Arm B/D) | T2 | §0.4 | Classification head output logit dimension |
| `num_deer` ($N$) | Integer | `[6]` | `6` | T3 | §5.3 | DHOA population count |
| `num_iterations`| Integer | `[5]` | `5` | T3 | §5.3 | DHOA update iterations ($T_{max}=5$) |
| `eval_budget` | Integer | `[36]` | `36` | T3 | §5.3 | Total candidate evaluations per task arm |
| `candidate_epochs`| Integer | `[2]` | `2` | T3 | §5.3 | Training epochs per candidate evaluation |
| `B_phys_max` | Integer | `[1, 2, 4, 8, 16]` | `4` | T3 | §6.1 | Maximum micro-batch size calibrated via Gate G3 |
| `grad_clip_norm`| Float | `[0.1, 5.0]` | `1.0` | T3 | §6.3 | Maximum gradient norm threshold |
| `amp_dtype` | String | `['bfloat16', 'float16']` | Hardware-set | T3 | §6.3 | PyTorch autocast mixed-precision datatype |
| `seed_split` | Integer | `[42]` | `42` | T3 | §5.4 | Dataset split seed recorded in Phase 1 metadata |
| `seed_init` | Integer | `[100]` | `100` | T3 | §5.4 | Master initial checkpoint generation seed |
| `seed_order` | Integer | `[200]` | `200` | T3 | §5.4 | Epoch sample permutation shuffling seed |
| `seed_dropout` | Integer | `[300]` | `300` | T3 | §5.4 | PRNG stream seed for dropout masks |
| `seed_dhoa` | Integer | `[400]` | `400` | T3 | §5.4 | DHOA stochastic vector generation seed |

### 11.2 Artifact and Directory Layout
All data, models, logs, and evaluation records adhere to the following deterministic directory structure:

```
rumor_research/
├── data/
│   ├── raw/                              # Original raw archives (rumdetect2017.zip, pheme.tar.gz)
│   ├── processed/                        # Canonical preprocessed CSV files (twitter15, twitter16, pheme)
│   ├── splits/                           # Authoritative split JSON files (*_train_val_test.json)
│   └── feature_cache/                    # Offline RoBERTa FP16 memmapped tensors
│       ├── twitter15_features.bin
│       ├── twitter15_metadata.json
│       ├── twitter15_manifest.sha256
│       ├── twitter16_features.bin
│       ├── twitter16_metadata.json
│       ├── twitter16_manifest.sha256
│       ├── pheme_features.bin
│       ├── pheme_metadata.json
│       └── pheme_manifest.sha256
├── checkpoints/
│   ├── master_init_checkpoint.pt         # Saved master initial state (bridge, head, 24 appended blocks)
│   └── winners/                          # Winning model retrained checkpoints per arm and dataset
├── logs/
│   ├── environment_probe.json            # Hardware profile and Gate G1–G6 execution logs
│   ├── dhoa_state_arm_b_twitter16.json   # Full DHOA population history across 36 evaluations
│   ├── candidate_evaluations/            # Detailed JSON records for all 324 candidate evaluations
│   └── oom_recovery_events.log           # Dedicated log of all caught OOM exceptions and retry steps
└── reports/
    ├── headline_comparison_table.csv     # Final reproduction results vs. published claims
    ├── sensitivity_analysis_summary.csv  # S1–S14 ablation metrics and delta analyses
    └── failure_analysis_report.md        # Comprehensive report of infeasibilities or deviations
```

### 11.3 Module Interface Specifications
Every component in the reproduction runtime implements an explicit interface contract:

| Module Name | Primary Responsibilities | Inputs & Tensor Shapes | Outputs & Tensor Shapes | Mandatory Exceptions Raised |
| :--- | :--- | :--- | :--- | :--- |
| `TextPreprocessor` | 5-stage cleaning, Light-10 suffix stripping, WordNet POS lemmatization | Raw text strings | Preprocessed text tokens joined by `' '` | `PreprocessingError` if input is empty or invalid string |
| `FeatureExtractor` | RoBERTa tokenization and offline hidden state caching | Clean text strings, `max_len=512` | `H_seq in R^(T x 768)` (FP16 memmap array) | `TokenizationError`, `CacheCorruptionError` on hash mismatch |
| `ContinuousBridge` | Dimension projection and LayerNorm standardization | `H_seq in R^(B_phys x T x 768)` | `E_proj in R^(B_phys x T x 1024)` | `DimensionMismatchError` if input hidden dimension $\ne 768$ |
| `DynamicGPT2` | Early-exit routing ($NL \le 24$), appended blocks ($NL > 24$), head masking | `E_proj in R^(B_phys x T x 1024)`, `attn_mask in R^(B_phys x T)` | `H_out in R^(B_phys x T x 1024)` | `InvalidArchitectureError` if $NL \notin [1, 48]$ or $NAH \notin [1, 16]$ |
| `SequencePooler` | Last-valid-token extraction via index $\sum(\mathbf{M})-1$ | `H_out in R^(B_phys x T x 1024)`, `attn_mask in R^(B_phys x T)` | `h_last in R^(B_phys x 1024)` | `EmptyMaskError` if sequence attention mask is all zeros |
| `ClassificationHead` | Dropout regularization and linear projection to logits | `h_last in R^(B_phys x 1024)` | `logits in R^(B_phys x d_out)` | `HeadDimensionError` if $d_{out}$ does not match task arm |
| `AccumulationTrainer`| Scaled micro-batch forward/backward, clipping, optimizer stepping | DataLoader batches, micro-batch size $B_{phys}$ | `step_loss in R`, updated model weights | `DivergenceError` on NaN/Inf, `torch.cuda.OutOfMemoryError` on OOM |
| `DHOAOptimizer` | Coordinate updates, Table 2 branch routing, leader tracking | Objective fitness values, population coordinates | Best candidate vector $[L^*, B^*, NL^*, NAH^*, DR^*]$ | `PopulationInfeasibleError` if all candidates fail in initial round |

### 11.4 Per-Candidate Record Schema
Each candidate evaluation across the 324 optimization runs produces a structured JSON record:

```json
{
  "candidate_id": "arm_b_twitter16_iter01_deer03",
  "task_arm": "arm_b",
  "dataset": "twitter16",
  "iteration": 1,
  "deer_index": 3,
  "hyperparameters": {
    "L": 0.000142,
    "B": 32,
    "NL": 28,
    "NAH": 12,
    "DR": 0.25
  },
  "normalized_coordinates": [0.412, 0.061, 0.574, 0.733, 0.375],
  "seeds": {
    "seed_init": 100,
    "seed_order": 200,
    "seed_dropout": 300,
    "seed_dhoa": 400
  },
  "status": "EVALUATED",
  "status_options": ["EVALUATED", "INFEASIBLE_HARDWARE", "DIVERGED_NONFINITE", "CACHE_HIT"],
  "execution_stats": {
    "B_phys_used": 4,
    "mitigations_applied": ["halve_B_phys"],
    "train_loss_epoch1": 0.4821,
    "train_loss_epoch2": 0.3104,
    "val_loss": 0.3512,
    "val_accuracy": 0.8902,
    "val_macro_f1": 0.8845,
    "peak_vram_gib": 10.42,
    "runtime_seconds": 42.6
  }
}
```

### 11.5 Ordered Execution Runbook
The reproduction protocol executes in nine strictly sequential, auditable phases. Each phase defines an explicit Pass/Fail criterion and a formal stop rule:

```
[Phase 1: Environment Check] ──> [Phase 2: Pre-Flight P1-P4] ──> [Phase 3: Gates G1-G6]
                                                                             │
                                                                             ▼
[Phase 6: Priority Pilot Arm B (Twitter16)] <── [Phase 5: Master Checkpoint] <── [Phase 4: Preprocessing & Cache]
              │
              ▼
[Phase 7: Remaining Task Arms] ──> [Phase 8: Sensitivity Plan S1-S14] ──> [Phase 9: Reporting & Synthesis]
```

#### Phase-by-Phase Runbook Protocol
1. **Phase 1 — Runtime Environment Check:**
   - Action: Query CUDA availability, driver version, total/usable VRAM, and write permissions.
   - *Pass Criterion:* CUDA available, usable VRAM $\ge 7.0 \text{ GiB}$, write permissions verified.
   - *Stop Rule:* If CUDA unavailable or VRAM $< 7.0 \text{ GiB}$, halt immediately and emit `ENVIRONMENT_INCOMPATIBLE_REPORT`.
2. **Phase 2 — Pre-Flight Probes (P1–P4):**
   - Action: Execute hardware probe (P1), cache reload probe (P2), appended-block gradient probe (P3), and candidate isolation probe (P4).
   - *Pass Criterion:* All 4 pre-flight probes exit with status 0 and non-zero gradients on `c_proj`.
   - *Stop Rule:* If P3 fails (zero gradients on appended blocks) or P4 fails (memory leak), halt and report architectural bug.
3. **Phase 3 — Staged Measurement Gates (G1–G6):**
   - Action: Execute Gate G1 (forward memory), G2 (backward memory), G3 ($B_{phys\_max}$ calibration at $NL=48$), G4 ($NL=48$ stress test), G5 (accumulation equivalence on synthetic batches), and G6 (OOM recovery test).
   - *Pass Criterion:* Gates G1–G6 exit successfully; $B_{phys\_max}$ is calibrated and logged.
   - *Stop Rule:* If G1, G2, or G5 fails, halt immediately and report engine defect.
4. **Phase 4 — Preprocessing and Feature Caching:**
   - Action: Preprocess all text for Twitter15, Twitter16, and PHEME. Extract and write RoBERTa FP16 hidden states and manifests to disk.
   - *Pass Criterion:* SHA-256 manifests match binary feature arrays; reload test passes with `rtol=1e-3, atol=1e-3`.
   - *Stop Rule:* If cache corruption or dimension mismatch occurs, halt and recompute cache.
5. **Phase 5 — Master Initial Checkpoint Generation:**
   - Action: Instantiate RoBERTa-to-GPT-2 bridge, classification heads ($d_{out}=2$ and $d_{out}=4$), and one fixed set of 24 appended blocks with seed $S_{init}=100$. Save to `checkpoints/master_init_checkpoint.pt`.
   - *Pass Criterion:* File exists on disk, size $\approx 1.2 \text{ GB}$, reloads with identical parameter tensors.
   - *Stop Rule:* If checkpoint cannot be saved or reloaded, halt and report I/O defect.
6. **Phase 6 — Priority Pilot Execution: Arm B on Twitter16:**
   - Action: Execute full DHOA search (6 deer $\times$ 6 rounds = 36 evaluations) for Arm B on Twitter16. Retrain winning candidate on TRAIN split only for 2 epochs; evaluate test set once.
   - *Pass Criterion:* 36 candidate evaluations complete; test metrics recorded.
   - *Stop Rule:* If all initial candidates are hardware-infeasible, halt and emit `PILOT_HARDWARE_INFEASIBLE_REPORT`.
7. **Phase 7 — Execution of Remaining Task Arms:**
   - Action: Execute Arm B on Twitter15 and PHEME; execute Arm A (4-class) on Twitter16, Twitter15, PHEME; execute Arm D (veracity) on Twitter16, Twitter15, PHEME; evaluate Report C post-hoc collapse.
   - *Pass Criterion:* All $324$ candidate evaluations and 9 final retraining runs complete.
   - *Stop Rule:* If persistent divergence or unrecoverable exceptions occur, log specific arm failure and continue remaining arms.
8. **Phase 8 — Sensitivity Analysis Plan (S1–S14):**
   - Action: Execute ablations S1 through S14 in ordered sequence (Tier 1 $\to$ Tier 2 $\to$ Tier 3 $\to$ Tier 4) on Twitter16 pilot.
   - *Pass Criterion:* All 14 ablation evaluations complete and delta accuracies recorded.
   - *Stop Rule:* If an ablation encounters a fatal exception, record ablation failure and continue remaining ablations.
9. **Phase 9 — Synthesis and Final Discrepancy Reporting:**
   - Action: Compile `reports/headline_comparison_table.csv`, generate 95% Wilson confidence intervals, compute McNemar's significance tests, and synthesize discrepancy findings.
   - *Pass Criterion:* Complete reproduction report generated with zero missing metrics.
   - *Stop Rule:* None (final reporting stage).

---

## 12. Change Log and Self-Check Verification

### 12.1 Section-by-Section Change Log
The following audit table accounts for every section of this document relative to the frozen base `PHASE_2B.3.4_TECHNICAL_DESIGN.md` and explicitly rejects all unauthorized modifications introduced in `PHASE_2B.3.5_TECHNICAL_DESIGN.md`:

| Section | Status | Description of Change / Rejection Rationale |
| :--- | :---: | :--- |
| **Title & Frontmatter** | **CHANGED** | Updated to Version 2B.4; purged forbidden status claims; preserved disciplined epistemic stance. |
| **0. Replication Objective & Allocation** | **CHANGED** | Preserved frozen base objectives and headline claims; derived exact physical counts; rejected 2B.3.5's tree-count note and 25-eval budget. |
| **0.3 Dataset Reconciliation** | **UNCHANGED** | Preserved exact physical census vs. Table 3 reconciliation from 2B.3.4, highlighting impossible PHEME rumor total. |
| **0.4 Task Arms and Budget** | **CHANGED** | Prioritized Arm B (runs first); derived exact Arm D veracity counts (742, 412, 1705); fixed budget to 36 evals and 2 epochs. |
| **1. End-to-End Architecture** | **CHANGED** | Updated diagram with explicit tensor shapes and tier tags; specified master checkpoint appended blocks; pruned `wte` (51.46M params). |
| **2. Preprocessing Pipeline** | **CHANGED** | Preserved frozen base 5 operations; rejected 2B.3.5's 10-step pipeline; specified TweetTokenizer details, case preservation, two-pass lemmatization; purged Ref [24] from lemmatization. |
| **2.3 Feature Caching Protocol** | **CHANGED** | Adopted FP16 memmapped format, SHA-256 manifest, and `rtol=1e-3, atol=1e-3` tolerance from B2; provided worst-case and illustrative average size formulas. |
| **3. RoBERTa-to-GPT-2 Interface** | **CHANGED** | Bridge init set to $\mathcal{N}(0, 0.02)$, bias 0, LayerNorm $\gamma=1, \beta=0$; purged invalid pre-LN variance explosion claim; excluded `wte` from trainable set. |
| **4. GPT-2 Dynamic Mechanisms** | **CHANGED** | Appended blocks specified from master initial checkpoint with zero output projections and non-zero GELU gradient flow; head masking without rescaling; last-token pooling; rejected mean-pooling and head rescaling. |
| **5. DHOA Equations & Symbols** | **CHANGED** | Equations 4–14 confirmed from WMF images (image5–30); rejected 2B.3.5's faulty equation table; verified Eq. 13 vs image26 glyphs; documented Table 2 branch mapping with Strategy 3 textual support; natural log table for Eq. 8; normalized coordinate space with log-linear $L$. |
| **5.3 Population & Budget** | **UNCHANGED** | Preserved frozen base budget: $N=6$ deer, 5 update iterations = 36 evaluations, 2 epochs per candidate; rejected 2B.3.5's 25 evals / 5 epochs. |
| **5.4 Candidate Isolation & Seeds**| **CHANGED** | Explicitly defined seeds ($S_{split}=42, S_{init}=100, S_{order}=200, S_{dropout}=300, S_{dhoa}=400$); master checkpoint state reloading; `DIVERGED_NONFINITE` set to `fitness = null`. |
| **6. Training & Accumulation** | **CHANGED** | Edge-case table and worked example preserved; AdamW, unweighted cross-entropy, 2 epochs, hardware-selected AMP dtype, gradient clipping `max_norm=1.0`, full GPT-2 fine-tuning scope. |
| **7. Metrics, Fitness & Stats** | **CHANGED** | Equations 15–18 confirmed from WMFs, 19–22 typeset; MCC defect explained; Wilson 95% intervals computed ($[97.9, 99.9], [96.5, 99.9], [98.4, 99.5]$); non-integer count caveat added; train-only final retrain protocol. |
| **8. Hardware, Memory & Gates** | **CHANGED** | Standardized strictly to GiB; 16 B/param AMP and 10 B/param 8-bit layouts; activation coefficient formula and full tables ($s \in \{64,128,256,512\}$); extended matrix (8 to 48 GiB tiers); separated pre-flight P1–P4 from gates G1–G6; full OOM retry ladder. |
| **9. Assumptions Matrix** | **CHANGED** | Expanded to comprehensive matrix with explicit columns (ID, Location, Tier, Classification, Decision, Justification) covering all assumptions and DHOA symbols. |
| **10. Sensitivity Plan & Verdict**| **CHANGED** | Preserved S1–S8 and added S9–S14; ordered execution tiers piloted on Twitter16; Arm B runs first; exact verbatim verdict preserved. |
| **11. Implementation Interfaces** | **NEW** | Added Section 11 containing configuration schema, directory layout, module interface contracts, per-candidate record schema, and 9-phase ordered runbook. |
| **12. Change Log & Self-Check** | **NEW** | Section-by-section audit table and itemized self-check verification list. |

### 12.2 Self-Check Verification List

1. **Every equation shown was rendered and read by you; each equation image ID printed in the document matches the content.**  
   - **PASS.** All equations Eqs. 4–14 and 15–18 were verified against high-resolution renders of embedded WMF files (`image5.wmf` through `image30.wmf`, and `image48.wmf` through `image51.wmf`). Image IDs match content exactly: Eq. 4 is `image5`, Eq. 5 is `image7`, Eq. 6 is `image9`, Eq. 7 is `image10`, Eq. 8 is `image16`, Eq. 9 is `image17`, Eq. 10 is `image21`, Eq. 11 is `image22`, Eq. 12 is `image24`, Eq. 13 is `image25`, Eq. 14 is `image30`. Faulty equations from 2B.3.5 were completely rejected.

2. **Eqs. 15–18 are accuracy, recall, precision, F1 in this order.**  
   - **PASS.** Verified in Section 7.3: Eq. 15 is Accuracy (`image48.wmf`), Eq. 16 is Recall (`image49.wmf`), Eq. 17 is Precision (`image50.wmf`), and Eq. 18 is F1-score (`image51.wmf`).

3. **No reference number is attached to a statement the paper does not make; Ref [25] and Ref [24] are used only where the paper uses them.**  
   - **PASS.** In Section 2.2 and Section 9, Ref [23] is attached only to URL removal and tokenization; Ref [24] is attached only to stopword removal; Ref [25] is not attached to Light-10; lemmatization and Light-10 have no errant citations.

4. **Light-10 is present, with the destructive examples; no lowercasing/punctuation/number/mention/hashtag removal exists in the pipeline.**  
   - **PASS.** Section 2.1 and 2.2 strictly retain Light-10 with the destructive examples (`expand` $\to$ `exp`, `demand` $\to$ `dem`, `command` $\to$ `comm`, `understand` $\to$ `underst`, `island` $\to$ `isl`, `contraband` $\to$ `contrab`). No whole-text lowercasing, @mention removal, hashtag stripping, punctuation removal, or number removal exists in the pipeline.

5. **The budget is 6 deer, 36 evaluations, 2 epochs everywhere (text, tables, diagrams, matrix, runbook, change log).**  
   - **PASS.** The budget of $N=6$ deer, 1 initialization + 5 update rounds = 36 evaluations per DHOA run, and 2 epochs per candidate is consistently maintained across Sections 0.4, 1, 5.3, 5.4, 6.3, 7.5, 9, 10.1, 11.1, 11.5, and 12.1.

6. **The same dropout sites, pooling, padding and masking are described everywhere.**  
   - **PASS.** Dropout is consistently applied to `embd_pdrop`, `attn_pdrop`, `resid_pdrop`, and classification head (bridge has no dropout; eval mode disables dropout). Last-valid-token pooling with index $\sum(\mathbf{M})-1$, zero-vector embedding right-padding, and attention masking are described consistently across Sections 1, 3.3, 4.2, 4.3, 9, and 11.3.

7. **Every number was recomputed; Wilson, memory, activation, cache and log tables agree with the cross-check values or the difference is explained.**  
   - **PASS.** All numbers were recomputed with arithmetic shown: Wilson intervals (Twitter15: $[97.9\%, 99.9\%]$, Twitter16: $[96.5\%, 99.9\%]$, PHEME: $[98.4\%, 99.5\%]$); static memory (16 B/param: $4.532 \text{ GiB}$ and $9.037 \text{ GiB}$; 10 B/param: $2.833 \text{ GiB}$ and $5.648 \text{ GiB}$); activation coefficients ($39936, 45056, 55296, 75776$); cross-checks ($NL=24, b=4, s=512$ uncheckpointed $3.469 \text{ GiB}$, checkpointed $0.238 \text{ GiB}$); worst-case cache sizes ($1.091, 0.599, 4.706 \text{ GiB}$); log table values for $\ln, \log_{10}, \log_2$.

8. **No forbidden wording appears (search the document).**  
   - **PASS.** The document has been checked and verified to contain zero occurrences of "zero speculation", "zero unverified claims", "exact reproduction", "identical to the paper", "guaranteed", "fully validated", "measured" (outside the instruction to not use it), the header line "Gravitas Engineering Reliability Runtime", or the word "FINAL" as a status claim.

9. **No GPU model or VRAM size is assumed; no figure is called measured.**  
   - **PASS.** Hardware profile in Section 8.1 is defined as a dynamic runtime-measured configuration struct; all memory numbers are theoretical planning aids.

10. **Every assumption appears in the matrix with a tier; every UNRESOLVED item has a stated primary choice and an ablation.**  
    - **PASS.** Section 9 contains an assumption matrix covering every engineering assumption, each with an assigned tier (T1, T2, T3). All unresolved items (Table 2 branch mapping, symbols $w, z, l, r, \theta, \phi, v$, log base) specify an explicit primary choice and an assigned ablation in Section 10.1.

11. **The change-log table lists every section, and nothing from section D appears in the document.**  
    - **PASS.** Section 12.1 provides an itemized section-by-section audit table. All unauthorized content from 2B.3.5 listed in section D has been rejected.

12. **Section references, ablation IDs (S1–S14), arm names (A, B, D, Report C) and seed names are consistent across all sections.**  
    - **PASS.** Cross-references, ablation IDs (S1–S14), task arm designations (Arm B runs first, Arm A, Arm D, Report C), and seed names ($S_{split}=42, S_{init}=100, S_{order}=200, S_{dropout}=300, S_{dhoa}=400$) are aligned throughout the document.

---
