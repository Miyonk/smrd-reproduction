# SMRD — rumour detection on Twitter15, Twitter16 and PHEME

A rumour-detection pipeline and a leakage-controlled evaluation of it on three public
benchmarks: 4-way classification (non-rumour / false / true / unverified) on Twitter15
and Twitter16, and rumour vs non-rumour on PHEME.

Four model families are compared on identical folds, with paired significance tests:

- **lexical** — TF-IDF of the source tweet and of the thread's words, logistic regression;
- **text** — fine-tuned RoBERTa on the source tweet (6 epochs; `text_long` up to 30);
- **graph** — BiGCN on the propagation tree;
- **fusion** and **stacks** — RoBERTa + BiGCN trained jointly, and weighted averages of arms.

Every fold uses 60% of the data for training, 20% for validation (model selection
only, never trained on) and 20% for testing (predicted once).

## Summary of results

Pooled 5-fold accuracy. *Grouped* folds keep near-duplicate tweets in one fold;
*story-held-out* folds keep whole stories (PHEME: whole events) out of training.

| Folds | Arm | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|---|
| | majority class | 0.253 | 0.251 | 0.626 |
| grouped | text (RoBERTa, 6 epochs) | 0.688 | 0.736 | 0.867 |
| grouped | text_long (RoBERTa, ≤ 30 epochs) | 0.784 | — | — |
| grouped | graph (BiGCN) | 0.804 | 0.797 | 0.858 |
| grouped | fusion | 0.786 | 0.781 | 0.871 |
| grouped | **lexical** | **0.848** | **0.840** | **0.871** |
| story-held-out | lexical | 0.683 | 0.686 | 0.701 |

What the experiments show:

1. The lexical arm beats BiGCN on all three datasets (+4.4 / +4.3 / +1.3 points,
   McNemar p ≤ 0.0003) and matches RoBERTa and fusion on PHEME.
2. Nothing added on top of it gives a measurable gain: stacking it with BiGCN or with
   the longer-trained RoBERTa, or tuning BiGCN with a Grey Wolf Optimizer search.
3. Keeping near-duplicates out of training lowers RoBERTa on Twitter16 from 0.799
   (random split) to 0.688.
4. On stories absent from training the lexical arm falls to about 0.68–0.70, and on
   Twitter15/16 the figure keeps falling as stories are defined more coarsely.

The proposed pipeline combines three branches (lexical, `text_long`, BiGCN) by stacking
their class probabilities (`stack_full`); `notebooks/colab_full_pipeline.ipynb` trains
and combines them from scratch. That three-branch combination has not been run yet.

The sections below follow the order the work was done in; each names its branch.
Paper-ready tables, the method description and related work are in
[`paper/results_and_methods.md`](paper/results_and_methods.md).

The repository started as an implementation of a published GPT-2-based model; that
code is still here and is described under [Legacy pipeline](#legacy-pipeline-gpt-2--dhoa).
Its status table:

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Data loading & splits | ✅ Complete | Stratified 70/10/20 splits with seed=42; dataset census reconciled. |
| Phase 2B — Implementation design | ✅ Complete | [`PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md`](PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md) (frozen specification). |
| Phase 3 — Codebase implementation | ✅ Complete | Full modular stack: preprocessing, features, model, DHOA, train, gates, eval, CLI. |
| Phase 4 — Hardware gates & execution | 🔄 Ready | Requires $\ge 7$ GiB VRAM GPU (Colab T4 / GPU workstation) for 324 candidate fits + 9 retrains. |

---

## Redesign — Stage 1: source-text encoder baseline (branch `stage1-encoder`)

The redesign replaces the GPT-2 backbone with a **fine-tuned RoBERTa-base** source
encoder (stage 1), later fused with a BiGCN propagation branch (stage 2) and tuned
with GWO (stage 3). The GPT-2/DHOA pipeline above is left untouched and is kept as
an ablation arm. Stage 1 lives entirely in new files:

| File | Role |
|---|---|
| `src/encoder_preprocess.py` | `normalize_tweet`: HTML unescape, URL→`URL`, @mention→`@USER`. No stemming, stopword removal or lemmatisation. |
| `src/dedup.py` | Near-duplicate clusters (char 3–5-gram TF-IDF cosine ≥ 0.9) → `data/splits/<dataset>_dup_clusters.json`; dedup-clean test subsets; cluster ids reusable as `groups=` for later event-separated splits. |
| `src/encoder_model.py` | `SourceTextClassifier` with separate `encode()` / `classify()`; versioned artifact save/load; `predict_texts` returns label + confidence. |
| `src/finetune.py` | AdamW + linear warmup/decay, label smoothing, class weights, AMP, early stopping on **validation macro-F1**, resume-safe per-epoch checkpoints, **single** test evaluation with best-val weights. |
| `src/run_encoder.py` | CLI: `dedup`, `baseline` (majority class), `train` (multi-seed, mean ± std summary). |
| `notebooks/colab_stage1.ipynb` | Colab runner with Drive checkpoints. |

Tasks: 4-way on Twitter15/16, rumour vs non-rumour on PHEME. Splits: the existing
seed-42 70/10/20 files, unchanged. Every result is reported on the full test split
and on the dedup-clean test subset, with Wilson 95% intervals and a majority-class
baseline.

```bash
python -m src.run_encoder dedup --all-datasets
python -m src.run_encoder baseline --dataset twitter16
python -m src.run_encoder train --dataset twitter16 --max-epochs 1 --tag smoke   # smoke test
python -m src.run_encoder train --dataset twitter16 --seeds 42 43 44 \
    --ckpt-root /content/drive/MyDrive/smrd/checkpoints/encoder                  # Colab
```

Outputs: `<ckpt-root>/<run>/seed<k>/{result.json, artifact/}` and
`results/stage1/*.json`. Re-running a command resumes unfinished seeds from their
last epoch and skips finished ones.

Stage 1 results (Colab L4, seeds 42/43/44, full test / dedup-clean test accuracy):
Twitter16 0.799 ± 0.006 / 0.771, Twitter15 0.732 ± 0.017 / 0.707,
PHEME 0.868 ± 0.005 / 0.861.

## Redesign — Stage 2: propagation trees (branch `stage2-propagation`)

| File | Role |
|---|---|
| `src/propagation.py` | Tree parsing, BFS-ordered `TreeGraph`, cache I/O, batching. |
| `src/propagation_model.py` | BiGCN (top-down + bottom-up GCN, root feature enhancement, DropEdge) in plain PyTorch; `fusion` = `[RoBERTa ; BiGCN] → linear`. |
| `src/train_propagation.py` | Training with the same rules as stage 1 (val macro-F1 selection, single test evaluation, resume-safe). |
| `src/run_propagation.py` | CLI: `build-graphs`, `stats` (tree stats + structure-only shortcut check), `train --arm graph|fusion`. |
| `data/graphs/*_graphs.json.gz` | Committed tree caches (~4 MB), so Colab needs no archives. |
| `notebooks/colab_stage2.ipynb` | Colab runner. |

Node features: Twitter15/16 use the public RvNN bag-of-words release
([Rumor_RvNN `resource/`](https://github.com/majingCUHK/Rumor_RvNN)), the input
published BiGCN results use (its vocabulary covers all threads, test included).
PHEME uses reply threads from the PHEME archive with a 5,000-word vocabulary built
from training threads only; reply nodes with no tweet JSON are contracted.

```bash
# only needed to rebuild the caches (archives in _archive/, RvNN files in _archive/rvnn/)
python -m src.run_propagation build-graphs --all-datasets \
    --rvnn-dir _archive/rvnn --pheme-tar _archive/PHEME_veracity.tar.bz2
python -m src.run_propagation stats --all-datasets
python -m src.run_propagation train --dataset twitter16 --arm graph  --seeds 42 43 44
python -m src.run_propagation train --dataset twitter16 --arm fusion --seeds 42 43 44
```

Stage 2 results on the seed-42 split (full test accuracy, 3 seeds): BiGCN
0.868 / 0.817 / 0.863 and fusion 0.829 / 0.758 / 0.877 on Twitter16 / Twitter15 /
PHEME; structure-only logistic regression 0.409 / 0.362 / 0.522.

## Evaluation protocol: grouped 5-fold cross-validation (branch `cv-eval`)

The seed-42 test sets hold 164 / 298 / 1,285 tweets, so accuracy carries a sampling
error of about ±3 / ±2.3 / ±1.1 points — as large as the gaps between arms, and not
captured by the seed-to-seed std. The validation sets (82 / 149 tweets) are too small
to tune against. Arm comparisons and any later tuning use cross-validation instead:

- `data/splits/<dataset>_cv5_grouped.json`: 5 folds, stratified by label, with
  near-duplicate clusters as groups (no test tweet has a near-copy in train/val).
- Fold *k* tests, fold *k+1* validates (model selection only), the rest trains.
- One run per fold; arms are compared on the pooled predictions (Wilson interval,
  paired accuracy difference, exact McNemar test).
- PHEME node vocabulary is re-selected per fold from training threads
  (`data/graphs/pheme_graphs_fullvocab.json.gz` + `restrict_vocab`).

```bash
python -m src.run_cv make-folds --all-datasets     # only to rebuild the fold files
python -m src.run_cv train --dataset twitter16 --arm text|graph|fusion --ckpt-root <dir>
python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>
```

Colab runner: `notebooks/colab_cv.ipynb`.

Results (Colab L4, pooled accuracy over all tweets, Wilson 95% interval):

| | Twitter16 (n=818) | Twitter15 (n=1,490) | PHEME binary (n=6,425) |
|---|---|---|---|
| majority class | 0.253 | 0.251 | 0.626 |
| text (RoBERTa) | 0.688 [0.656, 0.719] | 0.736 [0.713, 0.757] | 0.867 [0.858, 0.875] |
| graph (BiGCN) | 0.804 [0.776, 0.830] | 0.797 [0.776, 0.817] | 0.858 [0.849, 0.866] |
| fusion | 0.786 [0.757, 0.813] | 0.781 [0.759, 0.801] | 0.871 [0.863, 0.879] |

Paired differences (points, 95% interval, exact McNemar p):

| | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| graph − text | +11.6 [+8.3, +15.0], p<0.0001 | +6.2 [+3.8, +8.6], p<0.0001 | −0.9 [−1.8, +0.0], p=0.058 |
| fusion − graph | −1.8 [−4.6, +0.9], p=0.22 | −1.7 [−3.8, +0.5], p=0.14 | +1.3 [+0.5, +2.2], p=0.002 |
| fusion − text | +9.8 [+7.0, +12.5], p<0.0001 | +4.5 [+2.6, +6.4], p<0.0001 | +0.5 [−0.2, +1.1], p=0.16 |

Propagation trees help clearly on Twitter15/16 and not on PHEME. Fusion is not
distinguishable from BiGCN on Twitter (the 4–6 point gap on the seed-42 split was
mostly noise) and is slightly better than BiGCN on PHEME. Twitter16 text accuracy
is 11 points below its seed-42 value (0.799): that split had near-copies of 16.5%
of its test tweets in training, and these folds have none. The best arm so far is
BiGCN on Twitter15/16 (about 0.80) and fusion on PHEME (0.87).

## Redesign — Stage 3: GWO hyperparameter search (branch `stage3-gwo`)

| File | Role |
|---|---|
| `src/gwo.py` | Grey Wolf Optimizer over the unit cube (maximising, deterministic per seed) and the search spaces. |
| `src/run_tune.py` | Nested search: for each test fold, candidates are scored on the other four folds only; the best one is trained and tested once on the usual fold view. |
| `notebooks/colab_stage3.ipynb` | Colab runner. |

Budget per test fold: 6 wolves × (1 + 5 rounds) = 36 candidates, wolf 0 starting
at the stage 2 settings.
Fitness = mean validation macro-F1 over two training runs with different inner
validation folds. Searched for BiGCN: learning rate, weight decay, hidden and output
width, dropout, DropEdge rate, label smoothing, batch size. The tuned arm is written
as `<arm>_gwo` next to the untuned arm, same rows and seed per fold, and
`run_cv compare` adds the paired `graph_gwo − graph` line.

```bash
python -m src.run_tune search --dataset twitter16 --arm graph --ckpt-root <dir>
python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>
```

Result on Twitter16 (Colab, 5 folds): tuned BiGCN 0.806 [0.777, 0.831] against
0.804 untuned; paired difference +0.1 points [−1.3, +1.6], McNemar p = 1.0 (19 tweets
only the tuned model gets right, 18 only the untuned one). The stage 2 settings are
already as good as the search can find, so the search was not run on the other datasets.

## Lexical arm and stacked arm (branch `lexical-stack`)

| File | Role |
|---|---|
| `src/lexical.py` | `lexical` arm: TF-IDF of the source tweet (word 1–2-grams, character 2–5-grams) and of all words in the thread, logistic regression. CPU only. |
| `src/stack.py` | `stack` arm: weighted average of member arms' class probabilities. |
| `notebooks/colab_lexical.ipynb` | Colab runner (no GPU needed). |

Same folds and rules as the other arms. The validation fold is never trained on: it
selects the lexical model's `C` and the stack's weights (grid on the simplex, macro-F1),
and each test fold is predicted once. Neural members are re-scored from their saved
models to obtain validation and test probabilities.

```bash
python -m src.run_cv train --dataset twitter16 --arm lexical --ckpt-root <dir>
python -m src.run_cv stack --dataset twitter16 --members lexical graph --ckpt-root <dir>
python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>
```

Result (Colab, pooled accuracy on the 5 folds; the stack is lexical + BiGCN):

| | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| lexical | 0.848 | 0.840 | 0.871 |
| stack | 0.840 | 0.844 | 0.872 |
| lexical − BiGCN | +4.4 pts (p = 0.0001) | +4.3 pts (p < 0.0001) | +1.3 pts (p = 0.0003) |
| stack − lexical | −0.9 pts (p = 0.12) | +0.3 pts (p = 0.13) | +0.1 pts (p = 0.53) |

The lexical arm beats BiGCN on all three datasets. Adding BiGCN to it changes nothing
measurable: the stack differs from the lexical arm on 15, 7 and 91 test tweets
respectively, and none of the differences is significant (McNemar).

## Longer-trained text arm (branch `text-long`)

The stage 1 text arm trains RoBERTa for 6 epochs, which is under 200 optimizer steps on
a Twitter16 fold, and it scores well below the lexical arm on the same tweets
(0.688 vs 0.848). `text_long` is the same model and settings with up to 30 epochs
(patience 10); the best epoch is still chosen on the validation fold. `stack_text` is
the stack of `lexical` and `text_long`.

```bash
python -m src.run_cv train --dataset twitter16 --arm lexical --ckpt-root <dir>
python -m src.run_cv train --dataset twitter16 --arm text_long --light --ckpt-root <dir>
python -m src.run_cv stack --dataset twitter16 --members lexical text_long --tag text --ckpt-root <dir>
python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>
```

`--light` keeps only `result.json` and `probs.json` per fold (validation and test class
probabilities of the best epoch): no saved model and no per-epoch checkpoint, so an
interrupted fold starts again. `notebooks/colab_text_long.ipynb` runs all of this from
an empty Drive (T4 GPU).

Result on Twitter16 (Colab, pooled accuracy on the 5 folds): `text_long` 0.784
[0.754, 0.810], up from 0.688 [0.656, 0.719] for the 6-epoch model (separate runs, so
no paired test), but still 6.5 points below the lexical arm (0.848, McNemar
p < 0.0001). `stack_text` 0.856; `stack_text − lexical` +0.7 points [−0.7, +2.2],
p = 0.42 (22 tweets only the stack gets right, 16 only the lexical arm). Longer training
fixes most of the text arm's deficit, but the stack is not measurably better than the
lexical arm alone. Not run on Twitter15 or PHEME.

## Story-held-out evaluation (branch `story-eval`)

The grouped folds keep near-copies of a tweet together, but other tweets about the same
story still fall on both sides of a split. The `story` split uses the same 5-fold
protocol (60% train / 20% validation / 20% test, validation for model selection only)
with coarser groups, so each test fold holds only stories absent from training and
validation:

- **PHEME:** the group is the event. The 9 events are packed into 5 folds by size
  (Charlie Hebdo; Sydney siege; Ferguson; Ottawa + Gurlitt; the four smallest).
- **Twitter15/16:** there are no event ids, so source tweets are clustered by text
  (average-linkage on word TF-IDF, cosine ≥ 0.3) and merged with the near-duplicate
  clusters: 498 stories on Twitter16, 828 on Twitter15.

Fold files: `data/splits/<dataset>_cv5_story.json`. Every `run_cv` command takes
`--split story` before the command name; those runs are stored under
`<ckpt-root>/story/`.

```bash
python -m src.run_cv --split story train --dataset twitter16 --arm lexical --ckpt-root <dir>
python -m src.run_cv --split story compare --dataset twitter16 --ckpt-root <dir>
python scripts/story_threshold_sensitivity.py
```

Lexical arm, pooled accuracy (local CPU run, scikit-learn 1.5):

| | Twitter16 | Twitter15 | PHEME |
|---|---|---|---|
| grouped folds | 0.848 | 0.840 | 0.871 |
| story-held-out folds | 0.683 | 0.686 | 0.701 |
| majority class | 0.253 | 0.251 | 0.626 |

On PHEME the story-held-out model is close to the majority class within each event
(Charlie Hebdo 0.811 vs 0.780, Ferguson 0.760 vs 0.752), and on the fold of the four
smallest events it is below it (0.423 vs 0.637); macro-F1 is 0.635.

On Twitter15/16 the number depends on how coarsely stories are defined, and it falls
steadily with no plateau, so 0.68 is one point on a curve and not a fixed property of
the model:

| story threshold (cosine ≥) | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|
| Twitter16 stories | 633 | 570 | 498 | 393 | 234 |
| Twitter16 lexical accuracy | 0.793 | 0.758 | 0.683 | 0.661 | 0.506 |
| Twitter15 stories | 1118 | 978 | 828 | 648 | 373 |
| Twitter15 lexical accuracy | 0.785 | 0.743 | 0.686 | 0.595 | 0.548 |

Only the lexical arm has been run on these folds. `notebooks/colab_story.ipynb` runs
BiGCN and the longer-trained RoBERTa on them (T4 GPU, empty Drive is fine).

---

# Legacy pipeline (GPT-2 + DHOA)

Everything from here down describes the original GPT-2 / DHOA code the repository
started with (`src/model.py`, `src/dhoa.py`, `src/train.py`, `src/run.py`). It is kept
unchanged and is not used by the arms above.

## Design Document

[`PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md`](PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md) is the frozen, standalone specification for the full implementation. All code implements exactly what is described there, with all gaps cataloged in [`documentation/assumptions.md`](documentation/assumptions.md).

---

## Datasets

- **Twitter15** — 1,490 released threads (4-class: non-rumour / false / true / unverified)
- **Twitter16** — 818 released threads
- **PHEME** — 6,425 released threads (contains only 2,403 rumor threads; Table 3 claim of 3,853 is physically impossible)

Processed CSVs and train/val/test split files (seed=42) are included in `data/`.

---

## Architecture Overview

| Component | Specification |
|---|---|
| Feature extractor | RoBERTa-base (frozen, eval mode) |
| Bridge | `Linear(768→1024)` + `LayerNorm(1024)` |
| Backbone | GPT-2 Medium, $NL \in [1,48]$ layers (zero-init appended blocks for $NL > 24$) |
| Head masking | $NAH \in [1,16]$ attention heads |
| Optimizer | DHOA (Deer Hunting Optimization Algorithm, $N=6$, 5 update rounds = 36 evals/arm) |
| Training | AdamW, constant LR, 2 epochs/candidate, logical batch accumulation ($B \in [1, 512]$) |
| Task arms | Arm B (binary rumour detection, runs first), Arm A (4-class), Arm D (veracity), Report C (post-hoc collapse of Arm A) |

---

## Reproducibility

- Every paper-specified value is used unchanged.
- Every unspecified choice is labelled `T3 (IMPLEMENTATION ASSUMPTION)` in the design doc and in code comments.
- A failure to reproduce is a valid, reportable scientific outcome.

---

## Repository Structure

```
data/
  processed/       # Cleaned CSVs (twitter15, twitter16, pheme)
  raw/             # Original CSVs
  splits/          # Stratified 70/10/20 train/val/test JSON split files
src/
  __init__.py
  config.py        # Hyperparameter bounds, seeds, search spaces, CLI options
  data.py          # Task arm adapters (Arm A, B, D)
  data_loader.py   # Dataset loading and split generation
  dhoa.py          # Deer Hunting Optimization Algorithm (Eqs. 4-14)
  evaluate.py      # Metrics (Eqs. 15-22), Wilson CI, McNemar, bootstrap
  features.py      # RoBERTa feature caching to FP16 memmap arrays
  gates.py         # Hardware validation gates G1-G6 and preflight checks
  model.py         # RoBERTa-to-GPT-2 hybrid model, dynamic depth, head masking
  preprocess.py    # 7-step pipeline: URL -> slice -> tokenize -> stopwords -> Light-10 -> lemmatize
  run.py           # Unified runbook CLI (probe, cache, master, gates, search, retrain, pilot, all, report)
  train.py         # Logical batch accumulation training, AMP, AdamW, loss scaling
tests/
  test_dhoa.py                     # DHOA math, boundary, diversity tests
  test_metrics.py                  # Evaluation metrics & Wilson CI tests
  test_model.py                    # Bridge, dynamic depth, head masking tests
  test_preprocess_and_data.py      # Preprocessor pipeline and dataset tests
  test_train_and_accumulation.py   # Logical accumulation and loss scaling tests
  verify_phase1.py                 # Dataset census and split integrity tests
scripts/
  bench_compare.py                 # Candidate comparison benchmarks
  bench_step.py                    # Forward/backward step timing
  bench_sweep.py                   # Layer/batch sweep benchmarks
  bench_trainstep.py               # Accumulation benchmark
  dryrun_candidate.py              # Single candidate dry run
  setup_windows.ps1                # Windows venv setup script
notebooks/
  colab_runner.ipynb               # Google Colab notebook for GPU execution
documentation/
  assumptions.md                   # T3 implementation assumptions catalog
PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md
```

---

## How to Run

### System Requirements
- **Python:** 3.12+
- **GPU:** NVIDIA GPU with $\ge 7.0$ GiB usable VRAM (e.g. Google Colab T4/L4/A100 or local RTX 3080/4080/5090). A local 6 GiB card (e.g. RTX 3050 Laptop) can run preprocessing, feature caching, and smoke tests, but full DHOA search requires a T4-class GPU.

---

### Method 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/) and upload `notebooks/colab_runner.ipynb` (or open it directly from your GitHub repository).
2. Set runtime type to GPU:
   - Go to **Runtime** → **Change runtime type** → select **T4 GPU** (or better) → **Save**.
3. In Colab, clone the repository if not already cloned:
   ```python
   !git clone https://github.com/Miyonk/smrd-reproduction.git
   %cd smrd-reproduction
   ```
4. Run the notebook cells in order:
   - **Cell 1 (Dependencies):** Installs exact pinned requirements and checks GPU.
   - **Cell 2 (Data):** Sets project path and downloads NLTK corpora.
   - **Cell 3 (Hardware Probe):** Runs `python -m src.run probe` (verifies $\ge 7$ GiB VRAM).
   - **Cell 4 (Feature Caching & Master):** Runs `cache --all-datasets` and builds master seed checkpoint.
   - **Cell 5 (Gates):** Runs preflight P1–P4 and hardware gates G1–G6.
   - **Cell 6 (Pilot):** Runs Arm B on Twitter16 (36 DHOA evaluations + 1 final retrain).
   - **Cell 7 (Full Matrix & Reports):** Runs all nine arm $\times$ dataset combinations and builds final headline tables.

---

### Method 2: Local / GPU Workstation

#### 1. Setup Environment
**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('stopwords','wordnet','omw-1.4','punkt')]"
```

#### 2. End-to-End Execution Sequence
```bash
# Step 1: Probe hardware and preflight checks
python -m src.run probe
python -m src.run preflight

# Step 2: Build RoBERTa feature caches (FP16 memmap arrays)
python -m src.run cache --all-datasets

# Step 3: Build master initial checkpoint (seed=100)
python -m src.run master

# Step 4: Run hardware validation gates (G1-G6)
python -m src.run gates --include-stress

# Step 5: Run Pilot (Arm B on Twitter16: 36 evaluations + 1 retrain)
python -m src.run pilot

# Step 6: Run all nine combinations
python -m src.run all

# Step 7: Compile headline comparison report & discrepancy analysis
python -m src.run report
```

---

### Resuming & Individual Task Arms

- **Resume an interrupted run:**
  ```bash
  # Skips any arm/dataset combinations that already completed on disk:
  python -m src.run all --resume
  ```

- **Run an individual arm on a specific dataset:**
  ```bash
  # Run DHOA search (36 evaluations):
  python -m src.run search --arm arm_a --dataset twitter16

  # Retrain winning candidate on train split and evaluate on test:
  python -m src.run retrain --arm arm_a --dataset twitter16
  ```

- **Run post-matrix sensitivity check (Twitter16 extended epochs):**
  ```bash
  python scripts/check_twitter_collapse_sensitivity.py --epochs 10
  ```

- **Run fast smoke test (CPU / local laptop):**
  ```bash
  python -m src.run smoke
  ```

---

## Disclaimer

This repository reproduces methodology from a published paper for research verification purposes only. The paper's datasets are used under their original licences. This code is written independently; no authors' code was copied or referenced.

