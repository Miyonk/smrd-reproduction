# smrd-reproduction

An **independent reproduction** of the paper:

> *"Enhanced Large Language Model with a Self-Attention Mechanism for Social Media Rumor Detection"*

The goal is to verify the reported results (Twitter15: 99.50%, Twitter16: 99.30%, PHEME: 99.10% accuracy) against the publicly available datasets. **This is not the authors' original code.**

---

## Project Status

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

---

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

