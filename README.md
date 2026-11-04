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

## Execution Guide

```powershell
# 1. Environment probe & preflight
python -m src.run probe
python -m src.run preflight

# 2. Extract RoBERTa feature caches (offline FP16)
python -m src.run cache --dataset twitter16
python -m src.run cache --all-datasets

# 3. Create master initial checkpoint (seed=100)
python -m src.run master

# 4. Run hardware validation gates
python -m src.run gates --include-stress

# 5. Run Pilot (Arm B on Twitter16: 36 evaluations + 1 retrain)
python -m src.run pilot

# 6. Run all nine arm x dataset combinations
python -m src.run all

# 7. Generate comparison reports and discrepancy analysis
python -m src.run report
```

---

## Disclaimer

This repository reproduces methodology from a published paper for research verification purposes only. The paper's datasets are used under their original licences. This code is written independently; no authors' code was copied or referenced.

