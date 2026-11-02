# smrd-reproduction

An **independent reproduction** of the paper:

> *"Enhanced Large Language Model with a Self-Attention Mechanism for Social Media Rumor Detection"*

The goal is to verify the reported results (Twitter15: 99.50%, Twitter16: 99.30%, PHEME: 99.10% accuracy) against the publicly available datasets. **This is not the authors' original code.**

---

## Project Status

| Phase | Status |
|---|---|
| Phase 1 — Data loading & splits | ✅ Complete |
| Phase 2B — Implementation design | ✅ Complete (`PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md`) |
| Phase 3 — Implementation | ✅ Engine complete; local gates pass; real searches require ≥ 7 GiB VRAM |

See [`documentation/assumptions.md`](documentation/assumptions.md) for every
implementation assumption and hardware deviation.

---

## Design Document

[`PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md`](PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md) is the frozen, standalone specification for the full implementation. All code implements exactly what is described there, or documents deviations in `documentation/assumptions.md`.

---

## Datasets

- **Twitter15** — 1,491 threads (4-class: non-rumour / false / true / unverified)
- **Twitter16** — 818 threads
- **PHEME** — 6,425 threads

Processed CSVs and train/val/test split files (seed=42) are included in `data/`.

---

## Architecture Overview

| Component | Specification |
|---|---|
| Feature extractor | RoBERTa-base (frozen, eval mode, FP16 memmap cache) |
| Bridge | `Linear(768→1024)` + `LayerNorm(1024)` |
| Backbone | GPT-2 Medium, `NL ∈ [1,48]` layers, `NAH ∈ [1,16]` heads |
| Optimizer | DHOA (Deer Hunting Optimization Algorithm), 6 deer × 36 evaluations |
| Training | AdamW, 2 epochs/candidate, exact gradient accumulation |
| Task arms | Arm A (4-class), Arm B (rumour/non-rumour), Arm D (veracity), Report C |

---

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

or manually:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import nltk; [nltk.download(p, quiet=True) for p in ('stopwords','wordnet','omw-1.4','punkt')]"
```

---

## Runbook

```powershell
.\.venv\Scripts\python.exe -m pytest -q                 # 37 correctness tests
.\.venv\Scripts\python.exe -m src.run probe             # P1 hardware probe
.\.venv\Scripts\python.exe -m src.run cache --dataset twitter16
.\.venv\Scripts\python.exe -m src.run master            # 48-block master checkpoint
.\.venv\Scripts\python.exe -m src.run preflight         # P1-P4
.\.venv\Scripts\python.exe -m src.run gates             # G1, G2, G5
.\.venv\Scripts\python.exe -m src.run pilot             # Arm B / Twitter16 search + retrain
.\.venv\Scripts\python.exe -m src.run all               # all nine arm/dataset runs
.\.venv\Scripts\python.exe -m src.run report            # headline tables
```

The local RTX 3050 (5.03 GiB usable) is **below** the design's 7.0 GiB Phase 1
gate, so P1 correctly reports `ENVIRONMENT_INCOMPATIBLE_REPORT`. Use
[`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) on a Colab T4
for the real searches. Local-only mitigations:

```powershell
python -m src.run pilot --max-layers 24 --use-8bit --gradient-checkpointing --B-phys-max 1
```

---

## Structure

```
data/
  processed/       # Cleaned CSVs (twitter15, twitter16, pheme)
  raw/             # Original CSVs
  splits/          # train/val/test JSON split files
src/
  config.py        # all design constants + CLI overrides
  preprocess.py    # §2 spec pipeline (TextPreprocessor) + legacy Phase 1 helpers
  data.py          # arm views and split selection
  features.py      # RoBERTa FP16 memmap cache + manifest
  model.py         # bridge, dynamic GPT-2, head masking, checkpoints
  dhoa.py          # Equations 4-14, 36-evaluation search, resume
  train.py         # accumulation trainer, OOM ladder, test evaluation
  evaluate.py      # Equations 15-22, Wilson, McNemar, bootstrap
  gates.py         # P1-P4 and G1-G6
  run.py           # CLI orchestration
tests/             # pytest suite (37 tests)
scripts/           # setup + diagnostic benchmarks
notebooks/         # Colab T4 runner
documentation/     # T3 assumptions and deviations
```

---

## Reproducibility

- Every paper-specified value is used unchanged.
- Every unspecified choice is labelled `T3 (IMPLEMENTATION ASSUMPTION)` in the design doc and in code comments.
- A failure to reproduce is a valid and reportable outcome.

---

## Disclaimer

This repository reproduces methodology from a published paper for research verification purposes only. The paper's datasets are used under their original licences. This code is written independently; no authors' code was copied or referenced.
