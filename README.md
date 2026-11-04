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
| Phase 3 — Implementation | 🔄 In progress |

---

## Design Document

[`PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md`](PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md) is the frozen, standalone specification for the full implementation. All code must implement exactly what is described there, or document deviations in `documentation/assumptions.md`.

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
| Feature extractor | RoBERTa-base (frozen, eval mode) |
| Bridge | `Linear(768→1024)` + `LayerNorm(1024)` |
| Backbone | GPT-2 Medium, NL∈[1,48] layers |
| Optimizer | DHOA (Deer Hunting Optimization Algorithm) |
| Training | AdamW, constant LR, 2 epochs/candidate |
| Task arms | Arm A (4-class), Arm B (rumour/non-rumour), Arm D (veracity) |

---

## Reproducibility

- Every paper-specified value is used unchanged.
- Every unspecified choice is labelled `T3 (IMPLEMENTATION ASSUMPTION)` in the design doc and in code comments.
- A failure to reproduce is a valid and reportable outcome.

---

## Structure

```
data/
  processed/       # Cleaned CSVs (twitter15, twitter16, pheme)
  raw/             # Original CSVs
  splits/          # train/val/test JSON split files
src/
  data_loader.py   # Phase 1 data loading
  preprocess.py    # Phase 1 preprocessing stub
tests/
  verify_phase1.py # Phase 1 verification tests
PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md
```

---

## Disclaimer

This repository reproduces methodology from a published paper for research verification purposes only. The paper's datasets are used under their original licences. This code is written independently; no authors' code was copied or referenced.

