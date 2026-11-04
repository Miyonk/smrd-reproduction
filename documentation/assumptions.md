# Implementation assumptions and deviations (T3 log)

The frozen specification is
[`PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md`](../PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md).
This file records every place where the implementation had to choose a concrete
mechanism, and every deviation forced by the available hardware. Each item is
labelled with the design-document assumption ID where one exists.

Status: Codebase implementation complete; pipeline blocker resolved
(TextPreprocessor restored and verified). Hardware validation gates and DHOA
searches have NOT yet been run on disk (no gate logs exist yet). Real DHOA
searches are intended for a Colab T4 (16 GiB).

---

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.10 (CPython, MSC v.1943) |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.22.1 |
| GPU | NVIDIA RTX 3050 Laptop, 6.0 GiB total, ~5.03 GiB usable |
| Compute capability | 8.6 (BF16 supported) |
| CUDA | 12.8 |

### T3-HW-1 — Local VRAM is below the Phase 1 gate

Design §11.5 requires **usable VRAM ≥ 7.0 GiB** before any run
(`ENVIRONMENT_INCOMPATIBLE_REPORT` otherwise). The local machine reports
5.03 GiB, so the local probe **fails by design and is recorded as a failure**
(`logs/environment_probe.json`). This is not silently overridden.

Mitigations used for local development and pilot runs only:

* `--max-layers 24` (caps `NL` at 24; the design range remains `[1, 48]`),
* `--use-8bit` (bitsandbytes `AdamW8bit`),
* `--gradient-checkpointing`,
* `--B-phys-max 1` (the automatic OOM ladder still applies),
* BF16 autocast where supported.

All four appear in every candidate log under `execution_stats`. Final reported
runs must use the unmodified design configuration on ≥ 7 GiB hardware.

### T3-HW-2 — Optimizer memory

The design specifies AdamW with `weight_decay=0.01`, `betas=(0.9, 0.999)`.
AdamW8bit is a memory-only substitution: the update rule and hyperparameters
are unchanged. It is opt-in (`--use-8bit`) and recorded per candidate.

### T3-HW-3 — AMP dtype

The design leaves the AMP dtype to the hardware (§6, A-AMP). The code picks
BF16 when `torch.cuda.is_bf16_supported()`, else FP16, else FP32. The chosen
dtype is logged in `execution_stats.amp_dtype`. FP16 uses a `GradScaler`;
BF16 does not.

---

## 2. Preprocessing (§2)

Implemented exactly as §2.1/§2.2: URL removal → `text[:512]` → `TweetTokenizer`
(case-preserving) → case-insensitive NLTK stopword removal → Light-10 → two-pass
WordNet lemmatization (`verb` then `noun`, both on `token.lower()`) → single-space
join → RoBERTa BPE with `max_length=512, truncation=True,
add_special_tokens=True`.

### T3-PRE-1 — Literal `URL` placeholders

2,023 processed rows contain the literal token `URL` because the upstream
release replaced links before distribution. The pipeline has no HTML/URL-token
special case, so `URL` is lowercased by lemmatization to `url` and kept as an
ordinary token. It is **not** deleted: doing so would add an operation that the
paper does not specify.

### T3-PRE-2 — NLTK data availability

`nltk` resources (`stopwords`, `wordnet`, `omw-1.4`, `punkt`) must be installed.
If they are missing the code logs a warning and falls back to a bundled
stopword list plus a suffix lemmatizer. The backend used is stored in
`metadata.json` (`preprocessor_backend`: `nltk` for all reported runs). A run
with `fallback` is **not** spec-exact and must not be reported as such.

### T3-PRE-3 — Empty input

`TextPreprocessor` raises `PreprocessingError` for non-strings and for
whitespace-only strings, per the `TextPreprocessor` contract in §11.6. No
dataset row is empty (verified for Twitter15/16 and PHEME).

---

## 3. Feature cache (§2.3)

* Flat FP16 memmap `(sum T_i, 768)`, `metadata.json`, `manifest.sha256`.
* RoBERTa runs in `eval()` in FP32 on GPU; features are cast to FP16 on write.
* `reference_check` recomputes a sample in FP32 and compares against the FP16
  cache with `rtol=1e-3, atol=1e-3`.
* The manifest hashes the feature binary, the metadata file, the preprocessed
  text, the RoBERTa checkpoint name, and library versions.

---

## 4. Model (§3, §4)

### T3-MODEL-1 — Backbone contract

The design pins GPT-2 Medium (`d_model=1024`, 16 heads, 24 base blocks). The
loader rejects any checkpoint whose `n_embd`, `n_head`, or `n_layer` violates
that contract (`DimensionMismatchError`), instead of silently building a
mixed-width model. `gpt2_model` defaults to `gpt2-medium`.

### T3-MODEL-2 — `transformers` 5.x has no `head_mask`

`GPT2Attention.forward` no longer accepts `head_mask`. Dynamic head masking is
implemented with a forward **pre-hook** on every `attn.c_proj`: the flattened
head slices `[NAH*64:]` are zeroed before the output projection. This is
mathematically identical to zeroing the head outputs, and no variance rescaling
is applied (per §4 and S14). The hook is removed from the accounting of any
`NAH = 16` forward because it returns the input unchanged in that case.

### T3-MODEL-3 — Appended blocks

`NL > 24` uses zero-initialised `attn.c_proj`/`mlp.c_proj`, so each appended
block is an exact identity at initialisation (verified by probe P3 and
`tests/test_model.py`). Candidate builds materialise only
`max(0, NL − 24)` appended blocks; the master checkpoint stores all 24.

### T3-MODEL-4 — `wte` pruning

`base.wte` is set to `None`; inputs always arrive through `inputs_embeds`.
This removes 38.6M parameters (the design quotes 51.46M for the full embedding
matrix including position embeddings; the realised saving is measured and
recorded in `logs/master_checkpoint.json`).

### T3-MODEL-5 — Pooling

Default is last-valid-token pooling as specified. `--pooling mean` exists only
for ablation S12; it is never used in reported runs.

### T3-MODEL-6 — Master rotation

The master checkpoint is written once after `torch.manual_seed(seed_init)` and
is never rotated. Every candidate reloads it, which gives the isolation
guarantee of §6.2 (probe P4).

---

## 5. Training (§6)

* Exact accumulation: a logical batch of size `b` is split into micro-batches;
  each micro loss is multiplied by `len(micro)/b` before `backward()`. Gate G5
  verifies gradient equivalence at `max_abs_diff ≤ 1e-4`.
* OOM ladder (§6.4): halve `B_phys` → enable gradient checkpointing →
  `B_phys=1` → `INFEASIBLE_HARDWARE`. Divergence yields
  `DIVERGED_NONFINITE`, never fitness `0.0`.
* Gradient clipping `max_norm=1.0` (A-AMP).
* Two epochs per candidate; validation accuracy is the fitness.

---

## 6. DHOA (§5)

### T3-DHOA-1 — Search space

Optimisation runs in `[0,1]^5` and maps to `L ∈ [1e-5, 5e-4]` (log by default),
`B ∈ [1,512]`, `NL ∈ [1,48]`, `NAH ∈ [1,16]`, `DR ∈ [0.1,0.5]`. Rounding is
nearest-integer with clamping. `NL` uses `max_layers` as its upper bound when a
local cap is set (T3-HW-1).

### T3-DHOA-2 — Branch mapping (A-DHOA-BRANCH)

Primary mapping used for reported runs:

| Condition | Equation |
|---|---|
| `|l| < 1` and `|P| ≥ 1` | Eq. (7) |
| `|l| < 1` and `|P| < 1` | Eq. (14) |
| `|l| ≥ 1` | Eq. (13) |

The S10 alternative is selectable via `dhoa_branch_profile="alternative"`.

### T3-DHOA-3 — Stochastic symbols

`z ~ U(-1,1)`, `w ~ U(0,1)`, `l ~ U(0,2)`, `r ~ U(0,1)`, `θ0 ~ U(0,2π)`,
`φ0 ~ U(0,2π)`, `v = φ_{i+1}` (S10 alternative:
`v ~ U(-π/4, π/4)`), constant `θ` by default (S10 alternative: Gaussian random
walk). `|P|` is reduced to a scalar by mean absolute value by default
(`dhoa_p_norm`). All seeded from `seed_dhoa`.

### T3-DHOA-4 — Evaluation accounting

Exactly 6 deer × (1 initial + 5 update rounds) = 36 evaluations. Duplicate
coordinate tuples are served from a persistent cache; cache hits still count as
evaluations in the 36-slot budget but are marked `CACHE_HIT` and do not retrain.

---

## 7. Evaluation (§7)

* Accuracy, precision, recall, F1, macro-F1, FPR, FNR, MCC, NPV, confusion
  matrix, and per-class metrics as specified.
* 95 % Wilson score interval on test accuracy.
* Exact two-sided McNemar test on paired correctness; bootstrap accuracy uses a
  fixed seed (`seed_dhoa`) and 10,000 resamples.
* Report C collapses 4-class Arm A predictions to rumour/non-rumour.

---

## 8. Artifacts

```
checkpoints/master_init_checkpoint.pt   # 48-block master
checkpoints/winners/<arm>_<dataset>.pt  # final retrained winner
logs/environment_probe.json             # P1
logs/probe_p2_cache.json                # P2
logs/probe_p3_appended_grad.json        # P3
logs/probe_p4_isolation.json            # P4
logs/gate_g1..g6_*.json                 # G1-G6
logs/candidate_evaluations/*.json       # one record per evaluation
logs/dhoa_state_<arm>_<dataset>.json    # resumable DHOA state
logs/dhoa_result_<arm>_<dataset>.json   # leader/successor/history
logs/final_<arm>_<dataset>.json         # retrain + test metrics
reports/headline_comparison.{csv,md}    # cross-arm table
```

---

## 9. Known open items

1. Real DHOA searches have **not** been executed on ≥ 7 GiB hardware in this
   environment. The Colab runner (`notebooks/colab_runner.ipynb`) is provided
   for that step; the local 6 GiB machine is used for preprocessing, caching,
   master creation, correctness probes, and smoke tests only.
2. The design's target numbers (99.5 / 99.3 / 99.1) are not assumed. A failure
   to reproduce them is reported as a result, per README.
3. Ablation tiers S1–S14 are wired into the configuration switches
   (`pooling`, `head_rescale`, `dhoa_*`, preprocessing flags) but only the main
   runs are part of the default runbook.
