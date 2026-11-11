"""
Stacked arm: a weighted average of the class probabilities of several arms.

For one fold, every member arm supplies probabilities for the validation rows
and the test rows (``probs.json`` in its fold directory).  The weights are
chosen on the **validation fold** from a grid on the simplex (macro-F1, then
log-loss, then the most even weights); the test rows are then combined once
with those weights.  No member is retrained and no test label is read before
the weights are fixed.

Caveat: each member already used the validation fold for its own model
selection (early stopping / ``C``), so validation scores are optimistic and
the weights are chosen on slightly flattering numbers.  The test estimate is
unaffected.

NumPy only.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.evaluate import classification_metrics

__all__ = ["simplex_grid", "combine", "choose_weights", "read_probs"]


def simplex_grid(n_members: int, steps: int = 10) -> List[Tuple[float, ...]]:
    """All weight vectors with entries k/steps that sum to 1."""
    if n_members < 1:
        raise ValueError("need at least one member")
    return [
        tuple(k / steps for k in ks)
        for ks in itertools.product(range(steps + 1), repeat=n_members)
        if sum(ks) == steps
    ]


def combine(probs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if len(probs) != len(weights):
        raise ValueError("one weight per member")
    out = sum(w * np.asarray(p, dtype=float) for w, p in zip(weights, probs))
    return out / out.sum(axis=1, keepdims=True)


def choose_weights(val_probs: Sequence[np.ndarray], y_val: Sequence[int], n_classes: int, steps: int = 10) -> Dict:
    """Best simplex weights on the validation rows."""
    y = np.asarray(y_val, dtype=int)
    best_key, best = None, None
    for w in simplex_grid(len(val_probs), steps):
        p = combine(val_probs, w)
        m = classification_metrics(y.tolist(), p.argmax(1).tolist(), n_classes)
        log_loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())
        # higher macro-F1, then lower log-loss, then more even weights
        key = (-round(m["macro_f1"], 12), round(log_loss, 12), max(w))
        if best_key is None or key < best_key:
            best_key = key
            best = {"weights": list(w), "val_macro_f1": m["macro_f1"],
                    "val_accuracy": m["accuracy"], "val_log_loss": log_loss}
    return best


def read_probs(path: Path | str, val_rows: Sequence[int], test_rows: Sequence[int], n_classes: int):
    """``(val_probs, test_probs)`` from a member's ``probs.json``, checked
    against the fold's rows."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for name, rows in (("val", val_rows), ("test", test_rows)):
        if raw[name]["rows"] != [int(i) for i in rows]:
            raise ValueError(f"{path}: {name} rows do not match this fold")
        p = np.asarray(raw[name]["probs"], dtype=float)
        if p.shape != (len(rows), n_classes) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError(f"{path}: bad {name} probabilities")
        if not np.allclose(p.sum(1), 1.0, atol=1e-3):
            raise ValueError(f"{path}: {name} probabilities do not sum to 1")
        out.append(p)
    return out[0], out[1]
