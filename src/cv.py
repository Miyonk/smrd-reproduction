"""
Grouped, stratified 5-fold cross-validation and paired arm comparison.

Why: the seed-42 split has 164 / 298 / 1,285 test rows.  The sampling error of
accuracy on those sets (about +-3 / +-2.3 / +-1.1 points) is as large as the
differences between arms, and its validation sets (82 / 149 rows) are too
small to tune against.  Seed-to-seed std does not capture that error.

Protocol
--------
* ``StratifiedGroupKFold(5, shuffle, seed 42)`` on the 4-class label, with
  near-duplicate clusters (``src/dedup.py``) as groups: a tweet and its
  near-copies always share a fold, so no test row has a near-duplicate in
  train or validation.
* Fold ``k`` is the test set, fold ``(k + 1) % 5`` the validation set (model
  selection only), the other three folds train.
* Every row is tested exactly once; arms are compared on the pooled
  predictions with an exact McNemar test and a paired accuracy difference.

The fold assignment is written to ``data/splits/<dataset>_cv5_grouped.json``
and tracked in git: it is the source of truth (clusters are not recomputed at
run time, so library versions cannot move rows between folds).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import SPLITS_DIR
from src.data import ArmView

N_FOLDS = 5
FOLD_SEED = 42
FORMAT_VERSION = 1


def folds_path(dataset: str, splits_dir: Path | str = SPLITS_DIR) -> Path:
    return Path(splits_dir) / f"{dataset}_cv{N_FOLDS}_grouped.json"


def make_folds(
    labels: Sequence[int],
    groups: Sequence[int],
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
) -> List[int]:
    """Fold id per row; rows sharing a group share a fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = np.asarray(labels)
    fold_of_row = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        fold_of_row[test] = k
    if (fold_of_row < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")
    return fold_of_row.tolist()


def build_folds_file(dataset: str, df, threshold: Optional[float] = None, path: Optional[Path] = None) -> Dict:
    from src.dedup import DEFAULT_THRESHOLD, find_clusters

    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    labels = df["label"].astype(int).tolist()
    groups = find_clusters(df["raw_text"].astype(str).tolist(), threshold=threshold)
    fold_of_row = make_folds(labels, groups)
    n_classes = max(labels) + 1
    payload = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "n_folds": N_FOLDS,
        "seed": FOLD_SEED,
        "stratify": "label (4-class)",
        "grouping": f"near-duplicate clusters, char 3-5gram TF-IDF cosine >= {threshold}",
        "protocol": "test = fold k, val = fold (k+1) % n_folds, train = the rest",
        "fold_sizes": [fold_of_row.count(k) for k in range(N_FOLDS)],
        "label_counts_per_fold": [
            [sum(1 for f, y in zip(fold_of_row, labels) if f == k and y == c) for c in range(n_classes)]
            for k in range(N_FOLDS)
        ],
        "tweet_ids": df["tweet_id"].astype(str).tolist(),
        "group_of_row": [int(g) for g in groups],
        "fold_of_row": fold_of_row,
    }
    path = Path(path or folds_path(dataset))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


@dataclasses.dataclass
class Folds:
    dataset: str
    fold_of_row: List[int]
    group_of_row: List[int]
    n_folds: int
    sha256: str

    @classmethod
    def load(cls, dataset: str, tweet_ids: Sequence[str], path: Optional[Path] = None) -> "Folds":
        path = Path(path or folds_path(dataset))
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; build it with `python -m src.run_cv make-folds`")
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if raw.get("format_version") != FORMAT_VERSION or raw.get("dataset") != dataset:
            raise ValueError(f"{path} is not a v{FORMAT_VERSION} fold file for {dataset}")
        if [str(t) for t in tweet_ids] != raw["tweet_ids"]:
            raise ValueError(f"{path} does not match the dataset rows (tweet ids differ)")
        # hash the parsed assignment, not the bytes: immune to line-ending conversion
        digest = hashlib.sha256(json.dumps(raw["fold_of_row"]).encode()).hexdigest()
        return cls(dataset, list(raw["fold_of_row"]), list(raw["group_of_row"]), int(raw["n_folds"]), digest)

    def rows(self, fold: int) -> List[int]:
        return [i for i, f in enumerate(self.fold_of_row) if f == fold]

    def view(self, base: ArmView, fold: int) -> ArmView:
        """``base`` with train/val/test replaced by this fold's rows (rows the
        arm excludes stay excluded)."""
        if not 0 <= fold < self.n_folds:
            raise ValueError(f"fold must be in [0, {self.n_folds})")
        val_fold = (fold + 1) % self.n_folds
        keep = lambda rows: [i for i in rows if i in base.labels]
        test, val = keep(self.rows(fold)), keep(self.rows(val_fold))
        train = keep([i for i, f in enumerate(self.fold_of_row) if f not in (fold, val_fold)])
        return dataclasses.replace(base, train_indices=train, val_indices=val, test_indices=test)

    def split_source(self, fold: int) -> str:
        return (
            f"{folds_path(self.dataset).name} fold {fold} "
            f"(val fold {(fold + 1) % self.n_folds}; folds sha256 {self.sha256[:16]})"
        )

    def check_no_group_leak(self) -> None:
        fold_of_group: Dict[int, int] = {}
        for g, f in zip(self.group_of_row, self.fold_of_row):
            if fold_of_group.setdefault(g, f) != f:
                raise ValueError(f"duplicate cluster {g} spans folds")


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` vs ``c`` discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_difference(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict:
    """Accuracy(A) - accuracy(B) on the same rows, with a 95% interval for the
    paired difference (normal approximation) and the exact McNemar p-value."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired arms must cover the same rows")
    n = len(correct_a)
    only_a = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    only_b = sum(1 for a, b in zip(correct_a, correct_b) if b and not a)
    diff = (only_a - only_b) / n
    se = math.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0.0)) / n
    return {
        "n": n,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "accuracy_difference": diff,
        "difference_ci95": [diff - 1.96 * se, diff + 1.96 * se],
        "mcnemar_p": mcnemar_exact(only_a, only_b),
    }


def pool_fold_results(results: Sequence[Dict], expected_rows: Sequence[int]) -> Tuple[List[int], List[int], List[int]]:
    """``(rows, labels, preds)`` sorted by row; every expected row exactly once."""
    by_row: Dict[int, Tuple[int, int]] = {}
    for r in results:
        for p in r["test_predictions"]:
            if p["row"] in by_row:
                raise ValueError(f"row {p['row']} predicted in two folds")
            by_row[int(p["row"])] = (int(p["label"]), int(p["pred"]))
    if set(by_row) != set(expected_rows):
        raise ValueError(
            f"pooled predictions cover {len(by_row)} rows, expected {len(set(expected_rows))}"
        )
    rows = sorted(by_row)
    return rows, [by_row[i][0] for i in rows], [by_row[i][1] for i in rows]
