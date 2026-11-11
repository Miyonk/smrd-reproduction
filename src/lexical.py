"""
Lexical arm: TF-IDF features + logistic regression (CPU, scikit-learn only).

Features, all fitted on the training rows of the fold:

* source tweet: word 1-2-gram and character 2-5-gram TF-IDF of the normalised text;
* thread: TF-IDF of the bag of words of every node in the propagation tree
  (the same node features the BiGCN arm reads, with the tree structure dropped).

Same discipline as the neural arms: the regularisation strength ``C`` is
chosen on the **validation fold** (macro-F1), and the test fold is predicted
once, with the chosen model only.  Validation rows are never trained on.

Besides ``result.json`` (same layout as the other arms) the run writes
``probs.json`` with class probabilities for the validation and test rows, which
``src/stack.py`` combines with other arms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import replace_file
from src.data import ArmView
from src.evaluate import classification_metrics, test_report

LOGGER = logging.getLogger(__name__)

__all__ = ["LexicalConfig", "run_lexical", "thread_bags", "write_probs", "PROBS_FILE"]

PROBS_FILE = "probs.json"


@dataclass
class LexicalConfig:
    dataset: str = "twitter16"
    task: str = "4way"
    c_grid: Tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)
    word_ngrams: Tuple[int, int] = (1, 2)
    char_ngrams: Tuple[int, int] = (2, 5)
    use_thread: bool = True
    class_weight: str = "balanced"
    max_iter: int = 2000
    graph_cache_sha256: str = ""

    def to_dict(self) -> Dict:
        return json.loads(json.dumps(asdict(self)))  # tuples -> lists, as read back


def thread_bags(graphs: Sequence) -> "scipy.sparse.csr_matrix":
    """One row per tree: word counts summed over all its nodes."""
    import scipy.sparse as sp

    n_words = 1 + max((int(g.words.max()) for g in graphs if len(g.words)), default=0)
    rows = np.concatenate([np.full(len(g.words), i, dtype=np.int64) for i, g in enumerate(graphs)])
    cols = np.concatenate([np.asarray(g.words, dtype=np.int64) for g in graphs])
    vals = np.concatenate([np.asarray(g.counts, dtype=np.float64) for g in graphs])
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(graphs), n_words))  # duplicates are summed


def build_features(cfg: LexicalConfig, texts: Sequence[str], bags, train_rows: Sequence[int]):
    """Feature matrix for every row; vocabularies and idf from ``train_rows`` only."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

    train_texts = [texts[i] for i in train_rows]
    blocks = [
        TfidfVectorizer(ngram_range=tuple(cfg.word_ngrams), sublinear_tf=True)
        .fit(train_texts).transform(texts),
        TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg.char_ngrams), sublinear_tf=True)
        .fit(train_texts).transform(texts),
    ]
    if cfg.use_thread:
        blocks.append(TfidfTransformer(sublinear_tf=True).fit(bags[list(train_rows)]).transform(bags))
    return sp.hstack(blocks).tocsr()


def write_probs(path: Path | str, parts: Dict[str, Tuple[Sequence[int], np.ndarray]]) -> None:
    payload = {
        name: {"rows": [int(i) for i in rows], "probs": np.asarray(probs, dtype=float).round(6).tolist()}
        for name, (rows, probs) in parts.items()
    }
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    replace_file(tmp, path)


def run_lexical(
    cfg: LexicalConfig,
    ckpt_dir: Path | str,
    *,
    df,
    view: ArmView,
    texts: List[str],
    graphs: Optional[Sequence] = None,
    bags=None,
    split_source: Optional[str] = None,
) -> Dict:
    """Fit on train, choose ``C`` on validation, predict test once."""
    from sklearn.linear_model import LogisticRegression

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_path = ckpt_dir / "result.json"
    if result_path.exists() and (ckpt_dir / PROBS_FILE).exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("config") != cfg.to_dict():
            raise RuntimeError(
                f"{result_path} exists for a different configuration; "
                "use a different checkpoint directory or --fresh"
            )
        LOGGER.info("run already finished: %s", result_path)
        return previous

    if cfg.use_thread and bags is None:
        if graphs is None:
            raise ValueError("the thread features need graphs or precomputed bags")
        bags = thread_bags(graphs)
    train, val, test = view.train_indices, view.val_indices, view.test_indices
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("train / validation / test rows overlap")
    X = build_features(cfg, texts, bags, train)
    y_train = np.array(view.train_labels)
    n_classes = view.n_classes

    best, table = None, []
    for C in cfg.c_grid:
        clf = LogisticRegression(C=C, max_iter=cfg.max_iter, class_weight=cfg.class_weight)
        clf.fit(X[train], y_train)
        if list(clf.classes_) != list(range(n_classes)):
            raise ValueError("a class is missing from the training rows")
        val_probs = clf.predict_proba(X[val])
        m = classification_metrics(view.val_labels, val_probs.argmax(1).tolist(), n_classes)
        table.append({"C": C, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
        LOGGER.info("C=%g val_acc=%.4f val_macroF1=%.4f", C, m["accuracy"], m["macro_f1"])
        if best is None or m["macro_f1"] > best[0] + 1e-12:  # ties keep the stronger regularisation
            best = (m["macro_f1"], C, clf, val_probs)
    best_f1, best_C, clf, val_probs = best

    test_probs = clf.predict_proba(X[test])  # the only time test rows are scored
    preds = test_probs.argmax(1).tolist()
    write_probs(ckpt_dir / PROBS_FILE, {"val": (val, val_probs), "test": (test, test_probs)})
    result = {
        "config": cfg.to_dict(),
        "dataset": cfg.dataset,
        "task": cfg.task,
        "arm": "lexical",
        "class_names": view.class_names,
        "split": {
            "source": split_source or f"data/splits/{cfg.dataset}_train_val_test.json (seed 42)",
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "n_features": int(X.shape[1]),
        "selection": table,
        "selected": {"C": best_C},
        "epochs_run": len(table),
        "best_epoch": [row["C"] for row in table].index(best_C) + 1,
        "best_val_macro_f1": best_f1,
        "test_full": test_report(view.test_labels, preds, n_classes, view.class_names),
        "test_predictions": [
            {
                "row": int(i),
                "tweet_id": str(df["tweet_id"].iloc[i]),
                "label": int(view.labels[i]),
                "pred": int(p),
                "confidence": float(max(pr)),
            }
            for i, p, pr in zip(test, preds, test_probs.tolist())
        ],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    replace_file(tmp, result_path)
    return result
