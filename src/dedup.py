"""
Near-duplicate clustering of source tweets.

Random splits on Twitter15/16/PHEME put near-identical tweets from one event on
both sides of a split.  This module groups such tweets into clusters once per
dataset and writes them to ``data/splits/<dataset>_dup_clusters.json``.

The cluster file is the single source of truth for every later split:

* the standard protocol keeps the existing seed-42 splits and additionally
  reports a *dedup-clean* test subset (test rows whose cluster has no member
  in train or validation), see :meth:`DupClusters.clean_test_indices`;
* event-separated / chronological splits built later must keep each cluster
  on one side; :meth:`DupClusters.groups` returns cluster ids usable as the
  ``groups`` argument of scikit-learn's group splitters, and
  :meth:`DupClusters.cross_group_clusters` lists clusters that straddle a
  grouping such as PHEME events.

Rows are identified by their position in ``data/processed/<dataset>.csv``;
``tweet_ids`` is stored alongside so the file can be re-joined by id and is
validated against the CSV on load.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.config import SPLITS_DIR

MATCH_KEY_VERSION = "1"
DEFAULT_THRESHOLD = 0.9
METHOD = "tfidf_char_wb_3-5_cosine"

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_NON_ALNUM = re.compile(r"[^a-z0-9# ]+")
_URL_TOKEN = re.compile(r"\burl\b")
_RT = re.compile(r"^\s*rt\b")


def match_key(text: object) -> str:
    """Aggressive normalisation used only for duplicate matching."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _URL_TOKEN.sub(" ", text)
    text = _RT.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smallest index is the canonical representative.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def find_clusters(
    texts: Sequence[str], threshold: float = DEFAULT_THRESHOLD, block: int = 1024
) -> List[int]:
    """Return ``cluster_of[i]`` = smallest row index of row i's cluster.

    Two rows are linked when the cosine similarity of their character 3-5 gram
    TF-IDF vectors (on :func:`match_key`) is >= ``threshold``; clusters are the
    connected components.  Rows with an empty key stay singletons.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    keys = [match_key(t) for t in texts]
    uf = _UnionFind(n)
    nonempty = [i for i, k in enumerate(keys) if k]
    if len(nonempty) > 1:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32
        )
        matrix = vectorizer.fit_transform([keys[i] for i in nonempty])
        transposed = matrix.T.tocsc()
        m = matrix.shape[0]
        for start in range(0, m, block):
            sims = (matrix[start : start + block] @ transposed).toarray()
            rows, cols = np.nonzero(sims >= threshold - 1e-6)
            for r, c in zip(rows.tolist(), cols.tolist()):
                a = start + r
                if c > a:
                    uf.union(nonempty[a], nonempty[c])
    return [uf.find(i) for i in range(n)]


@dataclass
class DupClusters:
    dataset: str
    threshold: float
    tweet_ids: List[str]
    cluster_of: List[int]

    # -- construction / persistence --------------------------------------
    @classmethod
    def build(
        cls,
        dataset: str,
        tweet_ids: Sequence[object],
        texts: Sequence[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> "DupClusters":
        return cls(
            dataset=dataset,
            threshold=threshold,
            tweet_ids=[str(t) for t in tweet_ids],
            cluster_of=find_clusters(texts, threshold=threshold),
        )

    def multi_clusters(self) -> List[List[int]]:
        members: Dict[int, List[int]] = {}
        for row, cid in enumerate(self.cluster_of):
            members.setdefault(cid, []).append(row)
        return [rows for rows in members.values() if len(rows) > 1]

    def to_json(self, labels: Sequence[int] | None = None) -> Dict:
        multi = self.multi_clusters()
        payload = {
            "dataset": self.dataset,
            "method": METHOD,
            "match_key_version": MATCH_KEY_VERSION,
            "threshold": self.threshold,
            "row_index": "position in data/processed/<dataset>.csv",
            "n_rows": len(self.cluster_of),
            "n_clusters": len(set(self.cluster_of)),
            "n_multi_row_clusters": len(multi),
            "rows_in_multi_row_clusters": int(sum(len(c) for c in multi)),
            "tweet_ids": self.tweet_ids,
            "cluster_of": self.cluster_of,
            "multi_row_clusters": multi,
        }
        if labels is not None:
            payload["mixed_label_clusters"] = [
                rows for rows in multi if len({int(labels[r]) for r in rows}) > 1
            ]
        return payload

    def save(self, path: Path | str, labels: Sequence[int] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(labels)), encoding="utf-8")
        return path

    @classmethod
    def load(
        cls, path: Path | str, expected_tweet_ids: Sequence[object] | None = None
    ) -> "DupClusters":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        clusters = cls(
            dataset=raw["dataset"],
            threshold=float(raw["threshold"]),
            tweet_ids=[str(t) for t in raw["tweet_ids"]],
            cluster_of=[int(c) for c in raw["cluster_of"]],
        )
        if expected_tweet_ids is not None:
            expected = [str(t) for t in expected_tweet_ids]
            if expected != clusters.tweet_ids:
                raise ValueError(
                    f"{path} was built for a different row order/contents of "
                    f"{clusters.dataset}; rebuild it with `python -m src.run_encoder dedup`"
                )
        return clusters

    # -- split helpers ----------------------------------------------------
    def groups(self, indices: Iterable[int]) -> List[int]:
        """Cluster id per row, for GroupKFold/GroupShuffleSplit ``groups=``."""
        return [self.cluster_of[i] for i in indices]

    def clean_test_indices(
        self, train: Sequence[int], val: Sequence[int], test: Sequence[int]
    ) -> List[int]:
        """Test rows whose duplicate cluster has no member in train or val."""
        seen = {self.cluster_of[i] for i in list(train) + list(val)}
        return [i for i in test if self.cluster_of[i] not in seen]

    def leakage(
        self, train: Sequence[int], val: Sequence[int], test: Sequence[int]
    ) -> Dict[str, int]:
        train_c = {self.cluster_of[i] for i in train}
        val_c = {self.cluster_of[i] for i in val}
        clean = self.clean_test_indices(train, val, test)
        return {
            "test_rows": len(test),
            "test_rows_with_duplicate_in_train_or_val": len(test) - len(clean),
            "clean_test_rows": len(clean),
            "val_rows_with_duplicate_in_train": sum(
                1 for i in val if self.cluster_of[i] in train_c
            ),
            "test_rows_with_duplicate_in_val_only": sum(
                1
                for i in test
                if self.cluster_of[i] in val_c and self.cluster_of[i] not in train_c
            ),
        }

    def cross_group_clusters(self, group_of_row: Sequence[object]) -> List[List[int]]:
        """Clusters whose rows fall in more than one group (e.g. PHEME event)."""
        return [
            rows
            for rows in self.multi_clusters()
            if len({group_of_row[r] for r in rows}) > 1
        ]


def clusters_path(dataset: str, splits_dir: Path | str = SPLITS_DIR) -> Path:
    return Path(splits_dir) / f"{dataset}_dup_clusters.json"


def load_or_build_clusters(
    dataset: str,
    df,
    threshold: float = DEFAULT_THRESHOLD,
    splits_dir: Path | str = SPLITS_DIR,
    write: bool = True,
) -> DupClusters:
    """Load the cluster file for ``df`` or build (and optionally write) it."""
    path = clusters_path(dataset, splits_dir)
    if path.exists():
        clusters = DupClusters.load(path, expected_tweet_ids=df["tweet_id"].tolist())
        if abs(clusters.threshold - threshold) < 1e-9:
            return clusters
    clusters = DupClusters.build(
        dataset, df["tweet_id"].tolist(), df["raw_text"].astype(str).tolist(), threshold
    )
    if write:
        clusters.save(path, labels=df["label"].astype(int).tolist())
    return clusters
