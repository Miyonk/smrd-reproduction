"""
Propagation trees for stage 2 (the BiGCN branch).

Every dataset row gets one tree whose node 0 is the source tweet.  Nodes are
stored in breadth-first order (``parent[k] < k``), and each node carries a
bag-of-words vector over ``N_FEATURES`` vocabulary slots.

Sources
-------
* **Twitter15/16** - the public RvNN release (Ma et al., 2018;
  github.com/majingCUHK/Rumor_RvNN, ``resource/data.TD_RvNN.vol_5000.txt``),
  the same node features published BiGCN results use.  Caveat shared by every
  paper that uses this file: its vocabulary was built over the whole dataset,
  test threads included.
* **PHEME** - ``structure.json`` + ``reactions/*.json`` from the PHEME archive.
  The vocabulary (top ``N_FEATURES`` words by document frequency, min_df 2) is
  built from **training threads only** of the seed-42 split.  Reply nodes whose
  tweet JSON is missing are contracted: their children attach to the nearest
  ancestor that has text.

The parsed trees are written once to ``data/graphs/<dataset>_graphs.json.gz``
(tracked in git, so Colab needs neither archive) and loaded by
:func:`load_graphs`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.config import ROOT, replace_file
from src.encoder_preprocess import normalize_tweet

CACHE_DIR = ROOT / "data" / "graphs"
CACHE_FORMAT_VERSION = 1
N_FEATURES = 5000
RVNN_FILES = {
    "tree": "data.TD_RvNN.vol_5000.txt",
    "twitter15": "Twitter15_label_All.txt",
    "twitter16": "Twitter16_label_All.txt",
}
RVNN_URL = "https://raw.githubusercontent.com/majingCUHK/Rumor_RvNN/master/resource/"


# ---------------------------------------------------------------------------
# Tree container
# ---------------------------------------------------------------------------
@dataclass
class TreeGraph:
    """One propagation tree in BFS order; node 0 is the source tweet."""

    parent: np.ndarray  # (n,) int64, parent[0] == -1, parent[k] < k
    offsets: np.ndarray  # (n,) int64, start of node k's bag in words/counts
    words: np.ndarray  # (nnz,) int64 vocabulary ids
    counts: np.ndarray  # (nnz,) float32 counts

    @property
    def n_nodes(self) -> int:
        return int(len(self.parent))

    def depth(self) -> np.ndarray:
        d = np.zeros(self.n_nodes, dtype=np.int64)
        for k in range(1, self.n_nodes):
            d[k] = d[self.parent[k]] + 1
        return d

    def structure_features(self) -> List[float]:
        """Size/shape only (no text): used by the shortcut check."""
        n = self.n_nodes
        depth = self.depth()
        out_deg = np.bincount(self.parent[1:], minlength=n) if n > 1 else np.zeros(1)
        return [
            float(np.log1p(n)),
            float(depth.max()),
            float(depth.mean()),
            float(np.log1p(out_deg[0])),
            float(np.log1p(out_deg.max())),
            float((out_deg == 0).mean()),
        ]

    def to_json(self) -> Dict:
        return {
            "parent": self.parent.tolist(),
            "offsets": self.offsets.tolist(),
            "words": self.words.tolist(),
            "counts": [int(c) if float(c).is_integer() else float(c) for c in self.counts],
        }

    @classmethod
    def from_json(cls, d: Mapping, n_features: int = N_FEATURES) -> "TreeGraph":
        g = cls(
            parent=np.asarray(d["parent"], dtype=np.int64),
            offsets=np.asarray(d["offsets"], dtype=np.int64),
            words=np.asarray(d["words"], dtype=np.int64),
            counts=np.asarray(d["counts"], dtype=np.float32),
        )
        g.validate(n_features)
        return g

    def validate(self, n_features: int = N_FEATURES) -> None:
        n = self.n_nodes
        if n < 1 or self.parent[0] != -1:
            raise ValueError("tree must have a root at position 0")
        if n > 1 and not np.all((self.parent[1:] >= 0) & (self.parent[1:] < np.arange(1, n))):
            raise ValueError("nodes must be in BFS order (parent[k] < k)")
        if len(self.offsets) != n or (n and self.offsets[0] != 0):
            raise ValueError("offsets must have one entry per node, starting at 0")
        if np.any(np.diff(np.append(self.offsets, len(self.words))) < 0):
            raise ValueError("offsets must be non-decreasing")
        if len(self.words) != len(self.counts):
            raise ValueError("words/counts length mismatch")
        if len(self.words) and (self.words.min() < 0 or self.words.max() >= n_features):
            raise ValueError("word id out of range")


Bag = List[Tuple[int, float]]


def build_tree(
    root: Hashable,
    parent_of: Mapping[Hashable, Optional[Hashable]],
    bags: Mapping[Hashable, Bag],
    n_features: int = N_FEATURES,
) -> Tuple[TreeGraph, int]:
    """BFS-order a tree.  Nodes whose parent is unknown (or unreachable from the
    root, e.g. a cycle) are attached to the root; returns ``(tree, n_reattached)``."""
    nodes = set(parent_of) | {root}
    children: Dict[Hashable, List[Hashable]] = defaultdict(list)
    for node in sorted(nodes - {root}, key=str):
        p = parent_of.get(node)
        children[p if p in nodes and p != node else None].append(node)
    order, seen = [root], {root}
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        for ch in children.get(cur, []):
            if ch not in seen:
                seen.add(ch)
                order.append(ch)
                queue.append(ch)
    position = {node: k for k, node in enumerate(order)}
    parent = [-1] + [position[parent_of[node]] for node in order[1:]]
    # orphans and anything unreachable hang off the root
    rest = sorted(nodes - seen, key=str)
    for node in rest:
        position[node] = len(order)
        order.append(node)
        parent.append(0)
    offsets, words, counts = [], [], []
    for node in order:
        offsets.append(len(words))
        for w, c in sorted(bags.get(node, [])):
            words.append(int(w))
            counts.append(float(c))
    tree = TreeGraph(
        parent=np.asarray(parent, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        words=np.asarray(words, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.float32),
    )
    tree.validate(n_features)
    return tree, len(rest)


# ---------------------------------------------------------------------------
# Twitter15/16 (RvNN release)
# ---------------------------------------------------------------------------
def _parse_vec(vec: str) -> Bag:
    bag = []
    for tok in vec.split():
        w, c = tok.split(":")
        bag.append((int(w), float(c)))
    return bag


def parse_rvnn(rvnn_dir: Path | str, dataset: str, tweet_ids: Sequence[str]) -> Tuple[Dict[str, TreeGraph], Dict]:
    """Trees for ``tweet_ids`` from the RvNN TD file; checks labels against the
    RvNN label file so a mismatched download fails loudly."""
    rvnn_dir = Path(rvnn_dir)
    wanted = set(tweet_ids)
    rows: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    with open(rvnn_dir / RVNN_FILES["tree"], encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                raise ValueError(f"malformed RvNN line {line_no}")
            if p[0] in wanted:
                rows[p[0]].append((p[1], p[2], p[5]))
    missing = sorted(wanted - set(rows))
    if missing:
        raise ValueError(f"{len(missing)} {dataset} threads missing from RvNN file, e.g. {missing[:3]}")

    trees, reattached = {}, 0
    for tid in tweet_ids:
        parent_of: Dict[int, Optional[int]] = {}
        bags: Dict[int, Bag] = {}
        root = None
        for p, idx, vec in rows[tid]:
            node = int(idx)
            if node in bags:
                raise ValueError(f"duplicate node {node} in {dataset} tree {tid}")
            bags[node] = _parse_vec(vec)
            if p == "None":
                if root is not None:
                    raise ValueError(f"two roots in {dataset} tree {tid}")
                root = node
            else:
                parent_of[node] = int(p)
        if root is None:
            raise ValueError(f"no root in {dataset} tree {tid}")
        trees[tid], n = build_tree(root, parent_of, bags)
        reattached += n
    return trees, {"reattached_nodes": reattached}


def rvnn_labels(rvnn_dir: Path | str, dataset: str) -> Dict[str, str]:
    out = {}
    with open(Path(rvnn_dir) / RVNN_FILES[dataset], encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            out[p[2]] = p[0].lower()
    return out


# ---------------------------------------------------------------------------
# PHEME
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[#@]?[a-z0-9_']+")


def tokenize_bow(text: str) -> List[str]:
    return _TOKEN.findall(normalize_tweet(text).lower())


def build_vocab(docs: Iterable[List[str]], size: int = N_FEATURES, min_df: int = 2) -> List[str]:
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    ranked = sorted((w for w, c in df.items() if c >= min_df), key=lambda w: (-df[w], w))
    return ranked[:size]


def bag_of_words(tokens: List[str], index: Mapping[str, int]) -> Bag:
    c = Counter(index[t] for t in tokens if t in index)
    return sorted((w, float(n)) for w, n in c.items())


def read_pheme_threads(tar_path: Path | str, thread_ids: Iterable[str]) -> Dict[str, Dict]:
    """One pass over the archive: source text, reply texts/parents, structure."""
    wanted = set(thread_ids)
    threads: Dict[str, Dict] = {
        t: {"source": None, "replies": {}, "structure": None} for t in wanted
    }
    with tarfile.open(tar_path, "r:*") as tar:
        for m in tar:
            if not m.isfile() or "/._" in m.name or not m.name.endswith(".json"):
                continue
            p = m.name.split("/")
            if len(p) < 5 or p[0] != "all-rnr-annotated-threads" or p[3] not in wanted:
                continue
            th = threads[p[3]]
            if p[4] == "structure.json":
                th["structure"] = json.load(tar.extractfile(m))
            elif p[4] == "source-tweets" and len(p) >= 6:
                th["source"] = json.load(tar.extractfile(m)).get("text", "")
            elif p[4] == "reactions" and len(p) >= 6:
                d = json.load(tar.extractfile(m))
                rid = str(d.get("id_str") or p[5][: -len(".json")])
                th["replies"][rid] = {
                    "text": d.get("text", ""),
                    "in_reply_to": d.get("in_reply_to_status_id_str"),
                }
    return threads


def _structure_parents(structure, parent=None, out=None) -> Dict[str, Optional[str]]:
    out = {} if out is None else out
    if isinstance(structure, dict):
        for k, v in structure.items():
            out.setdefault(str(k), parent)
            _structure_parents(v, str(k), out)
    return out


def pheme_tree(thread_id: str, th: Dict, index: Mapping[str, int]) -> Tuple[TreeGraph, Dict]:
    """Tree for one PHEME thread; nodes without text are contracted."""
    texts = {thread_id: th["source"] or ""}
    texts.update({rid: r["text"] for rid, r in th["replies"].items()})
    struct_parent = _structure_parents(th["structure"] or {})
    struct_parent[thread_id] = None

    def known_parent(node: str) -> Optional[str]:
        if node in struct_parent:
            return struct_parent[node]
        return th["replies"].get(node, {}).get("in_reply_to")

    parent_of: Dict[str, Optional[str]] = {}
    contracted = 0
    for node in texts:
        if node == thread_id:
            continue
        p, hops = known_parent(node), 0
        while p is not None and p not in texts and hops < 10_000:
            p, hops = known_parent(p), hops + 1
            contracted += 1
        parent_of[node] = p if p in texts else thread_id
    bags = {node: bag_of_words(tokenize_bow(t), index) for node, t in texts.items()}
    tree, reattached = build_tree(thread_id, parent_of, bags, max(N_FEATURES, len(index)))
    return tree, {"contracted_hops": contracted, "reattached_nodes": reattached}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def cache_path(dataset: str, variant: Optional[str] = None) -> Path:
    """``variant="fullvocab"`` is the PHEME cache that keeps every word, for
    per-fold vocabulary selection (see :func:`restrict_vocab`)."""
    suffix = f"_{variant}" if variant else ""
    return CACHE_DIR / f"{dataset}_graphs{suffix}.json.gz"


def write_cache(dataset: str, trees: Mapping[str, TreeGraph], meta: Dict, path: Optional[Path] = None) -> Path:
    path = Path(path or cache_path(dataset))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset": dataset,
        "n_features": N_FEATURES,
        **meta,
        "trees": {tid: t.to_json() for tid, t in trees.items()},
    }
    tmp = path.with_suffix(".tmp")
    # mtime=0 keeps the gzip bytes (and so the sha256) reproducible
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        fh.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    replace_file(tmp, path)
    return path


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_graphs(
    dataset: str,
    tweet_ids: Sequence[str],
    path: Optional[Path] = None,
    variant: Optional[str] = None,
) -> Tuple[List[TreeGraph], Dict]:
    """Trees aligned with ``tweet_ids`` (dataframe row order) plus cache metadata."""
    path = Path(path or cache_path(dataset, variant))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with `python -m src.run_propagation build-graphs`"
        )
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("format_version") != CACHE_FORMAT_VERSION or payload.get("dataset") != dataset:
        raise ValueError(f"{path} is not a v{CACHE_FORMAT_VERSION} cache for {dataset}")
    trees = payload["trees"]
    missing = [t for t in tweet_ids if str(t) not in trees]
    if missing:
        raise ValueError(f"{len(missing)} rows have no tree in {path}, e.g. {missing[:3]}")
    n_features = int(payload["n_features"])
    graphs = [TreeGraph.from_json(trees[str(t)], n_features) for t in tweet_ids]
    meta = {k: v for k, v in payload.items() if k not in ("trees", "vocab")}
    meta["sha256"] = file_sha256(path)
    return graphs, meta


def restrict_vocab(
    graphs: Sequence[TreeGraph],
    train_rows: Sequence[int],
    size: int = N_FEATURES,
    min_df: int = 2,
) -> Tuple[List[TreeGraph], Dict]:
    """Re-index every tree onto the top ``size`` words by node-level document
    frequency over ``train_rows`` only (ties broken by word id); other words
    are dropped.  Used per cross-validation fold so that no validation/test
    thread influences the vocabulary."""
    n_words = 1 + max((int(g.words.max()) for g in graphs if len(g.words)), default=0)
    df = np.zeros(n_words, dtype=np.int64)
    for i in train_rows:
        # a word occurs at most once per node bag, so bincount is document frequency
        df += np.bincount(graphs[i].words, minlength=n_words)
    order = np.lexsort((np.arange(n_words), -df))
    chosen = [int(w) for w in order[:size] if df[w] >= min_df]
    remap = np.full(n_words, -1, dtype=np.int64)
    remap[chosen] = np.arange(len(chosen))
    out = []
    for g in graphs:
        n = g.n_nodes
        node_of_entry = np.repeat(np.arange(n), np.diff(np.append(g.offsets, len(g.words))))
        new = remap[g.words] if len(g.words) else g.words
        keep = new >= 0
        per_node = np.bincount(node_of_entry[keep], minlength=n)
        tree = TreeGraph(
            parent=g.parent,
            offsets=np.concatenate([[0], np.cumsum(per_node)[:-1]]).astype(np.int64),
            words=new[keep].astype(np.int64),
            counts=g.counts[keep],
        )
        tree.validate(size)
        out.append(tree)
    return out, {"vocab_size": len(chosen), "train_threads": len(train_rows), "min_df": min_df}


def graph_stats(graphs: Sequence[TreeGraph], labels: Sequence[int]) -> Dict:
    sizes = np.array([g.n_nodes for g in graphs])
    depths = np.array([int(g.depth().max()) for g in graphs])
    by_label = defaultdict(list)
    for s, y in zip(sizes, labels):
        by_label[int(y)].append(int(s))
    return {
        "threads": len(graphs),
        "nodes_total": int(sizes.sum()),
        "nodes_median": float(np.median(sizes)),
        "nodes_max": int(sizes.max()),
        "single_node_trees": int((sizes == 1).sum()),
        "depth_median": float(np.median(depths)),
        "depth_max": int(depths.max()),
        "empty_bag_nodes": int(
            sum(int(np.sum(np.diff(np.append(g.offsets, len(g.words))) == 0)) for g in graphs)
        ),
        "median_nodes_by_label": {k: float(np.median(v)) for k, v in sorted(by_label.items())},
    }


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
@dataclass
class GraphBatch:
    """Several trees as one disjoint graph (node ids are global)."""

    words: torch.Tensor  # (nnz,) long
    counts: torch.Tensor  # (nnz,) float
    offsets: torch.Tensor  # (N,) long, EmbeddingBag offsets
    src: torch.Tensor  # (E,) long, parent (top-down direction)
    dst: torch.Tensor  # (E,) long, child
    graph_of_node: torch.Tensor  # (N,) long
    roots: torch.Tensor  # (B,) long, global id of each source node
    n_graphs: int

    @property
    def n_nodes(self) -> int:
        return int(self.offsets.numel())

    def to(self, device) -> "GraphBatch":
        return GraphBatch(
            words=self.words.to(device),
            counts=self.counts.to(device),
            offsets=self.offsets.to(device),
            src=self.src.to(device),
            dst=self.dst.to(device),
            graph_of_node=self.graph_of_node.to(device),
            roots=self.roots.to(device),
            n_graphs=self.n_graphs,
        )


def collate_graphs(graphs: Sequence[TreeGraph]) -> GraphBatch:
    import torch  # lazy: building caches needs no torch

    words, counts, offsets, src, dst, gid, roots = [], [], [], [], [], [], []
    node_base, word_base = 0, 0
    for b, g in enumerate(graphs):
        n = g.n_nodes
        roots.append(node_base)
        offsets.append(g.offsets + word_base)
        words.append(g.words)
        counts.append(g.counts)
        child = np.arange(1, n, dtype=np.int64)
        src.append(g.parent[1:] + node_base)
        dst.append(child + node_base)
        gid.append(np.full(n, b, dtype=np.int64))
        node_base += n
        word_base += len(g.words)
    cat = lambda xs, dt: torch.from_numpy(np.concatenate(xs).astype(dt)) if xs else torch.zeros(0)
    return GraphBatch(
        words=cat(words, np.int64),
        counts=cat(counts, np.float32),
        offsets=cat(offsets, np.int64),
        src=cat(src, np.int64),
        dst=cat(dst, np.int64),
        graph_of_node=cat(gid, np.int64),
        roots=torch.tensor(roots, dtype=torch.long),
        n_graphs=len(graphs),
    )
