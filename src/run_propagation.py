"""
CLI for stage 2: propagation-tree (BiGCN) arm and text+graph fusion arm.

Examples
--------
    # once, locally (needs the archives; the output is committed to git)
    python -m src.run_propagation build-graphs --all-datasets \
        --rvnn-dir _archive/rvnn --pheme-tar _archive/PHEME_veracity.tar.bz2

    python -m src.run_propagation stats --all-datasets      # tree stats + shortcut check
    python -m src.run_propagation train --dataset twitter16 --arm graph  --seeds 42 43 44
    python -m src.run_propagation train --dataset twitter16 --arm fusion --seeds 42 43 44 \
        --ckpt-root /content/drive/MyDrive/smrd/checkpoints/stage2

Results: ``<ckpt-root>/<run>/seed<k>/result.json`` and
``results/stage2/<run>_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, Config
from src.data import build_arm_view, load_dataset
from src.dedup import load_or_build_clusters
from src.encoder_preprocess import normalize_tweet
from src.propagation import (
    N_FEATURES,
    RVNN_URL,
    build_vocab,
    cache_path,
    graph_stats,
    load_graphs,
    parse_rvnn,
    pheme_tree,
    read_pheme_threads,
    rvnn_labels,
    tokenize_bow,
    write_cache,
)

LOGGER = logging.getLogger("smrd.propagation")

# Mirrors src.finetune (which imports torch); kept here so `build-graphs` runs
# without torch.  tests/test_propagation.py checks they stay in sync.
TASK_TO_ARM = {"4way": "arm_a", "binary": "arm_b"}
DEFAULT_TASK = {"twitter15": "4way", "twitter16": "4way", "pheme": "binary"}
ARMS = ("fusion", "graph")
RESULTS_DIR = ROOT / "results" / "stage2"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "stage2"


def _view(dataset: str, task: str, df=None):
    return build_arm_view(Config(task_arm=TASK_TO_ARM[task], dataset_name=dataset), df=df)


def _write(path: Path, payload: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
def build_twitter(dataset: str, rvnn_dir: Path) -> Dict:
    df = load_dataset(dataset)
    ids = df["tweet_id"].astype(str).tolist()
    labels = rvnn_labels(rvnn_dir, dataset)
    mismatched = [
        t for t, r in zip(ids, df["raw_label"].astype(str)) if labels.get(t) != r.lower()
    ]
    if mismatched:
        raise ValueError(f"{len(mismatched)} {dataset} labels disagree with RvNN, e.g. {mismatched[:3]}")
    trees, info = parse_rvnn(rvnn_dir, dataset, ids)
    meta = {
        "source": f"RvNN release ({RVNN_URL}data.TD_RvNN.vol_5000.txt)",
        "feature_note": "RvNN bag-of-words; vocabulary built by the dataset authors over all threads (incl. test)",
        "build_info": info,
    }
    write_cache(dataset, trees, meta)
    return {"rows": len(ids), **info}


def build_pheme(tar_path: Path, threads: Optional[Dict] = None) -> Dict:
    df = load_dataset("pheme")
    ids = df["tweet_id"].astype(str).tolist()
    view = _view("pheme", DEFAULT_TASK["pheme"], df)
    threads = threads or read_pheme_threads(tar_path, ids)
    no_source = [t for t in ids if threads[t]["source"] is None]
    if no_source:
        raise ValueError(f"{len(no_source)} PHEME threads without a source tweet, e.g. {no_source[:3]}")
    train_ids = {ids[i] for i in view.train_indices}
    docs = []
    for t in sorted(train_ids):
        th = threads[t]
        docs.append(tokenize_bow(th["source"]))
        docs.extend(tokenize_bow(r["text"]) for r in th["replies"].values())
    vocab = build_vocab(docs, N_FEATURES)
    index = {w: k for k, w in enumerate(vocab)}
    trees, contracted, reattached = {}, 0, 0
    for t in ids:
        trees[t], info = pheme_tree(t, threads[t], index)
        contracted += info["contracted_hops"]
        reattached += info["reattached_nodes"]
    meta = {
        "source": "PHEME all-rnr-annotated-threads (structure.json + reactions)",
        "feature_note": (
            f"bag-of-words over normalize_tweet+lowercase tokens; top {len(vocab)} words by "
            "document frequency (min_df 2) over TRAIN threads of the seed-42 split only"
        ),
        "build_info": {"contracted_hops": contracted, "reattached_nodes": reattached,
                       "vocab_size": len(vocab), "train_threads": len(train_ids)},
        "vocab": vocab,
    }
    write_cache("pheme", trees, meta)
    full = build_pheme_fullvocab(ids, threads)
    return {"rows": len(ids), **meta["build_info"], "fullvocab": full}


def build_pheme_fullvocab(ids: List[str], threads: Dict) -> Dict:
    """Second PHEME cache keeping every word seen in >= 2 tweets (alphabetical
    ids).  Cross-validation picks the top-5000 per fold from training threads
    with ``restrict_vocab``, so no fixed split is baked into the features."""
    docs = []
    for t in ids:
        docs.append(tokenize_bow(threads[t]["source"]))
        docs.extend(tokenize_bow(r["text"]) for r in threads[t]["replies"].values())
    vocab = sorted(build_vocab(docs, size=10**9, min_df=2))
    index = {w: k for k, w in enumerate(vocab)}
    trees = {t: pheme_tree(t, threads[t], index)[0] for t in ids}
    meta = {
        "source": "PHEME all-rnr-annotated-threads (structure.json + reactions)",
        "feature_note": (
            "bag-of-words over ALL words with document frequency >= 2 (no selection); "
            "select a training-only vocabulary with src.propagation.restrict_vocab"
        ),
        "n_features": len(vocab),
        "vocab": vocab,
    }
    path = write_cache("pheme", trees, meta, cache_path("pheme", "fullvocab"))
    return {"vocab_size": len(vocab), "cache": str(path.relative_to(ROOT)),
            "cache_mb": round(path.stat().st_size / 1e6, 2)}


def cmd_build(datasets: List[str], rvnn_dir: Optional[str], pheme_tar: Optional[str]) -> Dict:
    out = {}
    for ds in datasets:
        if ds == "pheme":
            if not pheme_tar:
                raise SystemExit("--pheme-tar is required for pheme")
            out[ds] = build_pheme(Path(pheme_tar))
        else:
            if not rvnn_dir:
                raise SystemExit(f"--rvnn-dir is required for {ds} (files from {RVNN_URL})")
            out[ds] = build_twitter(ds, Path(rvnn_dir))
        out[ds]["cache"] = str(cache_path(ds).relative_to(ROOT))
        out[ds]["cache_mb"] = round(cache_path(ds).stat().st_size / 1e6, 2)
    return out


# ---------------------------------------------------------------------------
def shortcut_check(dataset: str, task: str) -> Dict:
    """How far tree size/shape alone (no text) gets: logistic regression on
    structure features, fit on train, scored on val and test."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.finetune import test_report

    df = load_dataset(dataset)
    view = _view(dataset, task, df)
    graphs, meta = load_graphs(dataset, df["tweet_id"].astype(str).tolist())
    X = np.array([g.structure_features() for g in graphs])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[view.train_indices], view.train_labels)
    clusters = load_or_build_clusters(dataset, df)
    clean = set(clusters.clean_test_indices(view.train_indices, view.val_indices, view.test_indices))
    pred_test = clf.predict(X[view.test_indices]).tolist()
    return {
        "dataset": dataset,
        "task": task,
        "features": ["log_nodes", "max_depth", "mean_depth", "log_root_children",
                     "log_max_children", "leaf_fraction"],
        "graph_stats": graph_stats(graphs, df["label"].astype(int).tolist()),
        "graph_cache": meta,
        "val": test_report(view.val_labels, clf.predict(X[view.val_indices]).tolist(),
                           view.n_classes, view.class_names),
        "test_full": test_report(view.test_labels, pred_test, view.n_classes, view.class_names),
        "test_clean": test_report(
            [view.labels[i] for i in view.test_indices if i in clean],
            [p for i, p in zip(view.test_indices, pred_test) if i in clean],
            view.n_classes, view.class_names,
        ),
    }


def cmd_stats(datasets: List[str]) -> Dict:
    out = {}
    for ds in datasets:
        r = shortcut_check(ds, DEFAULT_TASK[ds])
        _write(RESULTS_DIR / f"{ds}_{r['task']}_structure_shortcut.json", r)
        out[ds] = {
            **{k: r["graph_stats"][k] for k in ("threads", "nodes_median", "nodes_max",
                                               "single_node_trees", "median_nodes_by_label")},
            "structure_only_test_acc": round(r["test_full"]["accuracy"], 4),
            "structure_only_test_macro_f1": round(r["test_full"]["macro_f1"], 4),
        }
    return out


# ---------------------------------------------------------------------------
def cmd_train(args) -> Dict:
    from src.run_encoder import _summarize
    from src.train_propagation import ARM_DEFAULTS, PropagationConfig, run_propagation

    task = args.task or DEFAULT_TASK[args.dataset]
    df = load_dataset(args.dataset)
    view = _view(args.dataset, task, df)
    clusters = load_or_build_clusters(args.dataset, df)
    clean = clusters.clean_test_indices(view.train_indices, view.val_indices, view.test_indices)
    graphs, meta = load_graphs(args.dataset, df["tweet_id"].astype(str).tolist())
    texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]
    defaults = ARM_DEFAULTS[args.arm]

    def pick(name):
        value = getattr(args, name)
        return defaults[name] if value is None else value

    results = []
    for seed in args.seeds:
        cfg = PropagationConfig(
            dataset=args.dataset,
            task=task,
            arm=args.arm,
            encoder_name=args.encoder,
            max_len=args.max_len,
            batch_size=pick("batch_size"),
            micro_batch_size=args.micro_batch_size,
            lr_text=args.lr_text,
            lr_graph=args.lr_graph,
            schedule=pick("schedule"),
            warmup_ratio=pick("warmup_ratio"),
            edge_drop=args.edge_drop,
            label_smoothing=args.label_smoothing,
            class_weights=args.class_weights,
            max_epochs=pick("max_epochs"),
            patience=pick("patience"),
            seed=seed,
            amp=args.amp,
            graph_cache_sha256=meta["sha256"],
        )
        name = cfg.run_name + (f"_{args.tag}" if args.tag else "")
        ckpt_dir = Path(args.ckpt_root) / name / f"seed{seed}"
        if args.fresh and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        LOGGER.info("=== %s seed %d -> %s ===", name, seed, ckpt_dir)
        results.append(
            run_propagation(
                cfg,
                ckpt_dir,
                df=df,
                view=view,
                texts=texts,
                graphs=graphs,
                device=args.device,
                clean_test_indices=clean,
                graph_meta=meta,
                keep_checkpoints=args.keep_checkpoints,
            )
        )
    summary = {"run": name, "arm": args.arm, **_summarize(results)}
    _write(RESULTS_DIR / f"{name}_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 propagation / fusion")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-graphs", help="parse archives into data/graphs/*.json.gz")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--rvnn-dir", help=f"directory with the RvNN files from {RVNN_URL}")
    p.add_argument("--pheme-tar", help="PHEME_veracity.tar.bz2")

    p = sub.add_parser("stats", help="tree statistics + structure-only shortcut check")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all-datasets", action="store_true")

    p = sub.add_parser("train", help="train the graph or fusion arm")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--task", choices=sorted(TASK_TO_ARM))
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--encoder", default="roberta-base")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--max-epochs", type=int, default=None, help="default depends on --arm")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--schedule", choices=["linear", "constant"], default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr-text", type=float, default=2e-5)
    p.add_argument("--lr-graph", type=float, default=5e-4)
    p.add_argument("--edge-drop", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weights", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--amp", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--keep-checkpoints", action="store_true")
    return parser


def _datasets(args) -> List[str]:
    if not args.all_datasets and not args.dataset:
        raise SystemExit("pass --dataset or --all-datasets")
    return list(DATASETS) if args.all_datasets else [args.dataset]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.command == "build-graphs":
        out = cmd_build(_datasets(args), args.rvnn_dir, args.pheme_tar)
    elif args.command == "stats":
        out = cmd_stats(_datasets(args))
    else:
        out = cmd_train(args)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
