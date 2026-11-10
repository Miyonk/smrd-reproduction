"""
CLI for grouped 5-fold cross-validation of the three arms (see ``src/cv.py``).

    python -m src.run_cv make-folds --all-datasets            # once; output is committed
    python -m src.run_cv train --dataset twitter16 --arm text   --ckpt-root <dir>
    python -m src.run_cv train --dataset twitter16 --arm graph  --ckpt-root <dir>
    python -m src.run_cv train --dataset twitter16 --arm fusion --ckpt-root <dir>
    python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>

Arms are the stage 1 / stage 2 models with their default settings, unchanged:
``text`` (RoBERTa on the source tweet), ``graph`` (BiGCN), ``fusion``.
One run per fold, seed ``base_seed + fold``.  Fold runs live in
``<ckpt-root>/<dataset>_<task>_<arm>/fold<k>/`` and resume like any other run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, Config
from src.cv import Folds, build_folds_file, folds_path, paired_difference, pool_fold_results
from src.data import build_arm_view, load_dataset

LOGGER = logging.getLogger("smrd.cv")
RESULTS_DIR = ROOT / "results" / "cv"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "cv"
CV_ARMS = ("text", "graph", "fusion")
TASK_TO_ARM = {"4way": "arm_a", "binary": "arm_b"}
DEFAULT_TASK = {"twitter15": "4way", "twitter16": "4way", "pheme": "binary"}
# PHEME node features are re-selected per fold from training threads
FOLD_VOCAB_DATASETS = ("pheme",)


def _base_view(dataset: str, task: str, df):
    return build_arm_view(Config(task_arm=TASK_TO_ARM[task], dataset_name=dataset), df=df)


def _run_dir(ckpt_root: str, dataset: str, task: str, arm: str, tag: Optional[str]) -> Path:
    return Path(ckpt_root) / (f"{dataset}_{task}_{arm}" + (f"_{tag}" if tag else ""))


def _write(path: Path, payload: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
def cmd_make_folds(datasets: List[str]) -> Dict:
    out = {}
    for ds in datasets:
        df = load_dataset(ds)
        payload = build_folds_file(ds, df)
        folds = Folds.load(ds, df["tweet_id"].astype(str).tolist())
        folds.check_no_group_leak()
        out[ds] = {
            "file": str(folds_path(ds).relative_to(ROOT)),
            "fold_sizes": payload["fold_sizes"],
            "label_counts_per_fold": payload["label_counts_per_fold"],
            "groups": len(set(payload["group_of_row"])),
        }
    return out


# ---------------------------------------------------------------------------
def cmd_train(args) -> Dict:
    from src.encoder_preprocess import normalize_tweet
    from src.finetune import FinetuneConfig, run_finetune

    dataset, arm = args.dataset, args.arm
    task = DEFAULT_TASK[dataset]
    df = load_dataset(dataset)
    ids = df["tweet_id"].astype(str).tolist()
    base = _base_view(dataset, task, df)
    folds = Folds.load(dataset, ids)
    folds.check_no_group_leak()
    run_dir = _run_dir(args.ckpt_root, dataset, task, arm, args.tag)

    graphs = meta = None
    if arm != "text":
        from src.propagation import load_graphs, restrict_vocab
        from src.train_propagation import ARM_DEFAULTS, PropagationConfig, run_propagation

        variant = "fullvocab" if dataset in FOLD_VOCAB_DATASETS else None
        graphs, meta = load_graphs(dataset, ids, variant=variant)
        texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]

    done = []
    for fold in args.folds:
        view = folds.view(base, fold)
        seed = args.base_seed + fold
        ckpt_dir = run_dir / f"fold{fold}"
        if args.fresh and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        source = folds.split_source(fold)
        LOGGER.info("=== %s fold %d seed %d -> %s ===", run_dir.name, fold, seed, ckpt_dir)
        if arm == "text":
            cfg = FinetuneConfig(dataset=dataset, task=task, seed=seed)
            if args.max_epochs:
                cfg.max_epochs = args.max_epochs
            result = run_finetune(cfg, ckpt_dir, device=args.device, df=df, view=view, split_source=source)
        else:
            fold_graphs, fold_meta = graphs, dict(meta)
            if dataset in FOLD_VOCAB_DATASETS:
                fold_graphs, info = restrict_vocab(graphs, view.train_indices)
                fold_meta["fold_vocab"] = info
            params = dict(ARM_DEFAULTS[arm])
            if args.max_epochs:
                params["max_epochs"] = args.max_epochs
            cfg = PropagationConfig(
                dataset=dataset, task=task, arm=arm, seed=seed,
                graph_cache_sha256=meta["sha256"], **params,
            )
            result = run_propagation(
                cfg, ckpt_dir, df=df, view=view, texts=texts, graphs=fold_graphs,
                device=args.device, graph_meta=fold_meta, split_source=source,
            )
        if result["split"]["source"] != source:
            raise RuntimeError(
                f"{ckpt_dir} holds a result for a different fold definition "
                f"({result['split']['source']!r}); rerun with --fresh"
            )
        done.append({
            "fold": fold,
            "seed": seed,
            "test_n": result["test_full"]["n"],
            "test_accuracy": round(result["test_full"]["accuracy"], 4),
            "test_macro_f1": round(result["test_full"]["macro_f1"], 4),
            "best_val_macro_f1": round(result["best_val_macro_f1"], 4),
            "epochs_run": result["epochs_run"],
        })
    return {"run": run_dir.name, "folds": done}


# ---------------------------------------------------------------------------
def load_arm_results(ckpt_root: str, dataset: str, task: str, arm: str, folds: Folds) -> Optional[List[Dict]]:
    """All fold results for an arm, or ``None`` if any fold is unfinished."""
    results = []
    for fold in range(folds.n_folds):
        path = _run_dir(ckpt_root, dataset, task, arm, None) / f"fold{fold}" / "result.json"
        if not path.exists():
            return None
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["split"]["source"] != folds.split_source(fold):
            raise RuntimeError(f"{path} was produced with a different fold definition")
        results.append(r)
    return results


def cmd_compare(dataset: str, ckpt_root: str) -> Dict:
    from src.finetune import test_report

    task = DEFAULT_TASK[dataset]
    df = load_dataset(dataset)
    base = _base_view(dataset, task, df)
    folds = Folds.load(dataset, df["tweet_id"].astype(str).tolist())
    expected = sorted(base.labels)

    arms, correct, missing = {}, {}, []
    for arm in CV_ARMS:
        results = load_arm_results(ckpt_root, dataset, task, arm, folds)
        if results is None:
            missing.append(arm)
            continue
        rows, labels, preds = pool_fold_results(results, expected)
        pooled = test_report(labels, preds, base.n_classes, base.class_names)
        fold_acc = [r["test_full"]["accuracy"] for r in results]
        arms[arm] = {
            "pooled": pooled,
            "fold_accuracy": fold_acc,
            "fold_accuracy_mean": statistics.fmean(fold_acc),
            "fold_accuracy_std": statistics.stdev(fold_acc),
            "fold_macro_f1": [r["test_full"]["macro_f1"] for r in results],
            "best_val_macro_f1": [r["best_val_macro_f1"] for r in results],
            "best_epoch": [r["best_epoch"] for r in results],
        }
        correct[arm] = [y == p for y, p in zip(labels, preds)]

    pairs = {
        f"{a}_minus_{b}": paired_difference(correct[a], correct[b])
        for a, b in itertools.combinations(reversed([x for x in CV_ARMS if x in arms]), 2)
    }
    majority = max(set(base.labels.values()), key=list(base.labels.values()).count)
    out = {
        "dataset": dataset,
        "task": task,
        "protocol": "grouped stratified 5-fold; pooled test predictions; one seed per fold",
        "folds_sha256": folds.sha256,
        "n": len(expected),
        "majority_class_accuracy": list(base.labels.values()).count(majority) / len(expected),
        "arms": arms,
        "pairs": pairs,
        "unfinished_arms": missing,
    }
    _write(RESULTS_DIR / f"{dataset}_{task}_comparison.json", out)
    return out


def format_comparison(out: Dict) -> str:
    lines = [f"--- {out['dataset']} ({out['task']}, n={out['n']}, "
             f"majority class {out['majority_class_accuracy']:.3f})"]
    for arm, a in out["arms"].items():
        lo, hi = a["pooled"]["accuracy_wilson_95"]
        lines.append(
            f"{arm:<7} acc {a['pooled']['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]  "
            f"macroF1 {a['pooled']['macro_f1']:.3f}  per-fold acc {a['fold_accuracy_mean']:.3f}"
            f"±{a['fold_accuracy_std']:.3f}"
        )
    for name, p in out["pairs"].items():
        a, b = name.split("_minus_")
        lo, hi = p["difference_ci95"]
        lines.append(
            f"{a} - {b}: {100 * p['accuracy_difference']:+.1f} pts "
            f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  only {a} right {p['only_a_correct']}, "
            f"only {b} right {p['only_b_correct']}, McNemar p={p['mcnemar_p']:.4f}"
        )
    if out["unfinished_arms"]:
        lines.append(f"unfinished arms: {', '.join(out['unfinished_arms'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grouped 5-fold cross-validation")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-folds", help="write data/splits/<dataset>_cv5_grouped.json")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all-datasets", action="store_true")

    p = sub.add_parser("train", help="train one arm on the requested folds")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--arm", choices=CV_ARMS, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=None, help="override, e.g. 1 for a smoke run (use --tag)")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--fresh", action="store_true")

    p = sub.add_parser("compare", help="pooled metrics + paired arm comparison")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
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
    if args.command == "make-folds":
        print(json.dumps(cmd_make_folds(_datasets(args)), indent=2))
    elif args.command == "train":
        if args.max_epochs and not args.tag:
            raise SystemExit("--max-epochs changes the protocol; pass --tag so it gets its own directory")
        print(json.dumps(cmd_train(args), indent=2))
    else:
        for ds in _datasets(args):
            print(format_comparison(cmd_compare(ds, args.ckpt_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
