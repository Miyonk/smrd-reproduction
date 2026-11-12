"""
CLI for grouped 5-fold cross-validation of the three arms (see ``src/cv.py``).

    python -m src.run_cv make-folds --all-datasets            # once; output is committed
    python -m src.run_cv train --dataset twitter16 --arm text   --ckpt-root <dir>
    python -m src.run_cv train --dataset twitter16 --arm graph  --ckpt-root <dir>
    python -m src.run_cv train --dataset twitter16 --arm fusion --ckpt-root <dir>
    python -m src.run_cv train --dataset twitter16 --arm lexical --ckpt-root <dir>   # CPU
    python -m src.run_cv stack --dataset twitter16 --members lexical graph --ckpt-root <dir>
    python -m src.run_cv train --dataset twitter16 --arm text_long --light --ckpt-root <dir>
    python -m src.run_cv stack --dataset twitter16 --members lexical text_long --tag text --ckpt-root <dir>
    python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>

Arms: ``text`` (RoBERTa on the source tweet), ``graph`` (BiGCN) and ``fusion``
are the stage 1 / stage 2 models with their default settings, unchanged;
``lexical`` is TF-IDF + logistic regression (``src/lexical.py``); ``stack`` is a
validation-weighted average of other arms' probabilities (``src/stack.py``);
``text_long`` is the ``text`` model on a longer schedule (``TEXT_ARMS``) and
``stack_text`` the stack of ``lexical`` and ``text_long``.
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

import numpy as np

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
# Source-text arms: FinetuneConfig overrides on top of the stage 1 defaults.
# Stage 1's 6 epochs are under 200 optimizer steps on a Twitter16 fold;
# ``text_long`` gives the same model up to 30 epochs (best epoch still chosen
# on the validation fold).
TEXT_ARMS = {
    "text": {},
    "text_long": {"max_epochs": 30, "patience": 10},
}
TRAIN_ARMS = CV_ARMS + ("lexical", "text_long")
NEURAL_ARMS = CV_ARMS + ("text_long",)
# Optional arms, reported once started, and the arms each is paired against.
# *_gwo: stage 3 (src/run_tune.py).
OPTIONAL_ARMS = {
    "graph_gwo": ("graph",),
    "fusion_gwo": ("fusion",),
    "lexical": ("graph", "text"),
    "stack": ("lexical", "graph"),
    "text_long": ("lexical", "text"),
    # written by ``stack --members lexical text_long --tag text``
    "stack_text": ("lexical", "text_long"),
}
TUNED_ARMS = ("graph_gwo", "fusion_gwo")
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

    dataset, arm = args.dataset, args.arm
    task = DEFAULT_TASK[dataset]
    df = load_dataset(dataset)
    ids = df["tweet_id"].astype(str).tolist()
    base = _base_view(dataset, task, df)
    folds = Folds.load(dataset, ids)
    folds.check_no_group_leak()
    run_dir = _run_dir(args.ckpt_root, dataset, task, arm, args.tag)

    graphs = meta = bags = None
    texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]
    if args.light and arm not in TEXT_ARMS:
        raise SystemExit("--light applies to the text arms only")
    if arm not in TEXT_ARMS:
        from src.propagation import load_graphs, restrict_vocab

        variant = "fullvocab" if dataset in FOLD_VOCAB_DATASETS else None
        graphs, meta = load_graphs(dataset, ids, variant=variant)
    if arm == "lexical":
        from src.lexical import LexicalConfig, run_lexical, thread_bags

        bags = thread_bags(graphs)  # idf is fitted per fold, on training rows
    elif arm in TEXT_ARMS:
        from src.finetune import FinetuneConfig, run_finetune
    else:
        from src.train_propagation import ARM_DEFAULTS, PropagationConfig, run_propagation

    done = []
    for fold in args.folds:
        view = folds.view(base, fold)
        seed = args.base_seed + fold
        ckpt_dir = run_dir / f"fold{fold}"
        if args.fresh and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        source = folds.split_source(fold)
        LOGGER.info("=== %s fold %d seed %d -> %s ===", run_dir.name, fold, seed, ckpt_dir)
        if arm in TEXT_ARMS:
            cfg = FinetuneConfig(dataset=dataset, task=task, seed=seed, **TEXT_ARMS[arm])
            if args.max_epochs:
                cfg.max_epochs = args.max_epochs
            result = run_finetune(cfg, ckpt_dir, device=args.device, df=df, view=view,
                                  split_source=source, light=args.light)
        elif arm == "lexical":
            cfg = LexicalConfig(dataset=dataset, task=task, graph_cache_sha256=meta["sha256"])
            result = run_lexical(cfg, ckpt_dir, df=df, view=view, texts=texts, bags=bags, split_source=source)
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
            "best_epoch": result["best_epoch"],
        })
    return {"run": run_dir.name, "folds": done}


# ---------------------------------------------------------------------------
def _neural_probs(arm: str, artifact_dir: Path, parts: Dict[str, List[int]], texts, graphs, device) -> Dict:
    """Class probabilities of a trained neural arm for the given rows."""
    import torch

    from src.finetune import _chunks, _tokenize, resolve_device

    dev = resolve_device(device)
    if arm in TEXT_ARMS:
        from src.encoder_model import load_artifact

        model, tokenizer, cfg = load_artifact(artifact_dir, dev)
        max_len = int(cfg["max_len"])

        def forward(rows):
            return model(*_tokenize(tokenizer, [texts[i] for i in rows], max_len, dev))
    else:
        from src.propagation import collate_graphs
        from src.propagation_model import load_artifact

        model, tokenizer, cfg = load_artifact(artifact_dir, dev)
        max_len = int(cfg["model_config"]["max_len"])

        def forward(rows):
            g = collate_graphs([graphs[i] for i in rows]).to(dev)
            if model.text_encoder is None:
                return model(g)
            return model(g, *_tokenize(tokenizer, [texts[i] for i in rows], max_len, dev))

    model.eval()
    out = {}
    with torch.no_grad():
        for name, rows in parts.items():
            probs = [torch.softmax(forward(chunk).float(), dim=-1).cpu() for chunk in _chunks(rows, 64)]
            out[name] = torch.cat(probs).numpy()
    return out


def ensure_probs(ckpt_root, dataset, task, arm, fold, folds, view, texts, graphs, device) -> Path:
    """``probs.json`` (validation + test probabilities) for one member fold;
    neural arms are re-scored from their saved model if the file is missing."""
    from src.lexical import PROBS_FILE, write_probs

    fold_dir = _run_dir(ckpt_root, dataset, task, arm, None) / f"fold{fold}"
    result_path, probs_path = fold_dir / "result.json", fold_dir / PROBS_FILE
    if not result_path.exists():
        raise SystemExit(
            f"{arm} fold {fold} is not trained yet ({result_path} missing); run\n"
            f"  python -m src.run_cv train --dataset {dataset} --arm {arm} --ckpt-root {ckpt_root}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["split"]["source"] != folds.split_source(fold):
        raise RuntimeError(f"{result_path} was produced with a different fold definition")
    if probs_path.exists():
        return probs_path
    if arm not in NEURAL_ARMS:
        raise SystemExit(f"{probs_path} is missing; retrain {arm} with --fresh")
    artifact = fold_dir / "artifact"
    if not artifact.exists():
        raise SystemExit(f"{artifact} is missing, so {arm} fold {fold} cannot be re-scored; retrain it with --fresh")
    fold_graphs = graphs
    if arm not in TEXT_ARMS and dataset in FOLD_VOCAB_DATASETS:
        from src.propagation import restrict_vocab

        fold_graphs, _ = restrict_vocab(graphs, view.train_indices)
    LOGGER.info("scoring %s fold %d from %s", arm, fold, artifact)
    parts = {"val": view.val_indices, "test": view.test_indices}
    probs = _neural_probs(arm, artifact, parts, texts, fold_graphs, device)
    # the saved model must reproduce the test predictions it reported
    saved = {p["row"]: p["pred"] for p in result["test_predictions"]}
    now = dict(zip(view.test_indices, probs["test"].argmax(1).tolist()))
    changed = sum(saved[i] != now.get(i) for i in saved)
    if set(saved) != set(now) or changed > 0.02 * len(saved):
        raise RuntimeError(
            f"{artifact} does not reproduce the test predictions in {result_path} "
            f"({changed} of {len(saved)} differ)"
        )
    if changed:
        LOGGER.warning("%s fold %d: %d of %d test predictions differ from the training run "
                       "(numerical precision)", arm, fold, changed, len(saved))
    write_probs(probs_path, {k: (parts[k], probs[k]) for k in parts})
    return probs_path


def cmd_stack(args) -> Dict:
    from datetime import datetime, timezone

    from src.encoder_preprocess import normalize_tweet
    from src.evaluate import test_report
    from src.stack import choose_weights, combine, read_probs

    dataset, members = args.dataset, list(args.members)
    if len(members) < 2 or len(set(members)) != len(members):
        raise SystemExit("--members needs at least two different arms")
    task = DEFAULT_TASK[dataset]
    df = load_dataset(dataset)
    ids = df["tweet_id"].astype(str).tolist()
    base = _base_view(dataset, task, df)
    folds = Folds.load(dataset, ids)
    run_dir = _run_dir(args.ckpt_root, dataset, task, "stack", args.tag)
    config = {"members": members, "weight_steps": 10, "selection": "validation macro-F1"}
    texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]
    graphs = None
    if any(m in ("graph", "fusion") for m in members):
        from src.propagation import load_graphs

        graphs, _ = load_graphs(dataset, ids, variant="fullvocab" if dataset in FOLD_VOCAB_DATASETS else None)

    done = []
    for fold in args.folds:
        view = folds.view(base, fold)
        fold_dir = run_dir / f"fold{fold}"
        if args.fresh and fold_dir.exists():
            shutil.rmtree(fold_dir)
        result_path = fold_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["config"] != config or result["split"]["source"] != folds.split_source(fold):
                raise RuntimeError(f"{result_path} holds a different stack; use --tag or --fresh")
        else:
            val_p, test_p = [], []
            for m in members:
                path = ensure_probs(args.ckpt_root, dataset, task, m, fold, folds, view, texts, graphs, args.device)
                v, t = read_probs(path, view.val_indices, view.test_indices, base.n_classes)
                val_p.append(v)
                test_p.append(t)
            chosen = choose_weights(val_p, view.val_labels, base.n_classes, config["weight_steps"])
            probs = combine(test_p, chosen["weights"])  # test rows, once, with fixed weights
            preds = probs.argmax(1).tolist()
            y_test = view.test_labels
            result = {
                "config": config,
                "dataset": dataset,
                "task": task,
                "arm": "stack",
                "class_names": base.class_names,
                "split": {"source": folds.split_source(fold), "train": len(view.train_indices),
                          "val": len(view.val_indices), "test": len(view.test_indices)},
                "weights": dict(zip(members, chosen["weights"])),
                "selection": chosen,
                "member_test_accuracy": {
                    m: float((p.argmax(1) == np.array(y_test)).mean()) for m, p in zip(members, test_p)
                },
                "epochs_run": 0,
                "best_epoch": 0,
                "best_val_macro_f1": chosen["val_macro_f1"],
                "test_full": test_report(y_test, preds, base.n_classes, base.class_names),
                "test_predictions": [
                    {"row": int(i), "tweet_id": ids[i], "label": int(y), "pred": int(p), "confidence": float(max(pr))}
                    for i, y, p, pr in zip(view.test_indices, y_test, preds, probs.tolist())
                ],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            _write(result_path, result)
        done.append({
            "fold": fold,
            "weights": result["weights"],
            "member_test_accuracy": {m: round(a, 4) for m, a in result["member_test_accuracy"].items()},
            "test_accuracy": round(result["test_full"]["accuracy"], 4),
            "test_macro_f1": round(result["test_full"]["macro_f1"], 4),
        })
    return {"run": run_dir.name, "folds": done}


# ---------------------------------------------------------------------------
def load_arm_results(
    ckpt_root: str, dataset: str, task: str, arm: str, folds: Folds, tag: Optional[str] = None
) -> Optional[List[Dict]]:
    """All fold results for an arm, or ``None`` if any fold is unfinished."""
    results = []
    for fold in range(folds.n_folds):
        path = _run_dir(ckpt_root, dataset, task, arm, tag) / f"fold{fold}" / "result.json"
        if not path.exists():
            return None
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["split"]["source"] != folds.split_source(fold):
            raise RuntimeError(f"{path} was produced with a different fold definition")
        results.append(r)
    return results


def cmd_compare(dataset: str, ckpt_root: str, tuned_tag: Optional[str] = None) -> Dict:
    from src.evaluate import test_report

    task = DEFAULT_TASK[dataset]
    df = load_dataset(dataset)
    base = _base_view(dataset, task, df)
    folds = Folds.load(dataset, df["tweet_id"].astype(str).tolist())
    expected = sorted(base.labels)

    arms, correct, missing = {}, {}, []
    for arm in CV_ARMS + tuple(OPTIONAL_ARMS):
        tag = tuned_tag if arm in TUNED_ARMS else None
        results = load_arm_results(ckpt_root, dataset, task, arm, folds, tag)
        if results is None:
            # an arm is reported only once it has been started under this root
            if _run_dir(ckpt_root, dataset, task, arm, tag).exists():
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
    for arm, others in OPTIONAL_ARMS.items():
        for other in others:
            if arm in arms and other in arms:
                pairs[f"{arm}_minus_{other}"] = paired_difference(correct[arm], correct[other])
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
            f"{arm:<10} acc {a['pooled']['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]  "
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
    p.add_argument("--arm", choices=TRAIN_ARMS, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=None, help="override, e.g. 1 for a smoke run (use --tag)")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--light", action="store_true",
                   help="text arms: keep only result.json and probs.json (no saved model, no "
                        "per-epoch checkpoint; an interrupted fold starts again)")

    p = sub.add_parser("stack", help="validation-weighted average of trained arms")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--members", nargs="+", default=["lexical", "graph"], choices=TRAIN_ARMS)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--fresh", action="store_true")

    p = sub.add_parser("compare", help="pooled metrics + paired arm comparison")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--tuned-tag", default=None, help="tag of the stage 3 runs to include, if they used one")
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
    elif args.command == "stack":
        if args.members != ["lexical", "graph"] and not args.tag:
            raise SystemExit("a non-default member list needs --tag so it gets its own directory")
        print(json.dumps(cmd_stack(args), indent=2))
    else:
        for ds in _datasets(args):
            print(format_comparison(cmd_compare(ds, args.ckpt_root, args.tuned_tag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
