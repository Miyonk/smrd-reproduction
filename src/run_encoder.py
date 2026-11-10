"""
CLI for the fine-tuned encoder path (stage 1: source-text-only baseline).

Examples
--------
    python -m src.run_encoder dedup --all-datasets
    python -m src.run_encoder baseline --dataset twitter16
    python -m src.run_encoder train --dataset twitter16 --max-epochs 1 --seeds 42
    python -m src.run_encoder train --dataset pheme --seeds 42 43 44 \
        --ckpt-root /content/drive/MyDrive/smrd/checkpoints/encoder

The task defaults to 4-way on Twitter15/16 and rumour-vs-non-rumour on PHEME.
Results go to ``<ckpt-root>/<run>/seed<k>/result.json`` and a per-run summary
``results/stage1/<run>_summary.json`` (mean/std over seeds).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, Config
from src.data import build_arm_view, load_dataset
from src.dedup import DEFAULT_THRESHOLD, clusters_path, load_or_build_clusters
from src.finetune import (
    DEFAULT_TASK,
    TASK_TO_ARM,
    FinetuneConfig,
    run_finetune,
    test_report,
)

LOGGER = logging.getLogger("smrd.encoder")
RESULTS_DIR = ROOT / "results" / "stage1"
DEFAULT_CKPT_ROOT = ROOT / "checkpoints" / "encoder"


def _view(dataset: str, task: str, df=None):
    return build_arm_view(Config(task_arm=TASK_TO_ARM[task], dataset_name=dataset), df=df)


def _write(path: Path, payload: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
def cmd_dedup(datasets: List[str], threshold: float, rebuild: bool) -> Dict:
    report = {}
    for dataset in datasets:
        df = load_dataset(dataset)
        if rebuild and clusters_path(dataset).exists():
            clusters_path(dataset).unlink()
        clusters = load_or_build_clusters(dataset, df, threshold=threshold)
        payload = clusters.to_json(df["label"].astype(int).tolist())
        view = _view(dataset, DEFAULT_TASK[dataset], df)
        leak = clusters.leakage(view.train_indices, view.val_indices, view.test_indices)
        entry = {
            "file": str(clusters_path(dataset).relative_to(ROOT)),
            "rows": payload["n_rows"],
            "clusters": payload["n_clusters"],
            "multi_row_clusters": payload["n_multi_row_clusters"],
            "rows_in_multi_row_clusters": payload["rows_in_multi_row_clusters"],
            "mixed_label_clusters": len(payload["mixed_label_clusters"]),
            "seed42_split_leakage": leak,
        }
        if "event" in df.columns and df["event"].nunique() > 1:
            entry["clusters_spanning_events"] = len(
                clusters.cross_group_clusters(df["event"].astype(str).tolist())
            )
        # A few example pairs so the threshold can be eyeballed.
        texts = df["raw_text"].astype(str).tolist()
        entry["examples"] = [
            [texts[r][:120] for r in rows[:2]]
            for rows in sorted(payload["multi_row_clusters"], key=len, reverse=True)[:3]
        ]
        report[dataset] = entry
    _write(RESULTS_DIR / "dedup_report.json", report)
    return report


def cmd_baseline(dataset: str, task: str) -> Dict:
    df = load_dataset(dataset)
    view = _view(dataset, task, df)
    clusters = load_or_build_clusters(dataset, df)
    clean = clusters.clean_test_indices(view.train_indices, view.val_indices, view.test_indices)
    majority = Counter(view.train_labels).most_common(1)[0][0]
    clean_set = set(clean)
    out = {
        "dataset": dataset,
        "task": task,
        "baseline": "majority_class_of_train",
        "majority_class": view.class_names[majority],
        "test_full": test_report(
            view.test_labels, [majority] * len(view.test_indices), view.n_classes, view.class_names
        ),
        "test_clean": test_report(
            [view.labels[i] for i in view.test_indices if i in clean_set],
            [majority] * len(clean),
            view.n_classes,
            view.class_names,
        ),
    }
    _write(RESULTS_DIR / f"{dataset}_{task}_majority.json", out)
    return out


def _summarize(results: List[Dict]) -> Dict:
    def stats(values: List[float]) -> Dict:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }

    summary = {"n_seeds": len(results), "seeds": [r["config"]["seed"] for r in results]}
    for part in ("test_full", "test_clean"):
        if all(r.get(part) for r in results):
            summary[part] = {
                "n": results[0][part]["n"],
                "accuracy": stats([r[part]["accuracy"] for r in results]),
                "macro_f1": stats([r[part]["macro_f1"] for r in results]),
            }
    summary["best_val_macro_f1"] = stats([r["best_val_macro_f1"] for r in results])
    summary["epoch_seconds_mean"] = statistics.fmean(
        h["epoch_seconds"] for r in results for h in r["history"]
    ) if any(r["history"] for r in results) else None
    return summary


def cmd_train(args) -> Dict:
    task = args.task or DEFAULT_TASK[args.dataset]
    df = load_dataset(args.dataset)
    view = _view(args.dataset, task, df)
    clusters = load_or_build_clusters(args.dataset, df)
    clean = clusters.clean_test_indices(view.train_indices, view.val_indices, view.test_indices)

    results = []
    for seed in args.seeds:
        cfg = FinetuneConfig(
            dataset=args.dataset,
            task=task,
            encoder_name=args.encoder,
            max_len=args.max_len,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            lr=args.lr,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            class_weights=args.class_weights,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=seed,
            amp=args.amp,
        )
        ckpt_dir = Path(args.ckpt_root) / cfg.run_name / f"seed{seed}"
        if args.tag:
            ckpt_dir = Path(args.ckpt_root) / f"{cfg.run_name}_{args.tag}" / f"seed{seed}"
        if args.fresh and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        LOGGER.info("=== %s seed %d -> %s ===", cfg.run_name, seed, ckpt_dir)
        results.append(
            run_finetune(
                cfg,
                ckpt_dir,
                device=args.device,
                clean_test_indices=clean,
                df=df,
                view=view,
                keep_checkpoints=args.keep_checkpoints,
            )
        )

    name = FinetuneConfig(dataset=args.dataset, task=task, encoder_name=args.encoder).run_name
    if args.tag:
        name = f"{name}_{args.tag}"
    summary = {"run": name, **_summarize(results)}
    _write(RESULTS_DIR / f"{name}_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 encoder baseline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dedup", help="build near-duplicate clusters + leakage report")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--rebuild", action="store_true")

    p = sub.add_parser("baseline", help="majority-class baseline")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--task", choices=sorted(TASK_TO_ARM))

    p = sub.add_parser("train", help="fine-tune the source-text classifier")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--task", choices=sorted(TASK_TO_ARM))
    p.add_argument("--encoder", default="roberta-base")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--max-epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weights", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--amp", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--tag", default=None, help="suffix for the run directory, e.g. smoke")
    p.add_argument("--fresh", action="store_true", help="delete this run's checkpoints first")
    p.add_argument("--keep-checkpoints", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.command == "dedup":
        if not args.all_datasets and not args.dataset:
            raise SystemExit("pass --dataset or --all-datasets")
        datasets = list(DATASETS) if args.all_datasets else [args.dataset]
        out = cmd_dedup(datasets, args.threshold, args.rebuild)
    elif args.command == "baseline":
        out = cmd_baseline(args.dataset, args.task or DEFAULT_TASK[args.dataset])
    else:
        out = cmd_train(args)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
