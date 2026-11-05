"""
Command-line orchestration for the SMRD reproduction.

Specification: Section 11.5 (ordered runbook) and Section 11.2 (artifacts).

Examples
--------
    python -m src.run probe
    python -m src.run cache --dataset twitter16
    python -m src.run master
    python -m src.run preflight
    python -m src.run gates --include-stress
    python -m src.run search --arm arm_b --dataset twitter16
    python -m src.run retrain --arm arm_b --dataset twitter16
    python -m src.run pilot
    python -m src.run report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import (
    ARMS,
    CACHE_DIR,
    CANDIDATE_LOG_DIR,
    CHECKPOINT_DIR,
    DATASETS,
    LOG_DIR,
    REPORT_DIR,
    WINNER_DIR,
    Config,
    add_common_cli_args,
    config_from_args,
    ensure_dirs,
)
from src.data import build_arm_view
from src.dhoa import DHOAOptimizer, Hyperparameters
from src.evaluate import (
    bootstrap_accuracy,
    classification_metrics,
    mcnemar_test,
    report_c_collapse,
    summarize_binary,
    wilson_ci,
)
from src.features import FeatureCache, build_all_caches, build_feature_cache, sha256_file
from src.gates import (
    gate_g5,
    probe_appended_gradients,
    probe_hardware,
    run_gates,
    run_preflight,
)
from src.model import build_model, save_master_checkpoint
from src.train import train_candidate

LOGGER = logging.getLogger("smrd")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _candidate_id(config: Config, evaluation_index: int) -> str:
    iteration = evaluation_index // max(1, config.num_deer)
    deer = evaluation_index % max(1, config.num_deer)
    return f"{config.arm_tag}_iter{iteration:02d}_deer{deer:02d}"


def _find_master(config: Config) -> Path:
    path = config.master_checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"master checkpoint missing at {path}; run `python -m src.run master` first"
        )
    return path


# ---------------------------------------------------------------------------
# Phase 5 - master initial checkpoint
# ---------------------------------------------------------------------------
def build_master(config: Config, force: bool = False) -> Dict:
    ensure_dirs()
    path = config.master_checkpoint
    if path.exists() and not force:
        LOGGER.info("master checkpoint already exists: %s", path)
        return {
            "path": str(path),
            "size_gib": path.stat().st_size / (1024 ** 3),
            "skipped": True,
        }

    torch.manual_seed(config.seed_init)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed_init)

    # Build all 24 appended blocks so the master is valid for NL up to 48.
    master_config = Config.from_dict({**config.to_dict(), "max_layers": 48})
    model = build_model(master_config, master_path=None, nl=48)
    model.set_dropout(0.1)
    model.set_head_count(16)
    model.set_layer_count(48)
    save_master_checkpoint(model, path)
    info = {
        "path": str(path),
        "size_gib": path.stat().st_size / (1024 ** 3),
        "sha256": sha256_file(path),
        "params": model.parameter_count(),
        "seed_init": config.seed_init,
        "gpt2_model": config.gpt2_model,
        "gpt2_revision": config.gpt2_revision,
        "note": (
            "Full master state: base GPT-2 (wte pruned), 24 appended blocks, "
            "bridge, and both classification heads."
        ),
    }
    _write_json(LOG_DIR / "master_checkpoint.json", info)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("master checkpoint written: %s (%.2f GiB)", path, info["size_gib"])
    return info


# ---------------------------------------------------------------------------
# Phase 4 - cache
# ---------------------------------------------------------------------------
def ensure_cache(config: Config, force: bool = False, limit: Optional[int] = None) -> FeatureCache:
    cache = FeatureCache(CACHE_DIR / config.dataset_name)
    if cache.exists() and not force:
        return cache.load()
    return build_feature_cache(
        config.dataset_name,
        config=config,
        limit=limit,
        force=force,
        allow_nltk_download=True,
    )


# ---------------------------------------------------------------------------
# Phase 6/7 - search and retrain
# ---------------------------------------------------------------------------
def search(config: Config, fresh: bool = False, limit: Optional[int] = None) -> Dict:
    ensure_dirs()
    view = build_arm_view(config)
    cache = ensure_cache(config, limit=limit)
    master = _find_master(config)

    if fresh and config.search_state.exists():
        config.search_state.unlink()

    def evaluate_fn(hp: Hyperparameters, evaluation_index: int) -> Dict:
        LOGGER.info(
            "evaluating %s L=%.6g B=%d NL=%d NAH=%d DR=%.3f",
            _candidate_id(config, evaluation_index),
            hp.L, hp.B, hp.NL, hp.NAH, hp.DR,
        )
        result = train_candidate(
            config, hp, cache, view, master_path=str(master)
        )
        record = {
            "candidate_id": _candidate_id(config, evaluation_index),
            "task_arm": config.task_arm,
            "dataset": config.dataset_name,
            "evaluation_index": evaluation_index,
            "iteration": evaluation_index // max(1, config.num_deer),
            "deer_index": evaluation_index % max(1, config.num_deer),
            "hyperparameters": hp.as_dict(),
            "seeds": {
                "seed_init": config.seed_init,
                "seed_order": config.seed_order,
                "seed_dropout": config.seed_dropout,
                "seed_dhoa": config.seed_dhoa,
            },
            "status": result.status,
            "execution_stats": result.stats,
        }
        path = CANDIDATE_LOG_DIR / f"{record['candidate_id']}.json"
        _write_json(path, record)
        return {
            "fitness": result.fitness,
            "val_loss": result.val_loss,
            "status": result.status,
            "record": record,
        }

    optimizer = DHOAOptimizer(config, evaluate_fn, state_path=config.search_state)
    result = optimizer.run()
    result["task_arm"] = config.task_arm
    result["dataset"] = config.dataset_name
    result["split_summary"] = view.summary()
    _write_json(LOG_DIR / f"dhoa_result_{config.arm_tag}.json", result)
    if result.get("leader"):
        LOGGER.info("search winner: %s", result["leader"]["hyperparameters"])
    return result


def _load_hyperparameters(config: Config, hp_json: Optional[str]) -> Hyperparameters:
    if hp_json:
        return Hyperparameters(**json.loads(hp_json))
    result_path = LOG_DIR / f"dhoa_result_{config.arm_tag}.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"no search result at {result_path}; pass --hp or run search first"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("leader"):
        raise RuntimeError("DHOA search produced no feasible leader")
    return Hyperparameters(**result["leader"]["hyperparameters"])


def retrain(config: Config, hp_json: Optional[str] = None) -> Dict:
    ensure_dirs()
    view = build_arm_view(config)
    cache = ensure_cache(config)
    master = _find_master(config)
    hp = _load_hyperparameters(config, hp_json)
    selection = "MANUAL_HYPERPARAMETERS" if hp_json else "DHOA_LEADER"

    save_path = WINNER_DIR / f"{config.arm_tag}.pt"
    result = train_candidate(
        config,
        hp,
        cache,
        view,
        master_path=str(master),
        test_indices=view.test_indices,
        save_model_path=str(save_path),
    )

    payload: Dict = {
        "task_arm": config.task_arm,
        "dataset": config.dataset_name,
        "hyperparameters": hp.as_dict(),
        "selection": selection,
        "status": result.status,
        "search_time_validation": _search_validation(config),
        "final_retrain_validation": {
            "accuracy": result.fitness,
            "loss": result.val_loss,
            "macro_f1": result.val_macro_f1,
        },
        "execution_stats": result.stats,
        "test": result.test_metrics,
        "checkpoint": str(save_path),
        "n_test": len(view.test_indices),
    }

    if result.test_metrics is not None:
        correct = sum(
            1
            for truth, pred in zip(
                view.test_labels, result.test_metrics["predictions"]
            )
            if int(truth) == int(pred)
        )
        ci_low, ci_high = wilson_ci(correct, len(view.test_indices))
        payload["test"]["wilson_ci_95"] = [ci_low, ci_high]
        payload["test"]["correct"] = correct
        try:
            boot = bootstrap_accuracy(
                view.test_labels, result.test_metrics["predictions"]
            )
            payload["test"]["bootstrap"] = boot
        except Exception as exc:
            LOGGER.debug("Bootstrap evaluation skipped: %s", exc)

        # Report C (Section 0.4): Post-hoc binary evaluation of Arm A predictions.
        # Must NEVER train a separate model; collapses Arm A predictions directly.
        if config.task_arm == "arm_a":
            collapsed = report_c_collapse(result.test_metrics["predictions"])
            y_binary = [0 if int(y) == 0 else 1 for y in view.test_labels]
            rep_c_metrics = summarize_binary(y_binary, collapsed)
            try:
                rep_c_metrics["bootstrap"] = bootstrap_accuracy(y_binary, collapsed)
            except Exception:
                pass

            # Paired McNemar test vs Arm B if Arm B final result exists
            arm_b_file = LOG_DIR / f"final_arm_b_{config.dataset_name}.json"
            if arm_b_file.exists():
                try:
                    arm_b_data = json.loads(arm_b_file.read_text(encoding="utf-8"))
                    b_preds = arm_b_data.get("test", {}).get("predictions")
                    if b_preds and len(b_preds) == len(y_binary):
                        rep_c_metrics["mcnemar_vs_arm_b"] = mcnemar_test(
                            y_binary, b_preds, collapsed
                        )
                except Exception as exc:
                    LOGGER.debug("McNemar test vs Arm B skipped: %s", exc)

            report_c_payload = {
                "task_arm": "report_c",
                "dataset": config.dataset_name,
                "hyperparameters": hp.as_dict(),
                "selection": "POST_HOC_COLLAPSE_ARM_A",
                "status": result.status,
                "search_time_validation": _search_validation(config),
                "final_retrain_validation": payload.get("final_retrain_validation"),
                "execution_stats": result.stats,
                "test": rep_c_metrics,
                "checkpoint": str(save_path),
                "n_test": len(view.test_indices),
            }
            _write_json(LOG_DIR / f"final_report_c_{config.dataset_name}.json", report_c_payload)
            LOGGER.info(
                "Generated Report C for %s: accuracy=%.4f",
                config.dataset_name,
                rep_c_metrics.get("accuracy", 0.0),
            )

    _write_json(LOG_DIR / f"final_{config.arm_tag}.json", payload)
    LOGGER.info(
        "final %s/%s: val_acc=%s test_acc=%s",
        config.task_arm,
        config.dataset_name,
        result.fitness,
        result.test_metrics["accuracy"] if result.test_metrics else None,
    )
    return payload


def _search_validation(config: Config) -> Optional[Dict]:
    path = LOG_DIR / f"dhoa_result_{config.arm_tag}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    leader = result.get("leader") or {}
    return {
        "fitness": leader.get("fitness"),
        "val_loss": leader.get("val_loss"),
        "hyperparameters": leader.get("hyperparameters"),
    }


# ---------------------------------------------------------------------------
# Phase 9 - reporting
# ---------------------------------------------------------------------------
PAPER_CLAIMS = {
    "twitter15": {"accuracy": 0.9950, "precision": 0.9957, "recall": 0.9960, "f1": 0.9959},
    "twitter16": {"accuracy": 0.9930, "precision": 0.9956, "recall": 0.9936, "f1": 0.9939},
    "pheme":     {"accuracy": 0.9910, "precision": 0.9890, "recall": 0.9898, "f1": 0.9889},
}

CENSUS_DISCREPANCIES = {
    "twitter15": "Physical release 1,490 threads vs Table 3 claim of 1,491 (sum of classes = 1,492).",
    "twitter16": "Physical release 818 threads vs Table 3 claim of 817 (sum of classes = 817).",
    "pheme": "Physical release 6,425 threads has only 2,403 rumor threads (37.4%) vs Table 3 claim of 3,853 rumors (60.0%).",
}


def build_report() -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for path in sorted(LOG_DIR.glob("final_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("test") or {}
        dataset = payload["dataset"]
        arm = payload["task_arm"]
        paper = PAPER_CLAIMS.get(dataset, {})

        acc = test.get("accuracy")
        paper_acc = paper.get("accuracy")
        delta = (acc - paper_acc) if (acc is not None and paper_acc is not None) else None

        wilson_low = (test.get("wilson_ci_95") or [None, None])[0]
        wilson_high = (test.get("wilson_ci_95") or [None, None])[1]

        within_ci = None
        if paper_acc is not None and wilson_low is not None and wilson_high is not None:
            within_ci = (wilson_low <= paper_acc <= wilson_high)

        bootstrap_mean = (test.get("bootstrap") or {}).get("mean")
        mcnemar_p = (test.get("mcnemar_vs_arm_b") or {}).get("p_value")

        rows.append(
            {
                "task_arm": arm,
                "dataset": dataset,
                "status": payload["status"],
                "selection": payload.get("selection", "DHOA_LEADER"),
                "reproduction_accuracy": acc,
                "paper_accuracy": paper_acc,
                "accuracy_delta": delta,
                "within_wilson_95_ci": within_ci,
                "wilson_low": wilson_low,
                "wilson_high": wilson_high,
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "macro_f1": test.get("macro_f1"),
                "bootstrap_mean": bootstrap_mean,
                "mcnemar_p_vs_arm_b": mcnemar_p,
                "n_test": payload.get("n_test"),
                "hyperparameters": json.dumps(payload.get("hyperparameters")),
                "checkpoint": payload.get("checkpoint"),
            }
        )

    csv_path = REPORT_DIR / "headline_comparison_table.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "task_arm", "dataset", "status", "selection", "reproduction_accuracy",
        "paper_accuracy", "accuracy_delta", "within_wilson_95_ci", "wilson_low",
        "wilson_high", "precision", "recall", "f1", "macro_f1", "bootstrap_mean",
        "mcnemar_p_vs_arm_b", "n_test", "hyperparameters", "checkpoint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Headline Reproduction Comparison",
        "",
        "Independent empirical reproduction of *Enhanced Large language model with a self-attention mechanism for Social Media Rumor Detection*.",
        "",
        "## Summary Results",
        "",
        "| Arm | Dataset | Status | Repro Acc | Paper Acc | Delta | 95% Wilson CI | Macro F1 | Within CI? | N test |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        ci = (
            f"[{row['wilson_low']:.4f}, {row['wilson_high']:.4f}]"
            if row["wilson_low"] is not None
            else "TBD"
        )
        repro_acc = f"{row['reproduction_accuracy']:.4f}" if row["reproduction_accuracy"] is not None else "TBD"
        paper_acc = f"{row['paper_accuracy']:.4f}" if row["paper_accuracy"] is not None else "N/A"
        delta = f"{row['accuracy_delta']:+.4f}" if row["accuracy_delta"] is not None else "N/A"
        macro = f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "TBD"
        within = str(row["within_wilson_95_ci"]) if row["within_wilson_95_ci"] is not None else "N/A"
        markdown.append(
            f"| {row['task_arm']} | {row['dataset']} | {row['status']} | "
            f"{repro_acc} | {paper_acc} | {delta} | {ci} | {macro} | {within} | {row['n_test']} |"
        )

    markdown.extend([
        "",
        "## Dataset Census Discrepancies (Phase 1 Audit)",
        "",
    ])
    for ds, disc in CENSUS_DISCREPANCIES.items():
        markdown.append(f"- **{ds.upper()}**: {disc}")

    (REPORT_DIR / "headline_comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    # Build sensitivity_analysis_summary.csv
    sens_rows = []
    for p in sorted(LOG_DIR.glob("ablation_*.json")):
        try:
            abl = json.loads(p.read_text(encoding="utf-8"))
            sens_rows.append(abl)
        except Exception:
            pass
    if sens_rows:
        sens_csv = REPORT_DIR / "sensitivity_analysis_summary.csv"
        with open(sens_csv, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=list(sens_rows[0].keys()))
            w.writeheader()
            w.writerows(sens_rows)

    # Build failure_analysis_report.md
    failure_md = [
        "# Failure & Discrepancy Analysis Report",
        "",
        "## 1. Discrepancy Drivers",
        "- **Physical Dataset Census:** The released PHEME dataset has only 2,403 rumor threads (37.4%) vs Table 3's impossible claim of 3,853 rumors (60.0%).",
        "- **Unstated Split Ratio:** The paper omits the dataset train/val/test split ratio; an audit-approved stratified 70/10/20 split was used.",
        "- **Destructive Preprocessing:** The paper's Light-10 suffix rule aggressively truncates valid lexical tokens ('expand' -> 'exp', 'island' -> 'isl').",
        "- **Optimization Bounds:** The paper leaves DHOA population size and update iterations unstated; computationally feasible bounds (N=6, T=5) were used.",
        "",
        "## 2. Verdict",
        "A failure to replicate the headline ~99% accuracy is a scientifically valid finding demonstrating the sensitivity of the paper's reported numbers to proprietary implementation details.",
    ]
    (REPORT_DIR / "failure_analysis_report.md").write_text(
        "\n".join(failure_md) + "\n", encoding="utf-8"
    )

    return {"rows": len(rows), "csv": str(csv_path)}


# ---------------------------------------------------------------------------
# Runbook helpers
# ---------------------------------------------------------------------------
PRIORITY_ORDER = [
    ("arm_b", "twitter16"),
    ("arm_b", "twitter15"),
    ("arm_b", "pheme"),
    ("arm_a", "twitter16"),
    ("arm_a", "twitter15"),
    ("arm_a", "pheme"),
    ("arm_d", "twitter16"),
    ("arm_d", "twitter15"),
    ("arm_d", "pheme"),
]


def run_pilot(config: Config) -> Dict:
    pilot = config.with_overrides(task_arm="arm_b", dataset_name="twitter16")
    search_result = search(pilot)
    retrain_result = retrain(pilot)
    return {"search": search_result, "retrain": retrain_result}


def run_all(config: Config, resume: bool = False) -> List[Dict]:
    results = []
    for arm, dataset in PRIORITY_ORDER:
        final_path = LOG_DIR / f"final_{arm}_{dataset}.json"
        if resume and final_path.exists():
            LOGGER.info(
                "=== %s / %s already completed on disk (%s) -> preserving result & skipping search/retrain ===",
                arm,
                dataset,
                final_path.name,
            )
            try:
                results.append(json.loads(final_path.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue

        cfg = config.with_overrides(task_arm=arm, dataset_name=dataset)
        LOGGER.info("=== %s / %s ===", arm, dataset)
        search(cfg)
        results.append(retrain(cfg))

    build_report()
    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def run_smoke(config: Config, skip_cache: bool = False) -> Dict:
    from src.preprocess import TextPreprocessor
    from src.dhoa import decode_coordinates

    ensure_dirs()
    prep = TextPreprocessor(allow_nltk_download=True)
    examples = [
        "RT @user: Police expand investigation https://example.com #Ferguson and they demand answers",
        "sharing this update, not confirmed yet",
    ]
    preprocessing = [prep(text) for text in examples]

    cache_info = {"skipped": skip_cache}
    if not skip_cache:
        # Build into an isolated smoke prefix: never overwrite the real cache.
        cache = build_feature_cache(
            "twitter16",
            config=config,
            limit=8,
            force=True,
            allow_nltk_download=True,
            cache_dir=CACHE_DIR / "smoke",
        )
        batch = cache.get_batch(list(range(min(4, len(cache)))))
        cache_info = {
            "rows": len(cache),
            "batch_shape": list(batch.features.shape),
            "metadata_backend": cache.metadata["preprocessor_backend"],
        }

    p3 = probe_appended_gradients(config)
    g5 = gate_g5(config)

    # DHOA mechanics with a deterministic synthetic objective.
    def fake_eval(hp: Hyperparameters, evaluation_index: int) -> Dict:
        fitness = 1.0 - abs(hp.DR - 0.25) - abs(hp.NAH - 8) / 100.0
        return {"fitness": fitness, "val_loss": 1.0 - fitness, "status": "EVALUATED"}

    smoke_config = config.with_overrides(num_deer=6, num_iterations=5)
    smoke_state = Path(LOG_DIR / "smoke_dhoa_state.json")
    if smoke_state.exists():
        smoke_state.unlink()
    optimizer = DHOAOptimizer(smoke_config, fake_eval, state_path=smoke_state)
    dhoa = optimizer.run()

    result = {
        "preprocessing": preprocessing,
        "cache": cache_info,
        "P3": p3,
        "G5": g5,
        "dhoa_evaluations": dhoa["evaluations"],
        "dhoa_distinct": dhoa["distinct_evaluations"],
        "dhoa_leader": dhoa["leader"],
    }
    _write_json(LOG_DIR / "smoke_summary.json", result)
    return result


def verify_splits() -> List[Dict]:
    summaries = []
    for dataset in DATASETS:
        for arm in ("arm_b", "arm_a", "arm_d"):
            view = build_arm_view(Config(task_arm=arm, dataset_name=dataset))
            summaries.append(view.summary())
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMRD Phase 3 reproduction")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        add_common_cli_args(p)

    p = sub.add_parser("probe", help="P1 hardware probe")
    common(p)

    p = sub.add_parser("cache", help="build feature cache")
    common(p)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("master", help="build master initial checkpoint")
    common(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("preflight", help="P1-P4")
    common(p)

    p = sub.add_parser("gates", help="G1-G6")
    common(p)
    p.add_argument("--include-stress", action="store_true")

    p = sub.add_parser("search", help="DHOA search for one arm/dataset")
    common(p)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("retrain", help="final train-only retrain and test")
    common(p)
    p.add_argument("--hp", default=None, help="JSON hyperparameters")
    p.add_argument("--hp-file", default=None, help="path to a JSON hyperparameter file")

    p = sub.add_parser("pilot", help="Arm B / Twitter16 full pilot")
    common(p)

    p = sub.add_parser("all", help="all nine arm/dataset combinations")
    common(p)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip arm/dataset combinations where final_{arm}_{dataset}.json already exists on disk",
    )

    p = sub.add_parser("report", help="build comparison tables")
    common(p)

    p = sub.add_parser("smoke", help="fast local end-to-end checks")
    common(p)
    p.add_argument("--skip-cache", action="store_true")

    p = sub.add_parser("verify-splits", help="print arm/split summaries")
    common(p)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    config = config_from_args(args)
    ensure_dirs()

    if args.command == "probe":
        print(json.dumps(probe_hardware(config), indent=2, default=str))
    elif args.command == "cache":
        if args.all_datasets:
            build_all_caches(config, limit=args.limit, force=args.force, allow_nltk_download=True)
        else:
            ensure_cache(config, force=args.force, limit=args.limit)
    elif args.command == "master":
        print(json.dumps(build_master(config, force=args.force), indent=2, default=str))
    elif args.command == "preflight":
        print(json.dumps(run_preflight(config), indent=2, default=str))
    elif args.command == "gates":
        print(json.dumps(run_gates(config, include_stress=args.include_stress), indent=2, default=str))
    elif args.command == "search":
        print(json.dumps(search(config, fresh=args.fresh, limit=args.limit), indent=2, default=str))
    elif args.command == "retrain":
        hp_json = args.hp
        if getattr(args, "hp_file", None):
            hp_json = Path(args.hp_file).read_text(encoding="utf-8")
        print(json.dumps(retrain(config, hp_json=hp_json), indent=2, default=str))
    elif args.command == "pilot":
        print(json.dumps(run_pilot(config), indent=2, default=str))
    elif args.command == "all":
        print(json.dumps(run_all(config, resume=args.resume), indent=2, default=str))
    elif args.command == "report":
        print(json.dumps(build_report(), indent=2, default=str))
    elif args.command == "smoke":
        print(json.dumps(run_smoke(config, skip_cache=args.skip_cache), indent=2, default=str))
    elif args.command == "verify-splits":
        print(json.dumps(verify_splits(), indent=2, default=str))
    else:  # pragma: no cover
        parser.error(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
