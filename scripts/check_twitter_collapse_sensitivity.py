"""
Sensitivity Check: Twitter16 Extended-Epoch Diagnostic.

Section 3 of Reproduction Protocol:
After Arm A and Arm D are evaluated, this diagnostic retrains Twitter16
with the PHEME Arm B winner configuration (L≈6.2e-5, B=26, NL=2, NAH=1, DR≈0.13)
for extended epochs (default: 10 epochs, > 2) and records both accuracy and macro F1.

Diagnostic Question:
- If minority class (non-rumour, class 0) is still never predicted:
  The 2-epoch budget is NOT the primary cause of the Twitter collapse.
- If minority class starts being predicted:
  The 2-epoch budget was the limiting constraint preventing convergence.

NOTE: This is recorded as a labeled sensitivity diagnostic and does NOT
replace the pre-registered Arm B headline row.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR, LOG_DIR, REPORT_DIR, Config, ensure_dirs
from src.data import build_arm_view
from src.dhoa import Hyperparameters
from src.evaluate import classification_metrics, wilson_ci
from src.features import FeatureCache
from src.train import train_candidate

LOGGER = logging.getLogger("smrd.sensitivity")


def load_pheme_winner(log_dir: Path) -> Hyperparameters:
    """Attempt to load PHEME Arm B winner from disk; fallback to known parameters."""
    for filename in ("final_arm_b_pheme.json", "dhoa_result_arm_b_pheme.json"):
        path = log_dir / filename
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                leader_hp = data.get("leader", {}).get("hyperparameters") or data.get("hyperparameters")
                if leader_hp:
                    LOGGER.info("Loaded PHEME winner from %s: %s", path.name, leader_hp)
                    return Hyperparameters(**leader_hp)
            except Exception as exc:
                LOGGER.debug("Could not read %s: %s", path, exc)

    # Fallback to the user-recorded PHEME winner parameters
    fallback = Hyperparameters(L=6.2e-5, B=26, NL=2, NAH=1, DR=0.13)
    LOGGER.info("Using reference PHEME winner hyperparameters: %s", fallback.as_dict())
    return fallback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrain Twitter16 with PHEME winner for extended epochs"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (> 2, default: 10)"
    )
    parser.add_argument(
        "--max-layers", type=int, default=None, help="Optional max layers cap"
    )
    parser.add_argument(
        "--B-phys-max", type=int, default=4, help="Micro-batch size (default: 4)"
    )
    parser.add_argument(
        "--use-8bit", action="store_true", help="Use 8-bit AdamW optimizer"
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_dirs()

    config = Config(
        task_arm="arm_b",
        dataset_name="twitter16",
        candidate_epochs=args.epochs,
        max_layers=args.max_layers,
        B_phys_max=args.B_phys_max,
        use_8bit_optimizer=args.use_8bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    hp = load_pheme_winner(LOG_DIR)
    view = build_arm_view(config)
    cache_path = CACHE_DIR / config.dataset_name
    if not cache_path.exists():
        raise SystemExit(f"Feature cache missing at {cache_path}. Run cache build first.")
    cache = FeatureCache(cache_path).load()

    master = config.master_checkpoint
    if not master.exists():
        raise SystemExit(f"Master checkpoint missing at {master}. Run master build first.")

    print("=" * 60)
    print("SENSITIVITY DIAGNOSTIC: Twitter16 with PHEME Winner (Extended Epochs)")
    print(f"Dataset: Twitter16 | Task: Arm B (binary) | Epochs: {args.epochs}")
    print(f"Hyperparameters: {hp.as_dict()}")
    print("=" * 60)

    result = train_candidate(
        config,
        hp,
        cache,
        view,
        master_path=str(master),
        test_indices=view.test_indices,
    )

    test_metrics = result.test_metrics or {}
    predictions = test_metrics.get("predictions", [])
    y_test = view.test_labels

    c0_preds = sum(1 for p in predictions if int(p) == 0)
    c1_preds = sum(1 for p in predictions if int(p) == 1)
    acc = test_metrics.get("accuracy", 0.0)
    macro_f1 = test_metrics.get("macro_f1", 0.0)

    correct = sum(1 for t, p in zip(y_test, predictions) if int(t) == int(p))
    ci_low, ci_high = wilson_ci(correct, len(y_test))

    if c0_preds > 0:
        diagnosis = (
            f"MINORITY CLASS PREDICTED ({c0_preds}/{len(predictions)}). "
            f"Macro F1 reached {macro_f1:.4f}. "
            "Conclusion: The 2-epoch budget WAS the limiting constraint preventing convergence on Twitter16."
        )
    else:
        diagnosis = (
            f"MINORITY CLASS STILL NEVER PREDICTED (0/{len(predictions)}). "
            f"Model collapsed to constant majority class (Macro F1={macro_f1:.4f}). "
            "Conclusion: The 2-epoch budget is NOT the limit; dataset/feature separation or class prior dominates."
        )

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS:")
    print(f"Status: {result.status}")
    print(f"Validation Accuracy: {result.fitness}")
    print(f"Test Accuracy: {acc:.4f} (95% Wilson CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Predictions distribution: Class 0 (non-rumour)={c0_preds}, Class 1 (rumour)={c1_preds}")
    print(f"Diagnosis: {diagnosis}")
    print("=" * 60)

    payload = {
        "diagnostic_id": "sensitivity_twitter16_extended_epochs",
        "task_arm": "arm_b",
        "dataset": "twitter16",
        "epochs": args.epochs,
        "hyperparameters": hp.as_dict(),
        "status": result.status,
        "val_accuracy": result.fitness,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "wilson_ci_95": [ci_low, ci_high],
        "class_0_predictions": c0_preds,
        "class_1_predictions": c1_preds,
        "diagnosis": diagnosis,
        "test_metrics": test_metrics,
        "execution_stats": result.stats,
    }

    out_file = LOG_DIR / "ablation_twitter16_pheme_winner_extended_epochs.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved diagnostic record to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
