"""Train one candidate against the real cache — a timing/feasibility probe.

Example:
    python scripts/dryrun_candidate.py --max-layers 24 --epochs 1 \
        --use-8bit --gradient-checkpointing --B-phys-max 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR, Config, ensure_dirs
from src.data import build_arm_view
from src.dhoa import Hyperparameters
from src.features import FeatureCache
from src.train import train_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="arm_b")
    parser.add_argument("--dataset", default="twitter16")
    parser.add_argument("--L", type=float, default=1e-4)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--NL", type=int, default=None)
    parser.add_argument("--NAH", type=int, default=16)
    parser.add_argument("--DR", type=float, default=0.1)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--B-phys-max", type=int, default=1)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    config = Config(
        task_arm=args.arm,
        dataset_name=args.dataset,
        max_layers=args.max_layers,
        candidate_epochs=args.epochs,
        B_phys_max=args.B_phys_max,
        use_8bit_optimizer=args.use_8bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    view = build_arm_view(config)
    cache = FeatureCache(CACHE_DIR / config.dataset_name).load()
    master = config.master_checkpoint
    if not master.exists():
        raise SystemExit(f"master checkpoint missing: {master}")

    hp = Hyperparameters(
        L=args.L,
        B=args.B,
        NL=args.NL if args.NL is not None else (args.max_layers or 24),
        NAH=args.NAH,
        DR=args.DR,
    )
    print("view:", json.dumps(view.summary(), indent=2))
    print("hyperparameters:", hp.as_dict())
    result = train_candidate(
        config, hp, cache, view, master_path=str(master),
        test_indices=view.test_indices,
    )
    print("status:", result.status)
    print("fitness:", result.fitness)
    print("val_loss:", result.val_loss)
    print("stats:", json.dumps(result.stats, indent=2))
    if result.test_metrics:
        print("test_accuracy:", result.test_metrics["accuracy"])
        print("test_macro_f1:", result.test_metrics["macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
