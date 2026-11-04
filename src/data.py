"""
Dataset and task-arm construction.

The split JSON files produced in Phase 1 are authoritative and are never
regenerated.  This module only *selects* indices from those splits, applies
the Section 0.4 label mapping for each task arm, and exposes a compact view
that the trainer and feature cache can consume.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from src.config import DATASETS, DOUT_BY_ARM, PROCESSED_DIR, SPLITS_DIR, Config

LOGGER = logging.getLogger(__name__)

LABEL_TO_ID = {"non-rumor": 0, "false": 1, "true": 2, "unverified": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

ARM_CLASS_NAMES = {
    "arm_a": ["non-rumour", "false", "true", "unverified"],
    "arm_b": ["non-rumour", "rumour"],
    "arm_d": ["false", "true"],
    "report_c": ["non-rumour", "false", "true", "unverified"],
}

# Derived in Section 0.4 / 7.1 of the design.  Used as a warning gate, not a
# hard stop, because the physical archive is authoritative.
EXPECTED_COUNTS = {
    "arm_b": {
        "twitter15": {"train": 1043, "val": 149, "test": 298},
        "twitter16": {"train": 572, "val": 82, "test": 164},
        "pheme": {"train": 4497, "val": 643, "test": 1285},
    },
    "arm_a": {
        "twitter15": {"train": 1043, "val": 149, "test": 298},
        "twitter16": {"train": 572, "val": 82, "test": 164},
        "pheme": {"train": 4497, "val": 643, "test": 1285},
    },
    "arm_d": {
        "twitter15": {"train": 520, "val": 74, "test": 148},
        "twitter16": {"train": 288, "val": 41, "test": 83},
        "pheme": {"train": 1193, "val": 171, "test": 341},
    },
}


def load_dataset(dataset_name: str, data_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    if dataset_name not in DATASETS:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    path = Path(data_dir) / f"{dataset_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"processed dataset missing: {path}")
    return pd.read_csv(path)


def load_splits(
    dataset_name: str, splits_dir: Path | str = SPLITS_DIR
) -> Dict[str, List[int]]:
    path = Path(splits_dir) / f"{dataset_name}_train_val_test.json"
    if not path.exists():
        raise FileNotFoundError(f"split file missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = {"train_indices", "val_indices", "test_indices"} - set(raw)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    return {
        "train": list(raw["train_indices"]),
        "val": list(raw["val_indices"]),
        "test": list(raw["test_indices"]),
    }


def _arm_label(original_label: int, arm: str) -> int | None:
    """Map a 4-class physical label to an arm label (or None to exclude)."""
    if arm in ("arm_a", "report_c"):
        return original_label
    if arm == "arm_b":
        return 0 if original_label == 0 else 1
    if arm == "arm_d":
        if original_label == 1:
            return 0
        if original_label == 2:
            return 1
        return None
    raise ValueError(f"unknown arm {arm!r}")


@dataclass
class ArmView:
    arm: str
    dataset: str
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    labels: Dict[int, int]
    n_classes: int
    class_names: List[str]

    @property
    def train_labels(self) -> List[int]:
        return [self.labels[i] for i in self.train_indices]

    @property
    def val_labels(self) -> List[int]:
        return [self.labels[i] for i in self.val_indices]

    @property
    def test_labels(self) -> List[int]:
        return [self.labels[i] for i in self.test_indices]

    def summary(self) -> Dict[str, object]:
        def distribution(indices: Sequence[int]) -> Dict[str, int]:
            counts = [0] * self.n_classes
            for idx in indices:
                counts[self.labels[idx]] += 1
            return {
                self.class_names[c]: counts[c] for c in range(self.n_classes)
            }

        return {
            "arm": self.arm,
            "dataset": self.dataset,
            "n_classes": self.n_classes,
            "train": len(self.train_indices),
            "val": len(self.val_indices),
            "test": len(self.test_indices),
            "train_distribution": distribution(self.train_indices),
            "val_distribution": distribution(self.val_indices),
            "test_distribution": distribution(self.test_indices),
        }


def build_arm_view(config: Config, df: pd.DataFrame | None = None) -> ArmView:
    """Build the authoritative index view for ``config.task_arm``."""
    if config.task_arm == "report_c":
        # Report C reuses the Arm A model; training and splits are identical.
        build_arm = "arm_a"
    else:
        build_arm = config.task_arm

    if df is None:
        df = load_dataset(config.dataset_name)
    if "label" not in df.columns:
        raise ValueError("processed dataset must contain a 'label' column")

    splits = load_splits(config.dataset_name)
    labels: Dict[int, int] = {}
    for pos, original in enumerate(df["label"].astype(int).tolist()):
        mapped = _arm_label(original, build_arm)
        if mapped is not None:
            labels[pos] = mapped

    def keep(indices: Sequence[int]) -> List[int]:
        return [i for i in indices if i in labels]

    view = ArmView(
        arm=build_arm,
        dataset=config.dataset_name,
        train_indices=keep(splits["train"]),
        val_indices=keep(splits["val"]),
        test_indices=keep(splits["test"]),
        labels=labels,
        n_classes=DOUT_BY_ARM[build_arm],
        class_names=ARM_CLASS_NAMES[build_arm],
    )

    _warn_on_count_mismatch(view)
    return view


def _warn_on_count_mismatch(view: ArmView) -> None:
    expected = EXPECTED_COUNTS.get(view.arm, {}).get(view.dataset)
    if not expected:
        return
    actual = {
        "train": len(view.train_indices),
        "val": len(view.val_indices),
        "test": len(view.test_indices),
    }
    if actual != expected:
        LOGGER.warning(
            "Split counts for %s/%s differ from the design derivation: "
            "actual=%s expected=%s",
            view.arm,
            view.dataset,
            actual,
            expected,
        )


def all_arm_views() -> List[ArmView]:
    """Convenience helper for audits and tests."""
    views = []
    for dataset in DATASETS:
        for arm in ("arm_b", "arm_a", "arm_d"):
            views.append(build_arm_view(Config(task_arm=arm, dataset_name=dataset)))
    return views
