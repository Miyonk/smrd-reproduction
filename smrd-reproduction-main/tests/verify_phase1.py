"""
Gravitas Phase 1 Deterministic Acceptance Validator.
Tests data extraction, schema integrity, label distributions, split invariants,
and text cleaning across Twitter15, Twitter16, and PHEME.
"""

import os
import json
import pandas as pd
import numpy as np

EXPECTED_FILES = [
    "data/processed/twitter15.csv",
    "data/processed/twitter16.csv",
    "data/processed/pheme.csv",
    "data/splits/twitter15_5fold.json",
    "data/splits/twitter15_train_val_test.json",
    "data/splits/twitter16_5fold.json",
    "data/splits/twitter16_train_val_test.json",
    "data/splits/pheme_5fold.json",
    "data/splits/pheme_train_val_test.json",
    "data/splits/pheme_event_splits.json"
]

EXPECTED_COUNTS = {
    "twitter15": 1490,
    "twitter16": 818,
    "pheme": 6425
}

def verify_phase1():
    print("[1/6] Verifying required artifacts...")
    for f in EXPECTED_FILES:
        if not os.path.exists(f):
            print(f"FAIL: Missing expected file: {f}")
            print(f"VERDICT: FAIL -- Missing file {f}")
            return False
        if os.path.getsize(f) == 0:
            print(f"FAIL: File is empty: {f}")
            print(f"VERDICT: FAIL -- Empty file {f}")
            return False
    print("  Artifacts exist and are non-empty.")

    print("[2/6] Verifying dataset row counts and schema...")
    for name, expected_len in EXPECTED_COUNTS.items():
        df = pd.read_csv(f"data/processed/{name}.csv")
        if len(df) != expected_len:
            print(f"FAIL: {name} length is {len(df)}, expected {expected_len}")
            print(f"VERDICT: FAIL -- Row count mismatch for {name}")
            return False

        required_cols = {"tweet_id", "raw_text", "text_transformer", "text_classical", "raw_label", "label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"FAIL: {name} missing columns: {missing_cols}")
            print(f"VERDICT: FAIL -- Schema mismatch for {name}")
            return False

        null_counts = df[list(required_cols)].isnull().sum()
        if null_counts.any():
            print(f"FAIL: {name} contains null values:\n{null_counts}")
            print(f"VERDICT: FAIL -- Null values in {name}")
            return False

        unique_labels = set(df["label"].unique())
        if unique_labels != {0, 1, 2, 3}:
            print(f"FAIL: {name} invalid labels: {unique_labels}")
            print(f"VERDICT: FAIL -- Invalid labels in {name}")
            return False
            
    print("  Row counts and schema verified.")

    print("[3/6] Verifying 5-fold cross-validation split invariants...")
    for name, total_len in EXPECTED_COUNTS.items():
        with open(f"data/splits/{name}_5fold.json", "r", encoding="utf-8") as f:
            kfold_data = json.load(f)

        if len(kfold_data) != 5:
            print(f"FAIL: {name} does not have 5 folds")
            print(f"VERDICT: FAIL -- Fold count mismatch in {name}")
            return False

        all_val_indices = []
        for fold_info in kfold_data:
            train_idx = set(fold_info["train_indices"])
            val_idx = set(fold_info["val_indices"])

            if train_idx & val_idx:
                print(f"FAIL: Overlap between train and val in fold {fold_info['fold']} of {name}")
                print("VERDICT: FAIL -- Overlapping train/val fold indices")
                return False

            if len(train_idx) + len(val_idx) != total_len:
                print(f"FAIL: Total indices in fold does not equal dataset size")
                print("VERDICT: FAIL -- Incomplete fold coverage")
                return False

            all_val_indices.extend(fold_info["val_indices"])

        if sorted(all_val_indices) != list(range(total_len)):
            print(f"FAIL: Not all indices covered exactly once across validation folds for {name}")
            print("VERDICT: FAIL -- Broken k-fold partitioning")
            return False

    print("  5-fold cross-validation invariants passed.")

    print("[4/6] Verifying 70/10/20 train/val/test split invariants...")
    for name, total_len in EXPECTED_COUNTS.items():
        with open(f"data/splits/{name}_train_val_test.json", "r", encoding="utf-8") as f:
            tvt_data = json.load(f)

        train_idx = set(tvt_data["train_indices"])
        val_idx = set(tvt_data["val_indices"])
        test_idx = set(tvt_data["test_indices"])

        if (train_idx & val_idx) or (train_idx & test_idx) or (val_idx & test_idx):
            print(f"FAIL: Non-disjoint splits in {name}")
            print("VERDICT: FAIL -- Non-disjoint train/val/test splits")
            return False

        if len(train_idx) + len(val_idx) + len(test_idx) != total_len:
            print(f"FAIL: Union of train/val/test doesn't match total {total_len} for {name}")
            print("VERDICT: FAIL -- Incomplete train/val/test split coverage")
            return False

        train_ratio = len(train_idx) / total_len
        val_ratio = len(val_idx) / total_len
        test_ratio = len(test_idx) / total_len

        if not (0.68 <= train_ratio <= 0.72):
            print(f"FAIL: Train ratio {train_ratio:.3f} outside expected [0.68, 0.72]")
            print("VERDICT: FAIL -- Train split ratio deviation")
            return False
        if not (0.08 <= val_ratio <= 0.12):
            print(f"FAIL: Val ratio {val_ratio:.3f} outside expected [0.08, 0.12]")
            print("VERDICT: FAIL -- Val split ratio deviation")
            return False
        if not (0.18 <= test_ratio <= 0.22):
            print(f"FAIL: Test ratio {test_ratio:.3f} outside expected [0.18, 0.22]")
            print("VERDICT: FAIL -- Test split ratio deviation")
            return False

        df = pd.read_csv(f"data/processed/{name}.csv")
        global_props = df["label"].value_counts(normalize=True).sort_index().to_numpy()
        test_props = df.iloc[list(test_idx)]["label"].value_counts(normalize=True).sort_index().to_numpy()

        max_prop_diff = np.max(np.abs(global_props - test_props))
        if max_prop_diff > 0.05:
            print(f"FAIL: Max class proportion shift in test set is {max_prop_diff:.4f} (> 0.05)")
            print("VERDICT: FAIL -- Poor stratification")
            return False

    print("  70/10/20 train/val/test split invariants passed.")

    print("[5/6] Verifying PHEME event-based leave-one-event-out splits...")
    with open("data/splits/pheme_event_splits.json", "r", encoding="utf-8") as f:
        event_splits = json.load(f)

    if len(event_splits) != 9:
        print(f"FAIL: Expected 9 event splits for PHEME, found {len(event_splits)}")
        print("VERDICT: FAIL -- PHEME event splits count mismatch")
        return False

    pheme_df = pd.read_csv("data/processed/pheme.csv")
    for ev, sp in event_splits.items():
        tr = set(sp["train_indices"])
        te = set(sp["test_indices"])
        if tr & te:
            print(f"FAIL: Overlap in event split {ev}")
            print("VERDICT: FAIL -- PHEME event split overlap")
            return False
        if len(tr) + len(te) != len(pheme_df):
            print(f"FAIL: Incomplete coverage in event split {ev}")
            print("VERDICT: FAIL -- PHEME event split coverage error")
            return False
        # Test events match
        tested_events = set(pheme_df.iloc[list(te)]["event"].unique())
        if tested_events != {ev}:
            print(f"FAIL: Event leakage in test set for {ev}: found {tested_events}")
            print("VERDICT: FAIL -- Event leakage in PHEME split")
            return False

    print("  PHEME 9-event leave-one-out splits verified.")

    print("[6/6] Checking text cleaning integrity...")
    for name in EXPECTED_COUNTS.keys():
        df = pd.read_csv(f"data/processed/{name}.csv")
        for i, row in df.head(10).iterrows():
            if "http://" in str(row["text_transformer"]) or "https://" in str(row["text_transformer"]):
                print(f"FAIL: Transformer text still contains raw URL in {name} row {i}")
                print("VERDICT: FAIL -- Incomplete URL cleaning")
                return False

    print("  Text cleaning integrity passed.")
    print("\nAll checks completed successfully across Twitter15, Twitter16, and PHEME.")
    print("VERDICT: PASS")
    return True

if __name__ == "__main__":
    verify_phase1()
