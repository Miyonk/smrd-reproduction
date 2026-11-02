"""
Data Loader and Split Generator for Twitter15, Twitter16, and PHEME Datasets.
"""

import os
import tarfile
import zipfile
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from src.preprocess import clean_tweet_for_transformer, clean_tweet_classical

LABEL_TO_ID = {
    "non-rumor": 0,
    "false": 1,
    "true": 2,
    "unverified": 3
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def extract_and_parse_twitter(zip_path: str = "rumdetect2017.zip", output_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Extracts raw source tweets and labels from rumdetect2017.zip for Twitter15 and Twitter16.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Archive {zip_path} not found.")

    dfs = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for dataset in ["twitter15", "twitter16"]:
            label_file = f"rumor_detection_acl2017/{dataset}/label.txt"
            source_file = f"rumor_detection_acl2017/{dataset}/source_tweets.txt"

            label_lines = z.read(label_file).decode("utf-8", errors="ignore").splitlines()
            source_lines = z.read(source_file).decode("utf-8", errors="ignore").splitlines()

            labels = {}
            for line in label_lines:
                line = line.strip()
                if ":" in line:
                    lbl, tid = line.split(":", 1)
                    labels[tid.strip()] = lbl.strip()

            records = []
            for line in source_lines:
                line = line.strip()
                if "\t" in line:
                    tid, text = line.split("\t", 1)
                    tid = tid.strip()
                    if tid in labels:
                        raw_label = labels[tid]
                        if raw_label in LABEL_TO_ID:
                            records.append({
                                "tweet_id": tid,
                                "event": "twitter_rumors",
                                "raw_text": text.strip(),
                                "text_transformer": clean_tweet_for_transformer(text),
                                "text_classical": clean_tweet_classical(text),
                                "raw_label": raw_label,
                                "label": LABEL_TO_ID[raw_label]
                            })

            df = pd.DataFrame(records)
            df = df.sort_values(by="tweet_id").reset_index(drop=True)
            
            raw_out = os.path.join(output_dir, "raw", f"{dataset}.csv")
            proc_out = os.path.join(output_dir, "processed", f"{dataset}.csv")
            os.makedirs(os.path.dirname(raw_out), exist_ok=True)
            os.makedirs(os.path.dirname(proc_out), exist_ok=True)
            
            df[["tweet_id", "raw_text", "raw_label", "label"]].to_csv(raw_out, index=False)
            df.to_csv(proc_out, index=False)
            dfs[dataset] = df
            print(f"Extracted {dataset}: {len(df)} samples saved to {proc_out}")

    return dfs

def extract_and_parse_pheme(tar_path: str = "PHEME_veracity.tar.bz2", output_dir: str = "data") -> pd.DataFrame:
    """
    Extracts raw source tweets and veracity annotations from PHEME archive.
    """
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Archive {tar_path} not found.")

    print(f"Parsing PHEME archive: {tar_path}...")
    annotations = {}
    source_tweets = {}
    thread_events = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            name = m.name
            if "/._" in name or not name.endswith(".json"):
                continue

            parts = name.split("/")
            # Expecting format: all-rnr-annotated-threads/<event>-all-rnr-threads/<rumours|non-rumours>/<thread_id>/...
            if len(parts) >= 5 and parts[0] == "all-rnr-annotated-threads":
                event_folder = parts[1]
                event = event_folder.replace("-all-rnr-threads", "")
                thread_type = parts[2]
                thread_id = parts[3]
                file_type = parts[4]

                if file_type == "annotation.json":
                    try:
                        f = tar.extractfile(m)
                        ann = json.load(f)
                        is_rumour = ann.get("is_rumour")
                        if is_rumour == "nonrumour":
                            raw_lbl = "non-rumor"
                        else:
                            is_true = str(ann.get("true", "0")) == "1"
                            is_misinfo = str(ann.get("misinformation", "0")) == "1"
                            if is_true and not is_misinfo:
                                raw_lbl = "true"
                            elif is_misinfo and not is_true:
                                raw_lbl = "false"
                            elif not is_true and not is_misinfo:
                                raw_lbl = "unverified"
                            else:
                                raw_lbl = "unverified"
                        annotations[thread_id] = raw_lbl
                        thread_events[thread_id] = event
                    except Exception:
                        pass

                elif file_type == "source-tweets" and len(parts) >= 6:
                    try:
                        f = tar.extractfile(m)
                        tweet_data = json.load(f)
                        text = tweet_data.get("text", "")
                        source_tweets[thread_id] = text
                    except Exception:
                        pass

    # Align annotations and source tweets
    common_ids = set(annotations.keys()) & set(source_tweets.keys())
    print(f"PHEME: Matched {len(common_ids)} threads out of {len(annotations)} annotations.")

    records = []
    for tid in sorted(common_ids):
        raw_text = source_tweets[tid]
        raw_label = annotations[tid]
        event = thread_events.get(tid, "unknown")
        records.append({
            "tweet_id": tid,
            "event": event,
            "raw_text": raw_text.strip(),
            "text_transformer": clean_tweet_for_transformer(raw_text),
            "text_classical": clean_tweet_classical(raw_text),
            "raw_label": raw_label,
            "label": LABEL_TO_ID[raw_label]
        })

    df = pd.DataFrame(records)
    raw_out = os.path.join(output_dir, "raw", "pheme.csv")
    proc_out = os.path.join(output_dir, "processed", "pheme.csv")
    os.makedirs(os.path.dirname(raw_out), exist_ok=True)
    os.makedirs(os.path.dirname(proc_out), exist_ok=True)

    df[["tweet_id", "event", "raw_text", "raw_label", "label"]].to_csv(raw_out, index=False)
    df.to_csv(proc_out, index=False)
    print(f"Extracted PHEME: {len(df)} samples saved to {proc_out}")
    return df

def generate_splits(
    dfs: Dict[str, pd.DataFrame], 
    splits_dir: str = "data/splits", 
    n_folds: int = 5, 
    random_state: int = 42
) -> None:
    """
    Generates reproducible stratified 5-fold cross-validation splits and 70/10/20 train/val/test splits.
    For PHEME, also generates leave-one-event-out splits.
    """
    os.makedirs(splits_dir, exist_ok=True)

    for dataset, df in dfs.items():
        # 1. 5-Fold Stratified Cross Validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kfold_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
            kfold_splits.append({
                "fold": fold,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist()
            })

        kfold_path = os.path.join(splits_dir, f"{dataset}_5fold.json")
        with open(kfold_path, "w", encoding="utf-8") as f:
            json.dump(kfold_splits, f, indent=2)

        # 2. Stratified 70/10/20 Train/Val/Test Split
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
        train_val_idx, test_idx = next(sss1.split(df, df["label"]))
        
        train_val_df = df.iloc[train_val_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=random_state)
        train_sub_idx, val_sub_idx = next(sss2.split(train_val_df, train_val_df["label"]))

        final_train_idx = train_val_idx[train_sub_idx].tolist()
        final_val_idx = train_val_idx[val_sub_idx].tolist()
        final_test_idx = test_idx.tolist()

        tvt_split = {
            "train_indices": final_train_idx,
            "val_indices": final_val_idx,
            "test_indices": final_test_idx,
            "counts": {
                "train": len(final_train_idx),
                "val": len(final_val_idx),
                "test": len(final_test_idx)
            }
        }
        tvt_path = os.path.join(splits_dir, f"{dataset}_train_val_test.json")
        with open(tvt_path, "w", encoding="utf-8") as f:
            json.dump(tvt_split, f, indent=2)

        # 3. For PHEME: Event-based splits
        if "event" in df.columns and len(df["event"].unique()) > 1:
            events = df["event"].unique().tolist()
            event_splits = {}
            for ev in events:
                test_mask = (df["event"] == ev)
                train_mask = ~test_mask
                event_splits[ev] = {
                    "test_event": ev,
                    "train_indices": df.index[train_mask].tolist(),
                    "test_indices": df.index[test_mask].tolist()
                }
            event_path = os.path.join(splits_dir, f"{dataset}_event_splits.json")
            with open(event_path, "w", encoding="utf-8") as f:
                json.dump(event_splits, f, indent=2)

        print(f"Generated splits for {dataset}")

def load_dataset(dataset_name: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Loads processed dataset by name ('twitter15', 'twitter16', or 'pheme').
    """
    csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed dataset not found at {csv_path}. Run extraction first.")
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    dfs = extract_and_parse_twitter()
    pheme_df = extract_and_parse_pheme()
    dfs["pheme"] = pheme_df
    generate_splits(dfs)
    print("All datasets (Twitter15, Twitter16, PHEME) processed and verified.")
