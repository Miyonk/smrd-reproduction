"""
Offline RoBERTa feature extraction and memory-mapped cache.

Specification: Section 2.3 and Section 3.1 of the technical design.

The cache stores one contiguous FP16 tensor of shape ``(sum(T_i), 768)`` plus
a JSON metadata file and a SHA-256 manifest.  The cache is built once per
dataset and shared by every task arm and every DHOA candidate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import CACHE_DIR, DATASETS, PROCESSED_DIR, Config, ensure_dirs
from src.data import load_dataset
from src.preprocess import TextPreprocessor

LOGGER = logging.getLogger(__name__)

CACHE_DTYPE = np.float16
HIDDEN_SIZE = 768


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _versions() -> Dict[str, str]:
    import transformers

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


# ---------------------------------------------------------------------------
# Cache reader
# ---------------------------------------------------------------------------
@dataclass
class CacheBatch:
    features: torch.Tensor  # (B, T, 768)
    attention_mask: torch.Tensor  # (B, T)
    lengths: torch.Tensor  # (B,)


class FeatureCache:
    """Reader for a ``<dataset>_features.bin`` / ``_metadata.json`` pair."""

    def __init__(self, prefix: Path | str):
        self.prefix = Path(prefix)
        self.bin_path = self.prefix.with_name(self.prefix.name + "_features.bin")
        self.meta_path = self.prefix.with_name(self.prefix.name + "_metadata.json")
        self.manifest_path = self.prefix.with_name(self.prefix.name + "_manifest.sha256")
        self._features: np.memmap | None = None
        self._meta: Dict | None = None

    # -- loading -----------------------------------------------------------
    def exists(self) -> bool:
        return self.bin_path.exists() and self.meta_path.exists()

    def load(self) -> "FeatureCache":
        if not self.exists():
            raise FileNotFoundError(
                f"feature cache missing: {self.bin_path} / {self.meta_path}"
            )
        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        total_tokens = int(sum(self._meta["lengths"]))
        self._features = np.memmap(
            self.bin_path,
            dtype=CACHE_DTYPE,
            mode="r",
            shape=(total_tokens, HIDDEN_SIZE),
        )
        return self

    @property
    def metadata(self) -> Dict:
        if self._meta is None:
            self.load()
        assert self._meta is not None
        return self._meta

    @property
    def features(self) -> np.memmap:
        if self._features is None:
            self.load()
        assert self._features is not None
        return self._features

    def __len__(self) -> int:
        return len(self.metadata["lengths"])

    # -- access ------------------------------------------------------------
    def sequence(self, index: int) -> np.ndarray:
        meta = self.metadata
        lengths = meta["lengths"]
        offsets = meta["offsets"]
        length = int(lengths[index])
        start = int(offsets[index])
        return self.features[start : start + length]

    def get_batch(
        self,
        indices: Sequence[int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_length: int | None = None,
    ) -> CacheBatch:
        if not indices:
            raise ValueError("cannot build an empty batch")
        sequences = [self.sequence(i) for i in indices]
        lengths = [seq.shape[0] for seq in sequences]
        width = max(lengths)
        if max_length is not None:
            width = min(width, max_length)

        features = torch.zeros((len(indices), width, HIDDEN_SIZE), dtype=dtype)
        attention_mask = torch.zeros((len(indices), width), dtype=torch.float32)
        for row, seq in enumerate(sequences):
            take = min(seq.shape[0], width)
            features[row, :take] = torch.from_numpy(seq[:take].astype(np.float32)).to(dtype)
            attention_mask[row, :take] = 1.0
        return CacheBatch(
            features=features.to(device),
            attention_mask=attention_mask.to(device),
            lengths=torch.tensor(lengths, dtype=torch.long),
        )

    def label(self, index: int) -> int:
        return int(self.metadata["labels"][index])


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------
def _token_lengths(tokenizer, texts: List[str], max_seq_len: int) -> List[int]:
    encoded = tokenizer(
        texts,
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        padding=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def build_feature_cache(
    dataset_name: str,
    config: Config | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    force: bool = False,
    allow_nltk_download: bool = False,
    data_dir: Path | str = PROCESSED_DIR,
    cache_dir: Path | str = CACHE_DIR,
) -> FeatureCache:
    """Preprocess ``dataset_name`` and write the FP16 feature cache."""
    config = config or Config(dataset_name=dataset_name)
    ensure_dirs()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = cache_dir / dataset_name

    cache = FeatureCache(prefix)
    if cache.exists() and not force:
        LOGGER.info("feature cache already exists for %s (use force=True to rebuild)", dataset_name)
        return cache.load()

    df = load_dataset(dataset_name, data_dir=data_dir)
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    LOGGER.info("preprocessing %d rows for %s", len(df), dataset_name)
    preprocessor = TextPreprocessor(allow_nltk_download=allow_nltk_download)
    texts = [preprocessor(raw) for raw in df["raw_text"].astype(str).tolist()]

    from transformers import RobertaModel, RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained(config.roberta_model)
    lengths = _token_lengths(tokenizer, texts, config.max_seq_len)
    total_tokens = int(sum(lengths))
    LOGGER.info(
        "tokenised %d rows, total tokens=%d (%.2f MiB FP16)",
        len(texts),
        total_tokens,
        total_tokens * HIDDEN_SIZE * 2 / (1024 ** 2),
    )

    device = config.resolve_device()
    model = RobertaModel.from_pretrained(config.roberta_model)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    bin_path = prefix.with_name(prefix.name + "_features.bin")
    meta_path = prefix.with_name(prefix.name + "_metadata.json")
    manifest_path = prefix.with_name(prefix.name + "_manifest.sha256")

    array = np.memmap(
        bin_path, dtype=CACHE_DTYPE, mode="w+", shape=(total_tokens, HIDDEN_SIZE)
    )

    offsets: List[int] = []
    cursor = 0
    for length in lengths:
        offsets.append(cursor)
        cursor += length

    written = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"extract {dataset_name}"):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=config.max_seq_len,
                truncation=True,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state.float().cpu().numpy()
            mask = encoded["attention_mask"].cpu().numpy()
            for row in range(hidden.shape[0]):
                length = int(mask[row].sum())
                array[written : written + length] = hidden[row, :length].astype(CACHE_DTYPE)
                written += length

    array.flush()
    del array
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metadata = {
        "dataset": dataset_name,
        "roberta_model": config.roberta_model,
        "max_seq_len": config.max_seq_len,
        "hidden_size": HIDDEN_SIZE,
        "dtype": "float16",
        "lengths": lengths,
        "offsets": offsets,
        "labels": df["label"].astype(int).tolist(),
        "tweet_ids": df["tweet_id"].astype(str).tolist(),
        "events": df["event"].astype(str).tolist() if "event" in df.columns else [],
        "texts": texts,
        "text_sha256": sha256_text("\n".join(texts)),
        "preprocessor_backend": preprocessor.backend,
        "preprocessor_config": {
            "max_chars": preprocessor.max_chars,
            "remove_stopwords": preprocessor.remove_stopwords,
            "light10": preprocessor.light10,
            "lemmatize": preprocessor.lemmatize,
        },
        "limit": limit,
        "row_count": len(df),
        "total_tokens": total_tokens,
        "versions": _versions(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "features_sha256": sha256_file(bin_path),
        "metadata_sha256": sha256_file(meta_path),
        "text_sha256": metadata["text_sha256"],
        "row_count": metadata["row_count"],
        "total_tokens": total_tokens,
        "roberta_model": config.roberta_model,
        "preprocessor_backend": preprocessor.backend,
        "versions": metadata["versions"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s and manifest", bin_path)
    return FeatureCache(prefix).load()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_manifest(prefix: Path | str) -> Dict[str, object]:
    cache = FeatureCache(prefix)
    if not cache.manifest_path.exists():
        raise FileNotFoundError(cache.manifest_path)
    manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
    actual_features = sha256_file(cache.bin_path)
    actual_metadata = sha256_file(cache.meta_path)
    meta = json.loads(cache.meta_path.read_text(encoding="utf-8"))
    checks = {
        "features_hash_match": actual_features == manifest["features_sha256"],
        "metadata_hash_match": actual_metadata == manifest["metadata_sha256"],
        "lengths_match_total": int(sum(meta["lengths"])) == int(manifest["total_tokens"]),
        "row_count_match": int(meta["row_count"]) == int(manifest["row_count"]),
    }
    checks["ok"] = all(checks.values())
    checks["manifest"] = manifest
    return checks


def reference_check(
    prefix: Path | str,
    config: Config | None = None,
    sample: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    """Recompute a FP32 RoBERTa forward pass for ``sample`` rows and compare."""
    config = config or Config()
    cache = FeatureCache(prefix).load()
    meta = cache.metadata
    n = min(sample, len(cache))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(cache), size=n, replace=False).tolist())

    from transformers import RobertaModel, RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained(config.roberta_model)
    device = config.resolve_device()
    model = RobertaModel.from_pretrained(config.roberta_model).eval().to(device)

    max_abs = 0.0
    all_close = True
    with torch.no_grad():
        for index in indices:
            text = meta["texts"][index]
            encoded = tokenizer(
                text,
                max_length=config.max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            reference = model(**encoded).last_hidden_state[0].float().cpu().numpy()
            cached = cache.sequence(index).astype(np.float32)
            length = min(reference.shape[0], cached.shape[0])
            diff = np.abs(reference[:length] - cached[:length])
            max_abs = max(max_abs, float(diff.max()))
            if not np.allclose(
                cached[:length], reference[:length].astype(np.float16),
                rtol=1e-3, atol=1e-3,
            ):
                all_close = False

    return {
        "sample_size": n,
        "max_abs_error": max_abs,
        "allclose_rtol_1e-3_atol_1e-3": all_close,
        "ok": all_close,
        "backend": meta["preprocessor_backend"],
    }


def build_all_caches(
    config: Config | None = None,
    limit: int | None = None,
    force: bool = False,
    allow_nltk_download: bool = False,
) -> Dict[str, FeatureCache]:
    out: Dict[str, FeatureCache] = {}
    for dataset in DATASETS:
        out[dataset] = build_feature_cache(
            dataset,
            config=config,
            limit=limit,
            force=force,
            allow_nltk_download=allow_nltk_download,
        )
    return out
