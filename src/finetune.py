"""
End-to-end fine-tuning of the source-text classifier (stage 1 baseline).

Validation discipline
---------------------
* Model selection (early stopping, best-epoch choice) uses **validation
  macro-F1 only**.
* The test split is evaluated exactly **once**, after training has finished,
  with the best-validation weights.  Nothing in the loop reads test labels.
* Test metrics are reported on the full test split *and* on the dedup-clean
  subset (test rows with no near-duplicate in train/val, see ``src/dedup.py``).

Resume safety
-------------
After every epoch ``<ckpt_dir>/last.pt`` holds model, optimizer, scheduler,
grad-scaler, RNG state, history and early-stopping state, written atomically.
Re-running the same command resumes from the next epoch; a disconnect costs at
most one epoch.  When a run finishes, ``result.json`` and ``artifact/`` are
written and the bulky ``last.pt`` / ``best_model.pt`` are removed unless
``keep_checkpoints`` is set.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.config import Config, ROOT, replace_file
from src.data import ArmView, build_arm_view, load_dataset
from src.encoder_model import SourceTextClassifier, save_artifact
from src.encoder_preprocess import normalize_tweet, normalizer_config
from src.evaluate import classification_metrics, wilson_ci

LOGGER = logging.getLogger(__name__)

TASK_TO_ARM = {"4way": "arm_a", "binary": "arm_b"}
DEFAULT_TASK = {"twitter15": "4way", "twitter16": "4way", "pheme": "binary"}


@dataclass
class FinetuneConfig:
    dataset: str = "twitter16"
    task: str = "4way"  # 4way | binary
    encoder_name: str = "roberta-base"
    max_len: int = 128
    batch_size: int = 16
    micro_batch_size: Optional[int] = None  # gradient accumulation; None = batch_size
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    label_smoothing: float = 0.1
    class_weights: str = "auto"  # auto | on | off
    max_epochs: int = 6
    patience: int = 3
    seed: int = 42
    grad_clip: float = 1.0
    amp: str = "auto"  # auto | bfloat16 | float16 | float32
    eval_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.task not in TASK_TO_ARM:
            raise ValueError(f"task must be one of {sorted(TASK_TO_ARM)}")
        if self.class_weights not in ("auto", "on", "off"):
            raise ValueError("class_weights must be auto/on/off")
        if self.amp not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError("invalid amp")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("max_epochs, patience and batch_size must be >= 1")

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def run_name(self) -> str:
        short = self.encoder_name.split("/")[-1]
        return f"{self.dataset}_{self.task}_{short}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class EarlyStopper:
    """Maximise a score; stop after ``patience`` epochs without improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, score: float, epoch: int) -> bool:
        if self.best is None or score > self.best + 1e-12:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> Dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: Dict) -> None:
        self.best = state["best"]
        self.best_epoch = int(state["best_epoch"])
        self.bad_epochs = int(state["bad_epochs"])


class _SimulatedInterrupt(RuntimeError):
    """Raised by tests to emulate a Colab disconnect after an epoch."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(amp: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda" or amp == "float32":
        return torch.float32
    if amp == "bfloat16":
        return torch.bfloat16
    if amp == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _chunks(items: Sequence[int], size: int) -> List[List[int]]:
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _atomic_torch_save(payload: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    replace_file(tmp, path)


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def compute_class_weights(labels: Sequence[int], n_classes: int, mode: str) -> Optional[torch.Tensor]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(float)
    if mode == "off":
        return None
    if mode == "auto":
        nonzero = counts[counts > 0]
        if len(nonzero) == 0 or nonzero.max() / nonzero.min() < 1.5:
            return None
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def test_report(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int, class_names: Sequence[str]
) -> Dict:
    """Accuracy, per-class/macro F1, confusion matrix and Wilson 95% CI."""
    y_true = [int(y) for y in y_true]
    y_pred = [int(y) for y in y_pred]
    metrics = classification_metrics(y_true, y_pred, n_classes)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    low, high = wilson_ci(correct, len(y_true))
    for entry in metrics["per_class"]:
        entry["name"] = class_names[entry["class"]]
    return {
        "n": len(y_true),
        "correct": correct,
        "accuracy": metrics["accuracy"],
        "accuracy_wilson_95": [low, high],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": list(class_names),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def prepare_data(cfg: FinetuneConfig, df=None, view: Optional[ArmView] = None):
    if df is None:
        df = load_dataset(cfg.dataset)
    if view is None:
        view = build_arm_view(
            Config(task_arm=TASK_TO_ARM[cfg.task], dataset_name=cfg.dataset), df=df
        )
    texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]
    return df, view, texts


def _tokenize(tokenizer, texts: List[str], max_len: int, device: torch.device):
    enc = tokenizer(
        texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt"
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def evaluate_split(
    model: SourceTextClassifier,
    tokenizer,
    texts: List[str],
    indices: Sequence[int],
    labels: Dict[int, int],
    cfg: FinetuneConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    preds: List[int] = []
    probs: List[List[float]] = []
    for chunk in _chunks(indices, cfg.eval_batch_size):
        ids, mask = _tokenize(tokenizer, [texts[i] for i in chunk], cfg.max_len, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32
        ):
            logits = model(ids, mask)
        logits = logits.float()
        target = torch.tensor([labels[i] for i in chunk], device=device)
        total_loss += float(criterion(logits, target))
        p = torch.softmax(logits, dim=-1).cpu()
        probs.extend(p.tolist())
        preds.extend(p.argmax(dim=-1).tolist())
    model.train()
    y_true = [labels[i] for i in indices]
    metrics = classification_metrics(y_true, preds, model.n_classes)
    return {
        "loss": total_loss / max(1, len(indices)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "predictions": preds,
        "probabilities": probs,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _make_optimizer(model: nn.Module, cfg: FinetuneConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_finetune(
    cfg: FinetuneConfig,
    ckpt_dir: Path | str,
    *,
    device: Optional[str] = None,
    clean_test_indices: Optional[Sequence[int]] = None,
    df=None,
    view: Optional[ArmView] = None,
    tokenizer=None,
    model: Optional[SourceTextClassifier] = None,
    keep_checkpoints: bool = False,
    _stop_after_epoch: Optional[int] = None,
) -> Dict:
    """Train, early-stop on val macro-F1, evaluate test once, save the artifact."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_path = ckpt_dir / "result.json"
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best_model.pt"

    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("config") != cfg.to_dict():
            raise RuntimeError(
                f"{result_path} exists for a different configuration; "
                "use a different checkpoint directory or --fresh"
            )
        LOGGER.info("run already finished: %s", result_path)
        return previous

    dev = resolve_device(device)
    amp_dtype = resolve_amp(cfg.amp, dev)
    seed_everything(cfg.seed)

    df, view, texts = prepare_data(cfg, df=df, view=view)
    n_classes = view.n_classes
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    if model is None:
        model = SourceTextClassifier.from_pretrained(cfg.encoder_name, n_classes, cfg.dropout)
    model.to(dev)
    model.train()

    weights = compute_class_weights(view.train_labels, n_classes, cfg.class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(dev) if weights is not None else None,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = _make_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(view.train_indices) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.max_epochs
    scheduler = _make_scheduler(optimizer, total_steps, int(cfg.warmup_ratio * total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    stopper = EarlyStopper(cfg.patience)
    history: List[Dict] = []
    start_epoch = 0

    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config"] != cfg.to_dict():
            raise RuntimeError(
                f"{last_path} belongs to a different configuration; "
                "use a different checkpoint directory or --fresh"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stopper.load_state_dict(state["stopper"])
        history = state["history"]
        start_epoch = int(state["epoch"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        LOGGER.info("resumed %s at epoch %d", ckpt_dir, start_epoch)

    micro = cfg.micro_batch_size or cfg.batch_size
    LOGGER.info(
        "train=%d val=%d test=%d classes=%d device=%s amp=%s class_weights=%s",
        len(view.train_indices), len(view.val_indices), len(view.test_indices),
        n_classes, dev, str(amp_dtype).replace("torch.", ""),
        None if weights is None else [round(w, 3) for w in weights.tolist()],
    )

    for epoch in range(start_epoch + 1, cfg.max_epochs + 1):
        if stopper.should_stop:
            break
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        order = np.random.default_rng([cfg.seed, epoch]).permutation(view.train_indices)
        running, seen = 0.0, 0
        for batch in _chunks(order.tolist(), cfg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            for part in _chunks(batch, micro):
                ids, mask = _tokenize(tokenizer, [texts[i] for i in part], cfg.max_len, dev)
                target = torch.tensor([view.labels[i] for i in part], device=dev)
                with torch.autocast(
                    device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32
                ):
                    logits = model(ids, mask)
                loss = criterion(logits.float(), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                scaler.scale(loss * (len(part) / len(batch))).backward()
                running += loss.detach().item() * len(part)
                seen += len(part)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0

        val = evaluate_split(
            model, tokenizer, texts, view.val_indices, view.labels, cfg, dev, amp_dtype
        )
        epoch_seconds = time.perf_counter() - t0
        improved = stopper.update(val["macro_f1"], epoch)
        if improved:
            _atomic_torch_save({"model": model.state_dict(), "epoch": epoch}, best_path)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "improved": improved,
            "lr_end": scheduler.get_last_lr()[0],
            "train_seconds": round(train_seconds, 2),
            "epoch_seconds": round(epoch_seconds, 2),
            "train_examples_per_second": round(seen / max(train_seconds, 1e-9), 2),
            "peak_vram_gib": (
                round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                if dev.type == "cuda" else 0.0
            ),
        }
        history.append(record)
        LOGGER.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macroF1=%.4f "
            "%s time=%.1fs (%.1f ex/s) peak_vram=%.2fGiB",
            epoch, cfg.max_epochs, record["train_loss"], val["loss"], val["accuracy"],
            val["macro_f1"], "*best*" if improved else "", epoch_seconds,
            record["train_examples_per_second"], record["peak_vram_gib"],
        )
        _atomic_torch_save(
            {
                "config": cfg.to_dict(),
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "stopper": stopper.state_dict(),
                "history": history,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            last_path,
        )
        if _stop_after_epoch is not None and epoch == _stop_after_epoch:
            raise _SimulatedInterrupt(f"simulated interrupt after epoch {epoch}")

    # ---- finalize: single test evaluation with best-validation weights ----
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    test = evaluate_split(
        model, tokenizer, texts, view.test_indices, view.labels, cfg, dev, amp_dtype
    )
    y_test = view.test_labels
    full = test_report(y_test, test["predictions"], n_classes, view.class_names)
    clean = None
    if clean_test_indices is not None:
        pos = {idx: k for k, idx in enumerate(view.test_indices)}
        keep = [pos[i] for i in clean_test_indices if i in pos]
        clean = test_report(
            [y_test[k] for k in keep],
            [test["predictions"][k] for k in keep],
            n_classes,
            view.class_names,
        )

    artifact_dir = ckpt_dir / "artifact"
    save_artifact(
        model,
        tokenizer,
        artifact_dir,
        class_names=view.class_names,
        encoder_name=cfg.encoder_name,
        max_len=cfg.max_len,
        extra={
            "dataset": cfg.dataset,
            "task": cfg.task,
            "best_epoch": stopper.best_epoch,
            "best_val_macro_f1": stopper.best,
            "training_config": cfg.to_dict(),
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    result = {
        "config": cfg.to_dict(),
        "dataset": cfg.dataset,
        "task": cfg.task,
        "class_names": view.class_names,
        "split": {
            "source": f"data/splits/{cfg.dataset}_train_val_test.json (seed 42)",
            "train": len(view.train_indices),
            "val": len(view.val_indices),
            "test": len(view.test_indices),
            "clean_test": None if clean is None else clean["n"],
        },
        "device": str(dev),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "class_weights": None if weights is None else weights.tolist(),
        "normalizer": normalizer_config(),
        "epochs_run": len(history),
        "best_epoch": stopper.best_epoch,
        "best_val_macro_f1": stopper.best,
        "early_stopped": stopper.should_stop and len(history) < cfg.max_epochs,
        "history": history,
        "test_full": full,
        "test_clean": clean,
        "test_predictions": [
            {
                "row": int(i),
                "tweet_id": str(df["tweet_id"].iloc[i]),
                "label": int(view.labels[i]),
                "pred": int(p),
                "confidence": float(max(pr)),
            }
            for i, p, pr in zip(view.test_indices, test["predictions"], test["probabilities"])
        ],
        "artifact": str(artifact_dir),
        "git_commit": _git_commit(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    replace_file(tmp, result_path)

    if not keep_checkpoints:
        for path in (last_path, best_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    return result
