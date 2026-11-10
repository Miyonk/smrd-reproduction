"""
Training for stage 2 (BiGCN graph arm and text+graph fusion arm).

Same discipline as stage 1 (``src/finetune.py``):

* model selection uses **validation macro-F1 only**;
* the test split is evaluated exactly **once**, with the best-validation
  weights, on the full test split and on the dedup-clean subset;
* ``last.pt`` after every epoch makes a run resumable; ``result.json`` marks it
  finished.  A run refuses to resume with a different config or graph cache.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.config import replace_file
from src.data import ArmView
from src.evaluate import classification_metrics
from src.finetune import (
    TASK_TO_ARM,
    EarlyStopper,
    _SimulatedInterrupt,
    _atomic_torch_save,
    _chunks,
    _git_commit,
    _make_scheduler,
    _tokenize,
    compute_class_weights,
    resolve_amp,
    resolve_device,
    seed_everything,
    test_report,
)
from src.propagation import N_FEATURES, TreeGraph, collate_graphs
from src.propagation_model import ARMS, PropagationClassifier, build_model, save_artifact

LOGGER = logging.getLogger(__name__)

__all__ = ["PropagationConfig", "ARM_DEFAULTS", "run_propagation", "_SimulatedInterrupt"]

# Per-arm defaults the CLI fills in.  The graph arm follows BiGCN (lr 5e-4,
# weight decay 1e-4, patience 10, constant lr); the fusion arm follows stage 1
# for the text side with a short warmup.
ARM_DEFAULTS: Dict[str, Dict] = {
    "graph": dict(batch_size=64, schedule="constant", warmup_ratio=0.0, max_epochs=100, patience=10),
    "fusion": dict(batch_size=16, schedule="linear", warmup_ratio=0.1, max_epochs=8, patience=3),
}


@dataclass
class PropagationConfig:
    dataset: str = "twitter16"
    task: str = "4way"  # 4way | binary
    arm: str = "fusion"  # graph | fusion
    encoder_name: str = "roberta-base"  # fusion arm only
    max_len: int = 128
    batch_size: int = 16
    micro_batch_size: Optional[int] = None
    lr_text: float = 2e-5
    lr_graph: float = 5e-4
    weight_decay_text: float = 0.01
    weight_decay_graph: float = 1e-4
    schedule: str = "linear"  # linear | constant
    warmup_ratio: float = 0.1
    dropout: float = 0.1  # on the text embedding
    graph_hidden: int = 64
    graph_out: int = 64
    graph_dropout: float = 0.5
    edge_drop: float = 0.2
    n_features: int = N_FEATURES
    label_smoothing: float = 0.1
    class_weights: str = "auto"
    max_epochs: int = 8
    patience: int = 3
    seed: int = 42
    grad_clip: float = 1.0
    amp: str = "auto"
    eval_batch_size: int = 64
    graph_cache_sha256: str = ""

    def __post_init__(self) -> None:
        if self.task not in TASK_TO_ARM:
            raise ValueError(f"task must be one of {sorted(TASK_TO_ARM)}")
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.schedule not in ("linear", "constant"):
            raise ValueError("schedule must be linear or constant")
        if self.class_weights not in ("auto", "on", "off"):
            raise ValueError("class_weights must be auto/on/off")
        if self.amp not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError("invalid amp")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("max_epochs, patience and batch_size must be >= 1")
        if not 0.0 <= self.edge_drop < 1.0:
            raise ValueError("edge_drop must be in [0, 1)")

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def run_name(self) -> str:
        if self.arm == "graph":
            return f"{self.dataset}_{self.task}_bigcn"
        return f"{self.dataset}_{self.task}_fusion_{self.encoder_name.split('/')[-1]}"

    def model_config(self) -> Dict:
        return {
            "n_features": self.n_features,
            "hidden": self.graph_hidden,
            "out": self.graph_out,
            "graph_dropout": self.graph_dropout,
            "edge_drop": self.edge_drop,
            "dropout": self.dropout,
            "encoder_name": self.encoder_name if self.arm == "fusion" else None,
            "max_len": self.max_len,
        }


# ---------------------------------------------------------------------------
def _inputs(model: PropagationClassifier, tokenizer, texts, graphs, rows, cfg, device):
    g = collate_graphs([graphs[i] for i in rows]).to(device)
    if model.text_encoder is None:
        return (g,)
    ids, mask = _tokenize(tokenizer, [texts[i] for i in rows], cfg.max_len, device)
    return (g, ids, mask)


@torch.no_grad()
def evaluate_split(
    model: PropagationClassifier,
    tokenizer,
    texts: List[str],
    graphs: Sequence[TreeGraph],
    indices: Sequence[int],
    labels: Dict[int, int],
    cfg: PropagationConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    preds: List[int] = []
    probs: List[List[float]] = []
    for chunk in _chunks(indices, cfg.eval_batch_size):
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32
        ):
            logits = model(*_inputs(model, tokenizer, texts, graphs, chunk, cfg, device))
        logits = logits.float()
        target = torch.tensor([labels[i] for i in chunk], device=device)
        total_loss += criterion(logits, target).item()
        p = torch.softmax(logits, dim=-1).cpu()
        probs.extend(p.tolist())
        preds.extend(p.argmax(dim=-1).tolist())
    model.train()
    metrics = classification_metrics([labels[i] for i in indices], preds, model.n_classes)
    return {
        "loss": total_loss / max(1, len(indices)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "predictions": preds,
        "probabilities": probs,
    }


def _make_optimizer(model: PropagationClassifier, cfg: PropagationConfig) -> torch.optim.Optimizer:
    groups = {"text_decay": [], "text_no_decay": [], "graph": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("text_encoder."):
            no_decay = param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name
            groups["text_no_decay" if no_decay else "text_decay"].append(param)
        else:
            groups["graph"].append(param)
    param_groups = [
        {"params": groups["graph"], "lr": cfg.lr_graph, "weight_decay": cfg.weight_decay_graph},
    ]
    if groups["text_decay"] or groups["text_no_decay"]:
        param_groups += [
            {"params": groups["text_decay"], "lr": cfg.lr_text, "weight_decay": cfg.weight_decay_text},
            {"params": groups["text_no_decay"], "lr": cfg.lr_text, "weight_decay": 0.0},
        ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def run_propagation(
    cfg: PropagationConfig,
    ckpt_dir: Path | str,
    *,
    df,
    view: ArmView,
    texts: List[str],
    graphs: Sequence[TreeGraph],
    device: Optional[str] = None,
    clean_test_indices: Optional[Sequence[int]] = None,
    tokenizer=None,
    model: Optional[PropagationClassifier] = None,
    graph_meta: Optional[Dict] = None,
    keep_checkpoints: bool = False,
    _stop_after_epoch: Optional[int] = None,
) -> Dict:
    """Train, early-stop on val macro-F1, evaluate test once, save the artifact."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_path = ckpt_dir / "result.json"
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best_model.pt"
    if len(graphs) != len(texts):
        raise ValueError("graphs and texts must be aligned with dataframe rows")

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

    n_classes = view.n_classes
    if model is None:
        model = build_model(
            cfg.arm,
            n_classes,
            n_features=cfg.n_features,
            hidden=cfg.graph_hidden,
            out=cfg.graph_out,
            graph_dropout=cfg.graph_dropout,
            edge_drop=cfg.edge_drop,
            dropout=cfg.dropout,
            encoder_name=cfg.encoder_name,
        )
    if cfg.arm == "fusion" and tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
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
    if cfg.schedule == "linear":
        scheduler = _make_scheduler(optimizer, total_steps, int(cfg.warmup_ratio * total_steps))
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
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
        "arm=%s train=%d val=%d test=%d classes=%d device=%s amp=%s class_weights=%s",
        cfg.arm, len(view.train_indices), len(view.val_indices), len(view.test_indices),
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
                target = torch.tensor([view.labels[i] for i in part], device=dev)
                with torch.autocast(
                    device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32
                ):
                    logits = model(*_inputs(model, tokenizer, texts, graphs, part, cfg, dev))
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
            model, tokenizer, texts, graphs, view.val_indices, view.labels, cfg, dev, amp_dtype
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
            "%s time=%.1fs peak_vram=%.2fGiB",
            epoch, cfg.max_epochs, record["train_loss"], val["loss"], val["accuracy"],
            val["macro_f1"], "*best*" if improved else "", epoch_seconds, record["peak_vram_gib"],
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
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    test = evaluate_split(
        model, tokenizer, texts, graphs, view.test_indices, view.labels, cfg, dev, amp_dtype
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
        model_config=cfg.model_config(),
        class_names=view.class_names,
        extra={
            "dataset": cfg.dataset,
            "task": cfg.task,
            "best_epoch": stopper.best_epoch,
            "best_val_macro_f1": stopper.best,
            "training_config": cfg.to_dict(),
            "graph_cache": graph_meta,
            "git_commit": _git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    result = {
        "config": cfg.to_dict(),
        "dataset": cfg.dataset,
        "task": cfg.task,
        "arm": cfg.arm,
        "class_names": view.class_names,
        "split": {
            "source": f"data/splits/{cfg.dataset}_train_val_test.json (seed 42)",
            "train": len(view.train_indices),
            "val": len(view.val_indices),
            "test": len(view.test_indices),
            "clean_test": None if clean is None else clean["n"],
        },
        "graph_cache": graph_meta,
        "device": str(dev),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "class_weights": None if weights is None else weights.tolist(),
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
