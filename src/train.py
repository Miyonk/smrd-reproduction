"""
Candidate training with exact gradient accumulation.

Specification: Section 6 of the technical design.

The trainer is intentionally strict about the two things that dominate
reproducibility here:

1. **Loss scaling.**  A logical batch of ``b`` examples split into micro
   batches of ``n_j`` examples scales each micro loss by ``n_j / b`` so the
   accumulated gradient equals the mean-loss gradient of the logical batch.
2. **Candidate isolation.**  Every attempt rebuilds the model from the master
   initial checkpoint; no candidate ever inherits weights.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.data import ArmView
from src.features import FeatureCache
from src.model import DynamicGPT2, build_model
from src.dhoa import Hyperparameters

LOGGER = logging.getLogger(__name__)


class DivergenceError(RuntimeError):
    """Non-finite loss; candidate is DIVERGED_NONFINITE, never fitness 0.0."""


@dataclass
class TrainResult:
    status: str
    fitness: Optional[float]
    val_loss: Optional[float]
    val_macro_f1: Optional[float]
    stats: Dict = field(default_factory=dict)
    predictions: Optional[List[int]] = None
    probabilities: Optional[List[List[float]]] = None
    test_metrics: Optional[Dict] = None
    val_predictions: Optional[List[int]] = None


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(model: nn.Module, hp: Hyperparameters, use_8bit: bool):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if use_8bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                parameters,
                lr=hp.L,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            LOGGER.warning("8-bit AdamW unavailable (%s); using torch AdamW", exc)
    return torch.optim.AdamW(
        parameters, lr=hp.L, betas=(0.9, 0.999), weight_decay=0.01
    )


def _chunks(items: Sequence[int], size: int) -> List[Sequence[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _move_labels(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(labels), dtype=torch.long, device=device)


def _forward_batch(
    model: DynamicGPT2,
    cache: FeatureCache,
    indices: Sequence[int],
    labels: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = cache.get_batch(indices, device=device, dtype=torch.float32)
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=(dtype in (torch.float16, torch.bfloat16)),
    ):
        logits = model(batch.features, batch.attention_mask)
    target = _move_labels(labels, device)
    return logits, target


def evaluate_model(
    model: DynamicGPT2,
    cache: FeatureCache,
    indices: Sequence[int],
    labels: Sequence[int],
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_size: int = 8,
) -> Dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    total_loss = 0.0
    predictions: List[int] = []
    probabilities: List[List[float]] = []
    label_map = {index: label for index, label in zip(indices, labels)}
    with torch.no_grad():
        for chunk in _chunks(list(indices), batch_size):
            chunk_labels = [label_map[i] for i in chunk]
            logits, target = _forward_batch(
                model, cache, chunk, chunk_labels, device, amp_dtype
            )
            loss = criterion(logits.float(), target)
            total_loss += float(loss.item()) * len(chunk)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probabilities.extend(probs.tolist())
            predictions.extend(probs.argmax(axis=-1).tolist())

    avg_loss = total_loss / max(1, len(indices))
    accuracy = (
        float(np.mean(np.array(predictions) == np.array(labels))) if indices else 0.0
    )
    from src.evaluate import classification_metrics

    metrics = classification_metrics(
        list(labels), predictions, n_classes=model.config.d_out or 2
    )
    model.train()
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": metrics["macro_f1"],
        "predictions": predictions,
        "probabilities": probabilities,
    }


def _train_once(
    config: Config,
    hp: Hyperparameters,
    cache: FeatureCache,
    view: ArmView,
    master_path: Optional[str],
    device: torch.device,
    b_phys: int,
    gradient_checkpointing: bool,
    test_indices: Optional[Sequence[int]] = None,
    save_model_path: Optional[str] = None,
) -> TrainResult:
    _seed_everything(config.seed_dropout)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(
        config,
        master_path=master_path,
        nl=hp.NL,
        load_pretrained=config.use_pretrained_gpt2,
    )
    model.set_layer_count(hp.NL)
    model.set_head_count(hp.NAH)
    model.set_dropout(hp.DR)
    model.enable_gradient_checkpointing(gradient_checkpointing)
    model.to(device)
    model.train()

    optimizer = _make_optimizer(model, hp, config.use_8bit_optimizer)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    amp_dtype = config.resolved_amp_dtype()
    if device.type == "cpu":
        amp_dtype = torch.float32
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(amp_dtype == torch.float16 and device.type == "cuda")
    )

    start = time.time()
    epoch_losses: List[float] = []
    last_metrics: Dict = {}

    for epoch in range(1, config.candidate_epochs + 1):
        rng = np.random.default_rng(config.seed_order + epoch)
        order = rng.permutation(view.train_indices).tolist()
        logical_batches = _chunks(order, hp.B)
        epoch_loss = 0.0
        seen = 0

        for logical in logical_batches:
            b = len(logical)
            micro_batches = _chunks(list(logical), b_phys)
            optimizer.zero_grad(set_to_none=True)

            for micro in micro_batches:
                chunk_labels = [view.labels[i] for i in micro]
                logits, target = _forward_batch(
                    model, cache, micro, chunk_labels, device, amp_dtype
                )
                loss = criterion(logits.float(), target)
                if not torch.isfinite(loss):
                    raise DivergenceError(f"non-finite loss {loss.item()}")
                scaled = loss * (len(micro) / b)
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                epoch_loss += float(loss.item()) * len(micro)
                seen += len(micro)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=config.grad_clip_norm,
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_losses.append(epoch_loss / max(1, seen))
        last_metrics = evaluate_model(
            model,
            cache,
            view.val_indices,
            view.val_labels,
            device,
            amp_dtype,
            batch_size=max(1, min(b_phys, 16)),
        )
        LOGGER.info(
            "epoch %d/%d loss=%.4f val_acc=%.4f",
            epoch,
            config.candidate_epochs,
            epoch_losses[-1],
            last_metrics["accuracy"],
        )

    runtime = time.time() - start
    peak = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )
    stats = {
        "B_phys_used": b_phys,
        "gradient_checkpointing": gradient_checkpointing,
        **{f"train_loss_epoch{i+1}": epoch_losses[i] for i in range(len(epoch_losses))},
        "val_loss": last_metrics["loss"],
        "val_accuracy": last_metrics["accuracy"],
        "val_macro_f1": last_metrics["macro_f1"],
        "peak_vram_gib": peak,
        "runtime_seconds": runtime,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "optimizer": type(optimizer).__name__,
    }

    test_metrics = None
    if test_indices is not None and len(test_indices) > 0:
        test_metrics = evaluate_model(
            model,
            cache,
            list(test_indices),
            [view.labels[i] for i in test_indices],
            device,
            amp_dtype,
            batch_size=max(1, min(b_phys, 16)),
        )
        test_metrics.pop("probabilities", None)
        stats["test_accuracy"] = test_metrics["accuracy"]
        stats["test_macro_f1"] = test_metrics["macro_f1"]

    if save_model_path is not None:
        path = Path(save_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyperparameters": hp.as_dict(),
                "config": config.to_dict(),
            },
            path,
        )
        stats["checkpoint_path"] = str(path)

    return TrainResult(
        status="EVALUATED",
        fitness=last_metrics["accuracy"],
        val_loss=last_metrics["loss"],
        val_macro_f1=last_metrics["macro_f1"],
        stats=stats,
        predictions=test_metrics["predictions"] if test_metrics else last_metrics["predictions"],
        probabilities=last_metrics["probabilities"],
        test_metrics=test_metrics,
        val_predictions=last_metrics["predictions"],
    )


def train_candidate(
    config: Config,
    hp: Hyperparameters,
    cache: FeatureCache,
    view: ArmView,
    master_path: Optional[str] = None,
    test_indices: Optional[Sequence[int]] = None,
    save_model_path: Optional[str] = None,
) -> TrainResult:
    """Train one candidate, applying the deterministic OOM retry ladder."""
    device = config.resolve_device()
    b_phys = max(1, min(config.B_phys_max, hp.B))
    checkpointing = config.gradient_checkpointing
    mitigations: List[str] = []
    attempt = 0

    while True:
        attempt += 1
        try:
            result = _train_once(
                config,
                hp,
                cache,
                view,
                master_path,
                device,
                b_phys=b_phys,
                gradient_checkpointing=checkpointing,
                test_indices=test_indices,
                save_model_path=save_model_path,
            )
            result.stats["mitigations_applied"] = mitigations
            result.stats["attempts"] = attempt
            return result
        except DivergenceError as exc:
            LOGGER.warning("candidate diverged: %s", exc)
            return TrainResult(
                status="DIVERGED_NONFINITE",
                fitness=None,
                val_loss=None,
                val_macro_f1=None,
                stats={
                    "B_phys_used": b_phys,
                    "mitigations_applied": mitigations,
                    "attempts": attempt,
                    "error": str(exc),
                },
            )
        except BaseException as exc:  # noqa: BLE001 - OOM can surface as RuntimeError
            if not _is_oom(exc):
                raise
            LOGGER.warning("CUDA OOM on attempt %d (B_phys=%d, checkpointing=%s)", attempt, b_phys, checkpointing)
            mitigations.append(
                f"oom_attempt_{attempt}:B_phys={b_phys},checkpointing={checkpointing}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Deterministic ladder: halve -> checkpointing -> B_phys=1 -> stop.
            if b_phys > 1:
                b_phys = max(1, b_phys // 2)
                continue
            if not checkpointing:
                checkpointing = True
                continue
            return TrainResult(
                status="INFEASIBLE_HARDWARE",
                fitness=None,
                val_loss=None,
                val_macro_f1=None,
                stats={
                    "B_phys_used": b_phys,
                    "mitigations_applied": mitigations,
                    "attempts": attempt,
                    "error": str(exc),
                },
            )
