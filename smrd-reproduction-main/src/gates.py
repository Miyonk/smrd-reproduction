"""
Pre-flight probes and staged measurement gates.

Specification: Section 8.5 of the technical design.

P1-P4 are correctness probes.  G1-G6 are empirical memory/feasibility gates.
Nothing here assumes a GPU model or a VRAM size: every number is measured at
runtime and written to ``logs/``.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import CACHE_DIR, LOG_DIR, Config, ensure_dirs
from src.features import FeatureCache, reference_check
from src.model import DynamicGPT2, build_model, state_dict_digest

LOGGER = logging.getLogger(__name__)


def _write(name: str, payload: Dict) -> Path:
    ensure_dirs()
    path = LOG_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _fake_batch(batch: int, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.randn(batch, length, 768, device=device)
    mask = torch.ones(batch, length, device=device)
    return features, mask


def _peak_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ---------------------------------------------------------------------------
# P1 - hardware probe
# ---------------------------------------------------------------------------
def probe_hardware(config: Config | None = None) -> Dict:
    config = config or Config()
    payload: Dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        payload.update(
            {
                "gpu_name": props.name,
                "total_vram_gib": props.total_memory / (1024 ** 3),
                "usable_vram_gib": free / (1024 ** 3),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "cuda_version": torch.version.cuda,
            }
        )
    else:
        payload.update(
            {
                "gpu_name": None,
                "total_vram_gib": 0.0,
                "usable_vram_gib": 0.0,
                "compute_capability": None,
                "bf16_supported": False,
                "cuda_version": None,
            }
        )
    usage = shutil.disk_usage(Path(config.root))
    payload["disk_free_gib"] = usage.free / (1024 ** 3)
    payload["write_permissions"] = _check_write()
    payload["phase1_pass_criterion"] = "usable_vram_gib >= 7.0"
    payload["phase1_pass"] = bool(payload["cuda_available"] and payload["usable_vram_gib"] >= 7.0)
    if not payload["phase1_pass"]:
        payload["verdict"] = "ENVIRONMENT_INCOMPATIBLE_REPORT"
        payload["reason"] = (
            "Design Section 11.5 Phase 1 requires CUDA and usable VRAM >= 7.0 GiB. "
            "Override only with an explicit T3 hardware deviation."
        )
    else:
        payload["verdict"] = "ENVIRONMENT_COMPATIBLE"
    _write("environment_probe.json", payload)
    return payload


def _check_write() -> bool:
    try:
        ensure_dirs()
        probe = LOG_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# P2 - cache reload probe
# ---------------------------------------------------------------------------
def probe_cache(config: Config, sample: int = 50) -> Dict:
    prefix = CACHE_DIR / config.dataset_name
    cache = FeatureCache(prefix)
    if not cache.exists():
        return {
            "probe": "P2",
            "status": "SKIPPED",
            "reason": f"no feature cache at {prefix}",
        }
    result = reference_check(prefix, config=config, sample=sample)
    result["probe"] = "P2"
    result["status"] = "PASS" if result["ok"] else "FAIL"
    _write("probe_p2_cache.json", result)
    return result


# ---------------------------------------------------------------------------
# P3 - appended-block gradient probe
# ---------------------------------------------------------------------------
def probe_appended_gradients(config: Config) -> Dict:
    device = config.resolve_device()
    build_config = Config.from_dict({**config.to_dict(), "max_layers": 48})
    _peak_reset()
    model = build_model(
        build_config, master_path=None, nl=25,
        load_pretrained=build_config.use_pretrained_gpt2,
    )
    model.to(device)
    # Dropout must be off for the identity comparison, otherwise the two
    # forward passes draw different masks.
    model.set_dropout(0.0)
    model.eval()

    features, mask = _fake_batch(2, 16, device)
    with torch.no_grad():
        model.set_layer_count(25)
        output_25 = model(features, mask).float()
        model.set_layer_count(24)
        output_24 = model(features, mask).float()
    identity_max_abs = float((output_25 - output_24).abs().max().item())

    model.train()
    model.set_layer_count(25)
    loss = model(features, mask).float().sum()
    loss.backward()

    attn_grad = model.appended[0].attn.c_proj.weight.grad
    mlp_grad = model.appended[0].mlp.c_proj.weight.grad
    attn_norm = float(attn_grad.norm().item()) if attn_grad is not None else 0.0
    mlp_norm = float(mlp_grad.norm().item()) if mlp_grad is not None else 0.0

    # Freshly initialised appended blocks must be exact identity mappings:
    # only the zero output projections make this true, so the check is run
    # before the backward pass has a chance to change them.
    result = {
        "probe": "P3",
        "n_appended_blocks_materialised": len(model.appended),
        "identity_max_abs_diff_nl25_vs_nl24": identity_max_abs,
        "identity_ok": identity_max_abs < 1e-3,
        "attn_c_proj_grad_norm": attn_norm,
        "mlp_c_proj_grad_norm": mlp_norm,
        "non_zero_gradients": attn_norm > 0.0 and mlp_norm > 0.0,
        "status": "PASS"
        if (identity_max_abs < 1e-3 and attn_norm > 0.0 and mlp_norm > 0.0)
        else "FAIL",
        "peak_vram_gib": _peak_gib(),
    }
    del model, loss, features, mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _write("probe_p3_appended_grad.json", result)
    return result


# ---------------------------------------------------------------------------
# P4 - candidate isolation probe
# ---------------------------------------------------------------------------
def probe_candidate_isolation(config: Config) -> Dict:
    device = config.resolve_device()
    baseline = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated() / (1024 ** 3)

    build_config = Config.from_dict({**config.to_dict(), "max_layers": 2})
    master_path = config.master_checkpoint if config.master_checkpoint.exists() else None

    def build():
        # Candidate isolation means: same master in, same initial weights out.
        torch.manual_seed(config.seed_init)
        return build_model(
            build_config,
            master_path=master_path,
            nl=2,
            load_pretrained=build_config.use_pretrained_gpt2,
        ).to(device)

    first = build()
    digest_before = state_dict_digest(first)
    features, mask = _fake_batch(2, 16, device)
    loss = first(features, mask).float().sum()
    loss.backward()
    optimizer = torch.optim.SGD(first.parameters(), lr=0.1)
    optimizer.step()
    digest_after = state_dict_digest(first)
    del first, loss, features, mask, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    second = build()
    digest_second = state_dict_digest(second)
    del second
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "probe": "P4",
        "master_checkpoint": str(master_path) if master_path else None,
        "weights_changed_by_training": digest_before != digest_after,
        "reload_digest_matches_initial": digest_before == digest_second,
        "memory_before_gib": baseline,
        "memory_after_unload_gib": after_free,
        "memory_released": (baseline - after_free) < 0.05,
        "status": "PASS"
        if (
            digest_before != digest_after
            and digest_before == digest_second
            and abs(baseline - after_free) < 0.05
        )
        else "FAIL",
    }
    _write("probe_p4_isolation.json", result)
    return result


# ---------------------------------------------------------------------------
# G1 / G2 - forward and backward memory
# ---------------------------------------------------------------------------
def _memory_trial(
    config: Config, nl: int, batch: int, length: int, backward: bool
) -> Dict:
    device = config.resolve_device()
    _peak_reset()
    try:
        model = build_model(
            config, master_path=None, nl=nl,
            load_pretrained=config.use_pretrained_gpt2,
        ).to(device)
        features, mask = _fake_batch(batch, length, device)
        if backward:
            model.train()
            loss = model(features, mask).float().sum()
            loss.backward()
        else:
            model.eval()
            with torch.no_grad():
                model(features, mask)
        peak = _peak_gib()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "nl": nl,
            "batch": batch,
            "length": length,
            "backward": backward,
            "status": "PASS",
            "peak_vram_gib": peak,
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "nl": nl,
            "batch": batch,
            "length": length,
            "backward": backward,
            "status": "OOM" if "out of memory" in str(exc).lower() else "ERROR",
            "error": str(exc)[:500],
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def gate_g1(config: Config) -> Dict:
    result = _memory_trial(config, nl=24, batch=4, length=512, backward=False)
    result["gate"] = "G1"
    _write("gate_g1_forward.json", result)
    return result


def gate_g2(config: Config) -> Dict:
    result = _memory_trial(config, nl=24, batch=4, length=512, backward=True)
    result["gate"] = "G2"
    _write("gate_g2_backward.json", result)
    return result


# ---------------------------------------------------------------------------
# G3 - B_phys calibration at NL=48, s=512
# ---------------------------------------------------------------------------
def gate_g3(config: Config, candidates: Optional[List[int]] = None) -> Dict:
    candidates = candidates or [1, 2, 4, 8, 16]
    trials = [
        _memory_trial(config, nl=48, batch=size, length=512, backward=True)
        for size in candidates
    ]
    successes = [t["batch"] for t in trials if t["status"] == "PASS"]
    result = {
        "gate": "G3",
        "trials": trials,
        "B_phys_max": max(successes) if successes else 0,
        "status": "PASS" if successes else "FAIL",
    }
    _write("gate_g3_bphys.json", result)
    return result


# ---------------------------------------------------------------------------
# G4 - NL=48 stress test
# ---------------------------------------------------------------------------
def gate_g4(config: Config, max_logical_batches: int = 1) -> Dict:
    device = config.resolve_device()
    _peak_reset()
    try:
        model = build_model(
            config, master_path=None, nl=48,
            load_pretrained=config.use_pretrained_gpt2,
        ).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        features, mask = _fake_batch(4, 512, device)
        for _ in range(max_logical_batches):
            optimizer.zero_grad(set_to_none=True)
            loss = model(features, mask).float().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {
            "gate": "G4",
            "status": "PASS",
            "note": "synthetic full step; logical B=4 not B=512",
            "peak_vram_gib": _peak_gib(),
        }
    except BaseException as exc:  # noqa: BLE001
        result = {
            "gate": "G4",
            "status": "OOM" if "out of memory" in str(exc).lower() else "ERROR",
            "error": str(exc)[:500],
            "peak_vram_gib": _peak_gib(),
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write("gate_g4_stress.json", result)
    return result


# ---------------------------------------------------------------------------
# G5 - accumulation equivalence
# ---------------------------------------------------------------------------
def gate_g5(config: Config) -> Dict:
    device = config.resolve_device()
    trial_config = Config.from_dict(
        {**config.to_dict(), "max_layers": 1, "candidate_epochs": 1}
    )
    model = build_model(
        trial_config, master_path=None, nl=1,
        load_pretrained=trial_config.use_pretrained_gpt2,
    ).to(device)
    model.set_dropout(0.0)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    results = []
    for b, b_phys in ((5, 2), (7, 3)):
        features = torch.randn(b, 8, 768, device=device)
        mask = torch.ones(b, 8, device=device)
        labels = torch.randint(0, 2, (b,), device=device)

        model.zero_grad(set_to_none=True)
        logits = model(features, mask)
        loss = criterion(logits, labels)
        loss.backward()
        full_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        model.zero_grad(set_to_none=True)
        for start in range(0, b, b_phys):
            stop = min(start + b_phys, b)
            micro = slice(start, stop)
            logits_micro = model(features[micro], mask[micro])
            micro_loss = criterion(logits_micro, labels[micro])
            (micro_loss * ((stop - start) / b)).backward()
        accum_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        max_diff = 0.0
        for name, full in full_grads.items():
            if name in accum_grads:
                max_diff = max(
                    max_diff, float((full - accum_grads[name]).abs().max().item())
                )
        results.append(
            {
                "logical_batch": b,
                "b_phys": b_phys,
                "max_abs_grad_diff": max_diff,
                "ok": max_diff <= 1e-4,
            }
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "gate": "G5",
        "trials": results,
        "status": "PASS" if all(r["ok"] for r in results) else "FAIL",
        "tolerance": 1e-4,
    }
    _write("gate_g5_accumulation.json", result)
    return result


# ---------------------------------------------------------------------------
# G6 - OOM recovery
# ---------------------------------------------------------------------------
def gate_g6(config: Config) -> Dict:
    if not torch.cuda.is_available():
        result = {"gate": "G6", "status": "SKIPPED", "reason": "CUDA unavailable"}
        _write("gate_g6_oom.json", result)
        return result
    device = config.resolve_device()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    caught = False
    error = None
    try:
        torch.empty(int(free * 1.2), dtype=torch.uint8, device=device)
    except BaseException as exc:  # noqa: BLE001
        caught = True
        error = str(exc)[:300]
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    recovered = True
    try:
        probe = torch.zeros(1024, device=device)
        del probe
    except BaseException:
        recovered = False
    result = {
        "gate": "G6",
        "oom_caught": caught,
        "error": error,
        "recovered": recovered,
        "status": "PASS" if caught and recovered else "FAIL",
    }
    _write("gate_g6_oom.json", result)
    return result


def run_preflight(config: Config, cache_sample: int = 50) -> Dict:
    return {
        "P1": probe_hardware(config),
        "P2": probe_cache(config, sample=cache_sample),
        "P3": probe_appended_gradients(config),
        "P4": probe_candidate_isolation(config),
    }


def run_gates(config: Config, include_stress: bool = False) -> Dict:
    results = {"G1": gate_g1(config), "G2": gate_g2(config), "G5": gate_g5(config)}
    if include_stress:
        results["G3"] = gate_g3(config)
        results["G4"] = gate_g4(config)
        results["G6"] = gate_g6(config)
    _write("gates_summary.json", results)
    return results
