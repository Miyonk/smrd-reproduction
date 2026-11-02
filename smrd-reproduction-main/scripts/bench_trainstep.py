"""Replicate the trainer micro-step to find the real per-step cost."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import Config
from src.model import build_model


def run(label, batch, length, ckpt, use_8bit, amp, steps=20):
    config = Config(device="cuda", max_layers=24, use_8bit_optimizer=use_8bit)
    model = build_model(
        config, master_path=config.master_checkpoint, nl=24, load_pretrained=True
    ).to("cuda").train()
    model.enable_gradient_checkpointing(ckpt)
    params = [p for p in model.parameters() if p.requires_grad]
    if use_8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(params, lr=1e-4)
    features = torch.randn(batch, length, 768, device="cuda")
    mask = torch.ones(batch, length, device="cuda")
    dtype = torch.bfloat16 if amp else torch.float32

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=amp):
            logits = model(features, mask)
        loss = torch.nn.functional.cross_entropy(
            logits.float(), torch.zeros(batch, dtype=torch.long, device="cuda")
        )
        loss.backward()
        optimizer.step()

    step()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / steps
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{label}: {elapsed*1000:.1f} ms/step, peak {peak:.2f} GiB")
    del model, optimizer, features, mask
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> int:
    run("b1 ckpt=False 8bit=False amp=False", 1, 24, False, False, False)
    run("b1 ckpt=False 8bit=False amp=True ", 1, 24, False, False, True)
    run("b1 ckpt=False 8bit=True  amp=True ", 1, 24, False, True, True)
    run("b1 ckpt=True  8bit=True  amp=True ", 1, 24, True, True, True)
    run("b8 ckpt=False 8bit=True  amp=True ", 8, 24, False, True, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
