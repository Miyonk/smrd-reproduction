"""Memory/time sweep for micro-batch sizes on the local GPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bitsandbytes as bnb
import torch

from src.config import Config
from src.model import build_model


def trial(batch, length, ckpt):
    config = Config(device="cuda", max_layers=24, use_8bit_optimizer=True)
    model = build_model(
        config, master_path=config.master_checkpoint, nl=24, load_pretrained=True
    ).to("cuda").train()
    model.enable_gradient_checkpointing(ckpt)
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    features = torch.randn(batch, length, 768, device="cuda")
    mask = torch.ones(batch, length, device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(features, mask)
        loss = torch.nn.functional.cross_entropy(
            logits.float(), torch.zeros(batch, dtype=torch.long, device="cuda")
        )
        loss.backward()
        optimizer.step()

    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 5
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"batch={batch:2d} len={length:3d} ckpt={ckpt!s:5s}: "
        f"{elapsed*1000:7.1f} ms/step  {elapsed*1000/batch:6.1f} ms/example  "
        f"peak={peak:.2f} GiB"
    )
    del model, optimizer, features, mask
    torch.cuda.empty_cache()


def main() -> int:
    for length in (32, 64, 128):
        for batch in (1, 2, 4, 8):
            trial(batch, length, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
