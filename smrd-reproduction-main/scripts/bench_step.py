"""Microbenchmark: forward+backward cost for one candidate shape."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import Config
from src.model import build_model


def bench(nl: int, batch: int, length: int, steps: int, ckpt: bool) -> float:
    config = Config(max_layers=max(nl, 24), device="cuda")
    model = build_model(
        config, master_path=config.master_checkpoint, nl=nl,
        load_pretrained=True,
    ).to("cuda")
    model.train()
    model.enable_gradient_checkpointing(ckpt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    features = torch.randn(batch, length, 768, device="cuda")
    mask = torch.ones(batch, length, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, mask).float()
        loss = logits.sum()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    del model, features, mask, optimizer
    torch.cuda.empty_cache()
    return elapsed / steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args()
    for nl, batch, ckpt in (
        (24, 1, False),
        (24, 1, True),
        (24, 4, False),
        (48, 1, True),
    ):
        avg = bench(nl, batch, args.length, args.steps, ckpt)
        print(
            f"nl={nl} batch={batch} len={args.length} ckpt={ckpt}: "
            f"{avg*1000:.1f} ms/step"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
