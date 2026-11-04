"""Compare stock GPT2Model vs the custom dynamic loop."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import GPT2Config, GPT2Model

from src.config import Config
from src.model import build_model


def timeit(label, fn, steps=10):
    fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    print(f"{label}: {(time.time() - start) / steps * 1000:.1f} ms")


def main() -> int:
    torch.manual_seed(0)
    device = "cuda"
    batch, length = 4, 32

    stock = GPT2Model(GPT2Config.from_pretrained("gpt2-medium")).to(device).train()
    ids = torch.randint(0, 1000, (batch, length), device=device)
    with torch.no_grad():
        timeit("stock fwd", lambda: stock(ids))

    config = Config(device="cuda", max_layers=24)
    model = build_model(
        config, master_path=config.master_checkpoint, nl=24, load_pretrained=True
    ).to(device).train()
    features = torch.randn(batch, length, 768, device=device)
    mask = torch.ones(batch, length, device=device)

    with torch.no_grad():
        timeit("custom fwd", lambda: model(features, mask))

    def step():
        model.zero_grad(set_to_none=True)
        model(features, mask).float().sum().backward()

    timeit("custom fwd+bwd", step, steps=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
