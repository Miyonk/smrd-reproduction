"""
Source-text classifier: a fine-tuned RoBERTa/BERTweet encoder plus a linear head.

The module has two deliberate boundaries so a serving layer (and the stage 2
graph branch) can reuse it without touching training code:

* :meth:`SourceTextClassifier.encode`   - text ids -> one embedding per thread
* :meth:`SourceTextClassifier.classify` - embedding -> logits

``save_artifact`` / ``load_artifact`` write and read one self-contained,
versioned directory::

    artifact/
      config.json        # labels, encoder name, max_len, normaliser, provenance
      encoder/           # HF save_pretrained of the fine-tuned encoder
      tokenizer/         # HF tokenizer files
      head.safetensors   # classification head weights

``predict_texts`` returns a label *and* a confidence for each input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from src.encoder_preprocess import normalize_tweet, normalizer_config

ARTIFACT_FORMAT_VERSION = 1


class SourceTextClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.n_classes = n_classes

    @property
    def embedding_dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    @classmethod
    def from_pretrained(
        cls, encoder_name: str, n_classes: int, dropout: float = 0.1
    ) -> "SourceTextClassifier":
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False)
        return cls(encoder, n_classes, dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """First-token (``<s>``) hidden state, shape ``(B, hidden)``."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(embedding))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(input_ids, attention_mask))


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------
def save_artifact(
    model: SourceTextClassifier,
    tokenizer,
    out_dir: Path | str,
    *,
    class_names: Sequence[str],
    encoder_name: str,
    max_len: int,
    extra: Optional[Dict] = None,
) -> Path:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
        str(out_dir / "head.safetensors"),
    )
    config = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": "source_text_classifier",
        "encoder_name": encoder_name,
        "pooling": "first_token",
        "n_classes": model.n_classes,
        "class_names": list(class_names),
        "max_len": int(max_len),
        "normalizer": normalizer_config(),
        **(extra or {}),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out_dir


def load_artifact(path: Path | str, device: str | torch.device = "cpu"):
    """Return ``(model, tokenizer, config)`` from a saved artifact directory."""
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"unsupported artifact format in {path}")
    encoder = AutoModel.from_pretrained(path / "encoder", add_pooling_layer=False)
    model = SourceTextClassifier(encoder, int(config["n_classes"]))
    model.head.load_state_dict(load_file(str(path / "head.safetensors")))
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    model.to(device).eval()
    return model, tokenizer, config


@torch.no_grad()
def predict_texts(
    model: SourceTextClassifier,
    tokenizer,
    config: Dict,
    texts: Sequence[str],
    batch_size: int = 32,
) -> List[Dict]:
    """Normalise, encode and classify raw tweets; returns label + confidence."""
    model.eval()
    device = next(model.parameters()).device
    names = config["class_names"]
    results: List[Dict] = []
    for start in range(0, len(texts), batch_size):
        batch = [normalize_tweet(t) for t in texts[start : start + batch_size]]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=int(config["max_len"]),
            padding=True,
            return_tensors="pt",
        )
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        for row in probs:
            idx = int(row.argmax())
            results.append(
                {
                    "label": names[idx],
                    "label_id": idx,
                    "confidence": float(row[idx]),
                    "probabilities": {n: float(p) for n, p in zip(names, row.tolist())},
                }
            )
    return results
