"""
Stage 2 models: a Bi-directional GCN over the propagation tree (BiGCN; Bian et
al., AAAI 2020) and its fusion with the stage 1 source-text encoder.

BiGCN, per direction (top-down = parent->child, bottom-up = child->parent)::

    h1 = GCN1(X)                                   # X: node bag-of-words
    h2 = relu(GCN2(dropout(relu([h1 ; X_root]))))  # root feature enhancement
    z  = mean_nodes([h2 ; h1_root])

and the thread embedding is ``[z_TD ; z_BU]`` (``2 * (hidden + out)`` dims).
During training each direction drops ``edge_drop`` of its edges independently
(DropEdge).  GCN layers use symmetric normalisation with self-loops, as in
``torch_geometric.nn.GCNConv``; they are written in plain PyTorch so no extra
dependency is needed.  Sparse inputs go through ``nn.EmbeddingBag`` (sum of
word rows weighted by counts), which equals ``X @ W`` for bag-of-words ``X``.

Deviation from the reference implementation: dropout on the root-feature part
of the GCN2 input is applied after projection (on ``X_root @ W``) rather than
on the raw 5000-dim vector, which would need a dense copy per node.

Arms
----
* ``graph``  - BiGCN alone (comparable with published BiGCN numbers).
* ``fusion`` - ``[text_embedding ; graph_embedding] -> linear``; the text side
  is a RoBERTa encoder fine-tuned end to end, pooled like stage 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.propagation import GraphBatch

ARMS = ("graph", "fusion")
ARTIFACT_FORMAT_VERSION = 1


def gcn_propagate(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """``D^-1/2 (A + I) D^-1/2 h`` for messages ``src -> dst`` (in-degree incl. self-loop)."""
    n = h.shape[0]
    deg = torch.ones(n, device=h.device, dtype=h.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
    inv_sqrt = deg.rsqrt()
    out = h * (inv_sqrt * inv_sqrt).unsqueeze(1)
    norm = (inv_sqrt[src] * inv_sqrt[dst]).unsqueeze(1)
    return out.index_add(0, dst, h[src] * norm)


def mean_pool(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    total = torch.zeros(n, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, index, x)
    count = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(
        0, index, torch.ones_like(index, dtype=x.dtype)
    )
    return total / count.clamp(min=1).unsqueeze(1)


class _Direction(nn.Module):
    def __init__(self, n_features: int, hidden: int, out: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.EmbeddingBag(n_features, hidden, mode="sum")
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2_hidden = nn.Linear(hidden, out, bias=False)
        self.w2_root = nn.EmbeddingBag(n_features, out, mode="sum")
        self.b2 = nn.Parameter(torch.zeros(out))
        self.dropout = dropout
        for emb in (self.w1, self.w2_root):  # glorot, like GCNConv
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.w2_hidden.weight)

    def forward(self, g: GraphBatch, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h1 = gcn_propagate(self.w1(g.words, g.offsets, per_sample_weights=g.counts), src, dst) + self.b1
        root_of_node = g.roots[g.graph_of_node]
        root_proj = self.w2_root(g.words, g.offsets, per_sample_weights=g.counts)[root_of_node]
        hid = F.dropout(F.relu(h1), self.dropout, self.training)
        root_proj = F.dropout(root_proj, self.dropout, self.training)
        h2 = F.relu(gcn_propagate(self.w2_hidden(hid) + root_proj, src, dst) + self.b2)
        z = torch.cat([h2, h1[root_of_node]], dim=1)
        return mean_pool(z, g.graph_of_node, g.n_graphs)


class BiGCNEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        out: int = 64,
        dropout: float = 0.5,
        edge_drop: float = 0.2,
    ) -> None:
        super().__init__()
        self.td = _Direction(n_features, hidden, out, dropout)
        self.bu = _Direction(n_features, hidden, out, dropout)
        self.edge_drop = edge_drop
        self.output_dim = 2 * (hidden + out)

    def _drop(self, src: torch.Tensor, dst: torch.Tensor):
        if not self.training or self.edge_drop <= 0 or src.numel() == 0:
            return src, dst
        keep = torch.rand(src.shape[0], device=src.device) >= self.edge_drop
        return src[keep], dst[keep]

    def forward(self, g: GraphBatch) -> torch.Tensor:
        td = self.td(g, *self._drop(g.src, g.dst))
        bu_dst, bu_src = self._drop(g.src, g.dst)  # reverse direction, independent mask
        bu = self.bu(g, bu_src, bu_dst)
        return torch.cat([td, bu], dim=1)


class PropagationClassifier(nn.Module):
    """``graph`` arm: BiGCN -> linear.  ``fusion`` arm: [text ; BiGCN] -> linear."""

    def __init__(
        self,
        arm: str,
        graph: BiGCNEncoder,
        n_classes: int,
        text_encoder: Optional[nn.Module] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if (arm == "fusion") != (text_encoder is not None):
            raise ValueError("the fusion arm needs a text encoder; the graph arm must not have one")
        self.arm = arm
        self.graph = graph
        self.text_encoder = text_encoder
        text_dim = int(text_encoder.config.hidden_size) if text_encoder is not None else 0
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(text_dim + graph.output_dim, n_classes)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.n_classes = n_classes

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def encode_graph(self, g: GraphBatch) -> torch.Tensor:
        # sparse bag sums and index_add are kept in float32 under autocast
        with torch.autocast(device_type=g.words.device.type, enabled=False):
            return self.graph(g)

    def forward(
        self,
        g: GraphBatch,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encode_graph(g)
        if self.text_encoder is not None:
            t = self.dropout(self.encode_text(input_ids, attention_mask)).float()
            z = torch.cat([t, z], dim=1)
        return self.head(z)

    def graph_state_dict(self) -> Dict[str, torch.Tensor]:
        """Everything except the HF text encoder (saved separately)."""
        return {
            k: v for k, v in self.state_dict().items() if not k.startswith("text_encoder.")
        }


def build_model(
    arm: str,
    n_classes: int,
    *,
    n_features: int,
    hidden: int = 64,
    out: int = 64,
    graph_dropout: float = 0.5,
    edge_drop: float = 0.2,
    dropout: float = 0.1,
    encoder_name: Optional[str] = None,
    text_encoder: Optional[nn.Module] = None,
) -> PropagationClassifier:
    graph = BiGCNEncoder(n_features, hidden, out, graph_dropout, edge_drop)
    if arm == "fusion" and text_encoder is None:
        from transformers import AutoModel

        text_encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False)
    return PropagationClassifier(arm, graph, n_classes, text_encoder if arm == "fusion" else None, dropout)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------
def save_artifact(
    model: PropagationClassifier,
    tokenizer,
    out_dir: Path | str,
    *,
    model_config: Dict,
    class_names: Sequence[str],
    extra: Optional[Dict] = None,
) -> Path:
    """``config.json`` + ``model.safetensors`` (graph branch and head), plus
    ``encoder/`` and ``tokenizer/`` for the fusion arm."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.graph_state_dict().items()},
        str(out_dir / "model.safetensors"),
    )
    if model.text_encoder is not None:
        model.text_encoder.save_pretrained(out_dir / "encoder")
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir / "tokenizer")
    config = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": f"propagation_{model.arm}",
        "arm": model.arm,
        "n_classes": model.n_classes,
        "class_names": list(class_names),
        "model_config": model_config,
        **(extra or {}),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out_dir


def load_artifact(path: Path | str, device: str | torch.device = "cpu"):
    """Return ``(model, tokenizer_or_None, config)``."""
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(f"unsupported artifact format in {path}")
    mc = config["model_config"]
    text_encoder, tokenizer = None, None
    if config["arm"] == "fusion":
        from transformers import AutoModel, AutoTokenizer

        text_encoder = AutoModel.from_pretrained(path / "encoder", add_pooling_layer=False)
        if (path / "tokenizer").exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
            except Exception:  # pragma: no cover - stub tokenizers in tests
                tokenizer = None
    model = build_model(
        config["arm"],
        int(config["n_classes"]),
        n_features=mc["n_features"],
        hidden=mc["hidden"],
        out=mc["out"],
        graph_dropout=mc["graph_dropout"],
        edge_drop=mc["edge_drop"],
        dropout=mc["dropout"],
        text_encoder=text_encoder,
    )
    missing, unexpected = model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    missing = [k for k in missing if not k.startswith("text_encoder.")]
    if missing or unexpected:
        raise ValueError(f"artifact mismatch: missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    return model, tokenizer, config
