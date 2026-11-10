"""Stage 2: tree parsing, batching, BiGCN maths, graph/fusion training, resume.

CPU only; the fusion arm uses the tiny random RoBERTa + stub tokenizer from
the stage 1 tests, so nothing is downloaded.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.propagation import (
    TreeGraph,
    build_tree,
    build_vocab,
    collate_graphs,
    load_graphs,
    parse_rvnn,
    pheme_tree,
    write_cache,
)
from src.propagation_model import (
    BiGCNEncoder,
    PropagationClassifier,
    build_model,
    gcn_propagate,
    load_artifact,
    mean_pool,
)
from src.train_propagation import PropagationConfig, _SimulatedInterrupt, run_propagation
from tests.test_encoder_pipeline import StubTokenizer, tiny_model, toy_data

N_FEAT = 20


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_build_tree_bfs_order_and_orphans():
    parent_of = {"b": "a", "c": "a", "d": "b", "x": "missing", "y": "y"}
    bags = {"a": [(1, 1.0)], "d": [(3, 2.0), (0, 1.0)]}
    tree, reattached = build_tree("a", parent_of, bags)
    assert tree.parent.tolist() == [-1, 0, 0, 1, 0, 0]  # a, b, c, d, then x/y reattached
    assert reattached == 2
    assert tree.words.tolist() == [1, 0, 3]  # bag of d sorted by word id
    assert tree.offsets.tolist() == [0, 1, 1, 1, 3, 3]


def test_tree_validate_rejects_bad_order():
    with pytest.raises(ValueError):
        TreeGraph(np.array([-1, 2, 0]), np.zeros(3, dtype=np.int64),
                  np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)).validate()


def test_parse_rvnn(tmp_path):
    lines = [
        "111\tNone\t1\t2\t9\t1:1 3:2",
        "111\t1\t2\t2\t9\t0:2",
        "111\t2\t3\t2\t9\t4:1",
        "111\t1\t4\t2\t9\t1:1",
        "222\tNone\t1\t0\t9\t2:1",
        "999\tNone\t1\t0\t9\t2:1",  # not requested
    ]
    (tmp_path / "data.TD_RvNN.vol_5000.txt").write_text("\n".join(lines) + "\n")
    trees, info = parse_rvnn(tmp_path, "twitter16", ["111", "222"])
    assert set(trees) == {"111", "222"} and info["reattached_nodes"] == 0
    t = trees["111"]
    assert t.n_nodes == 4 and t.parent[0] == -1
    assert t.depth().max() == 2
    assert trees["222"].n_nodes == 1
    with pytest.raises(ValueError):
        parse_rvnn(tmp_path, "twitter16", ["111", "333"])


def test_pheme_tree_contracts_missing_nodes():
    th = {
        "source": "Police at the scene #ottawa",
        "structure": {"1": {"2": {"3": []}, "4": []}},
        "replies": {
            # "2" has no JSON (deleted) -> "3" attaches to "1"
            "3": {"text": "is this true police", "in_reply_to": "2"},
            "4": {"text": "scene confirmed", "in_reply_to": "1"},
            "5": {"text": "not in structure", "in_reply_to": "4"},
        },
    }
    vocab = build_vocab([["police", "scene"], ["police", "scene", "true"]], size=10, min_df=2)
    assert vocab == ["police", "scene"]
    tree, info = pheme_tree("1", th, {w: k for k, w in enumerate(vocab)})
    assert tree.n_nodes == 4
    assert info["contracted_hops"] == 1
    # node order: 1, then children 3 and 4 (sorted), then 5 under 4
    assert tree.parent.tolist() == [-1, 0, 0, 2]


def test_cache_roundtrip_and_alignment(tmp_path):
    tree, _ = build_tree(0, {1: 0, 2: 1}, {0: [(1, 1.0)], 2: [(4, 3.0)]})
    path = write_cache("twitter16", {"a": tree, "b": tree}, {"source": "test"}, tmp_path / "c.json.gz")
    graphs, meta = load_graphs("twitter16", ["b", "a"], path)
    assert [g.n_nodes for g in graphs] == [3, 3] and len(meta["sha256"]) == 64
    # deterministic bytes -> stable sha256 (resume checks depend on it)
    path2 = write_cache("twitter16", {"a": tree, "b": tree}, {"source": "test"}, tmp_path / "d.json.gz")
    assert path.read_bytes() == path2.read_bytes()
    with pytest.raises(ValueError):
        load_graphs("twitter16", ["a", "zzz"], path)


def test_cli_task_tables_match_finetune():
    import src.finetune as ft
    import src.run_propagation as rp
    from src.propagation_model import ARMS

    assert rp.TASK_TO_ARM == ft.TASK_TO_ARM and rp.DEFAULT_TASK == ft.DEFAULT_TASK
    assert set(rp.ARMS) == set(ARMS)


# ---------------------------------------------------------------------------
# Batching + BiGCN maths
# ---------------------------------------------------------------------------
def toy_tree(n, rng, n_features=N_FEAT, label=0):
    parent = [-1] + [int(rng.integers(0, k)) for k in range(1, n)]
    # words drawn from a label-specific block of 5 ids, so labels are learnable
    bags = {k: [(int(w + 5 * label) % n_features, 1.0) for w in rng.choice(5, 3, replace=False)]
            for k in range(n)}
    tree, _ = build_tree(0, {k: parent[k] for k in range(1, n)}, bags)
    return tree


def test_collate_offsets_edges_roots():
    rng = np.random.default_rng(0)
    a, b = toy_tree(3, rng), toy_tree(4, rng)
    g = collate_graphs([a, b])
    assert g.n_nodes == 7 and g.n_graphs == 2
    assert g.roots.tolist() == [0, 3]
    assert g.graph_of_node.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert g.src.numel() == 2 + 3
    assert torch.all(g.dst[2:] >= 3) and torch.all(g.src[2:] >= 3)  # second tree offset
    assert g.offsets.tolist()[3] == len(a.words)


def test_gcn_propagate_matches_dense():
    torch.manual_seed(0)
    src = torch.tensor([0, 0, 1])
    dst = torch.tensor([1, 2, 3])
    h = torch.randn(4, 5)
    A = torch.eye(4)
    A[dst, src] = 1.0  # row = receiver
    deg = A.sum(1)
    dense = torch.diag(deg.rsqrt()) @ A @ torch.diag(deg.rsqrt()) @ h
    assert torch.allclose(gcn_propagate(h, src, dst), dense, atol=1e-6)


def test_embedding_bag_equals_dense_bow():
    rng = np.random.default_rng(1)
    g = collate_graphs([toy_tree(5, rng), toy_tree(2, rng)])
    emb = torch.nn.EmbeddingBag(N_FEAT, 3, mode="sum")
    X = torch.zeros(g.n_nodes, N_FEAT)
    ends = torch.cat([g.offsets[1:], torch.tensor([g.words.numel()])])
    for k in range(g.n_nodes):
        for j in range(int(g.offsets[k]), int(ends[k])):
            X[k, g.words[j]] += g.counts[j]
    out = emb(g.words, g.offsets, per_sample_weights=g.counts)
    assert torch.allclose(out, X @ emb.weight, atol=1e-6)


def test_mean_pool():
    x = torch.tensor([[1.0], [3.0], [10.0]])
    assert mean_pool(x, torch.tensor([0, 0, 1]), 2).flatten().tolist() == [2.0, 10.0]


def test_bigcn_shapes_edge_drop_and_single_node_trees():
    rng = np.random.default_rng(2)
    g = collate_graphs([toy_tree(6, rng), toy_tree(1, rng), toy_tree(3, rng)])
    enc = BiGCNEncoder(N_FEAT, hidden=8, out=4, dropout=0.5, edge_drop=0.5)
    enc.eval()
    z1, z2 = enc(g), enc(g)
    assert z1.shape == (3, enc.output_dim) == (3, 24)
    assert torch.equal(z1, z2)  # eval: no dropout / edge drop
    enc.train()
    torch.manual_seed(0)
    a = enc(g)
    torch.manual_seed(1)
    b = enc(g)
    assert not torch.allclose(a, b)
    assert torch.isfinite(a).all()


def test_graph_isolation_in_batch():
    """A tree's embedding must not depend on what else is in the batch."""
    rng = np.random.default_rng(3)
    t1, t2, t3 = toy_tree(5, rng), toy_tree(4, rng), toy_tree(7, rng)
    enc = BiGCNEncoder(N_FEAT, hidden=8, out=4).eval()
    alone = enc(collate_graphs([t1]))
    together = enc(collate_graphs([t2, t1, t3]))[1:2]
    assert torch.allclose(alone, together, atol=1e-6)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def toy_graph_data(n=48):
    df, view = toy_data(n)
    rng = np.random.default_rng(5)
    graphs = [toy_tree(int(rng.integers(1, 8)), rng, label=y) for y in df["label"]]
    texts = df["raw_text"].tolist()
    return df, view, texts, graphs


def _cfg(arm="graph", **kw):
    base = dict(arm=arm, n_features=N_FEAT, graph_hidden=8, graph_out=4, max_len=16,
                batch_size=8, micro_batch_size=4, lr_graph=1e-2, lr_text=1e-3,
                schedule="linear", warmup_ratio=0.1, max_epochs=3, patience=3,
                amp="float32", eval_batch_size=16, seed=7, graph_cache_sha256="toy")
    base.update(kw)
    return PropagationConfig(**base)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    text = tiny_model(seed=seed).encoder if cfg.arm == "fusion" else None
    return build_model(cfg.arm, 4, n_features=N_FEAT, hidden=8, out=4, text_encoder=text)


def _run(cfg, path, seed=0, **kw):
    df, view, texts, graphs = toy_graph_data()
    return run_propagation(
        cfg, path, df=df, view=view, texts=texts, graphs=graphs, device="cpu",
        tokenizer=StubTokenizer() if cfg.arm == "fusion" else None,
        model=_model(cfg, seed), graph_meta={"sha256": "toy"}, **kw,
    )


@pytest.mark.parametrize("arm", ["graph", "fusion"])
def test_run_end_to_end(tmp_path, arm):
    df, view, _, _ = toy_graph_data()
    clean = view.test_indices[:-2]
    result = _run(_cfg(arm, max_epochs=2), tmp_path / arm, clean_test_indices=clean)
    assert result["epochs_run"] == 2 and result["arm"] == arm
    assert result["test_full"]["n"] == len(view.test_indices)
    assert result["test_clean"]["n"] == len(clean)
    art = tmp_path / arm / "artifact"
    assert (art / "model.safetensors").exists()
    assert (art / "encoder").exists() == (arm == "fusion")
    assert not (tmp_path / arm / "last.pt").exists()
    again = _run(_cfg(arm, max_epochs=2), tmp_path / arm)
    assert again["finished_at"] == result["finished_at"]


def test_graph_arm_learns_toy_signal(tmp_path):
    result = _run(_cfg("graph", max_epochs=15, patience=15, lr_graph=2e-2, schedule="constant",
                       warmup_ratio=0.0, edge_drop=0.0), tmp_path / "learn")
    assert result["best_val_macro_f1"] > 0.6


def test_artifact_roundtrip_same_predictions(tmp_path):
    _run(_cfg("graph", max_epochs=1), tmp_path / "a")
    model, tok, cfg = load_artifact(tmp_path / "a" / "artifact")
    assert tok is None and cfg["arm"] == "graph"
    _, _, _, graphs = toy_graph_data()
    out = model(collate_graphs(graphs[:5]))
    assert out.shape == (5, 4) and torch.isfinite(out).all()


@pytest.mark.parametrize("arm", ["graph", "fusion"])
def test_resume_after_interrupt_matches(tmp_path, arm):
    full = _run(_cfg(arm), tmp_path / "a")
    with pytest.raises(_SimulatedInterrupt):
        _run(_cfg(arm), tmp_path / "b", _stop_after_epoch=1)
    resumed = _run(_cfg(arm), tmp_path / "b", seed=123)
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]
    for a, b in zip(full["history"], resumed["history"]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-4, abs=1e-6)
        assert a["val_macro_f1"] == pytest.approx(b["val_macro_f1"])


def test_changed_graph_cache_refuses_resume(tmp_path):
    with pytest.raises(_SimulatedInterrupt):
        _run(_cfg(), tmp_path / "r", _stop_after_epoch=1)
    with pytest.raises(RuntimeError):
        _run(_cfg(graph_cache_sha256="other"), tmp_path / "r")


def test_test_split_evaluated_once(tmp_path, monkeypatch):
    import src.train_propagation as tp

    _, view, _, _ = toy_graph_data()
    calls, original = [], tp.evaluate_split

    def spy(model, tok, texts, graphs, indices, *a):
        calls.append("test" if list(indices) == list(view.test_indices) else "val")
        return original(model, tok, texts, graphs, indices, *a)

    monkeypatch.setattr(tp, "evaluate_split", spy)
    _run(_cfg(max_epochs=3), tmp_path / "s")
    assert calls == ["val", "val", "val", "test"]


def test_fusion_requires_text_encoder():
    enc = BiGCNEncoder(N_FEAT, 8, 4)
    with pytest.raises(ValueError):
        PropagationClassifier("fusion", enc, 4, text_encoder=None)
    with pytest.raises(ValueError):
        PropagationClassifier("graph", enc, 4, text_encoder=tiny_model().encoder)
