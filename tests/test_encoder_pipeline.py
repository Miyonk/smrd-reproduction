"""Stage 1 encoder path: normalisation, dedup, model, training loop, resume.

Everything runs on CPU with a tiny randomly initialised RoBERTa and a stub
tokenizer, so no weights are downloaded.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data import ArmView
from src.dedup import DupClusters, find_clusters, match_key
from src.encoder_model import SourceTextClassifier, load_artifact, predict_texts
from src.encoder_preprocess import normalize_tweet
from src.finetune import (
    EarlyStopper,
    FinetuneConfig,
    _SimulatedInterrupt,
    compute_class_weights,
    run_finetune,
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_normalize_keeps_natural_text():
    text = "RT @bbc: Police &amp; army are NOT at the scene http://t.co/x #Ottawa"
    out = normalize_tweet(text)
    assert out == "RT @USER: Police & army are NOT at the scene URL #Ottawa"
    # no stopword removal / lemmatisation / Light-10
    assert "are NOT at the" in out


def test_normalize_handles_non_strings():
    assert normalize_tweet(None) == ""
    assert normalize_tweet(float("nan")) == ""


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
def test_match_key_ignores_urls_mentions_rt():
    a = match_key("RT @a: Soldier shot at war memorial http://t.co/abc")
    b = match_key("soldier shot at war memorial URL")
    assert a == b


def test_find_clusters_groups_near_duplicates():
    texts = [
        "breaking: soldier shot at national war memorial in ottawa URL",
        "BREAKING: soldier shot at national war memorial in Ottawa http://t.co/zz",
        "completely unrelated tweet about the weather today",
        "",
        "",
    ]
    clusters = find_clusters(texts, threshold=0.9)
    assert clusters[0] == clusters[1] == 0
    assert clusters[2] == 2
    # empty keys never merge
    assert clusters[3] == 3 and clusters[4] == 4


def test_clean_test_and_groups(tmp_path):
    c = DupClusters("toy", 0.9, ["a", "b", "c", "d"], [0, 0, 2, 3])
    assert c.clean_test_indices(train=[0], val=[2], test=[1, 3]) == [3]
    leak = c.leakage(train=[0], val=[2], test=[1, 3])
    assert leak["test_rows_with_duplicate_in_train_or_val"] == 1
    assert c.groups([1, 3]) == [0, 3]
    assert c.cross_group_clusters(["e1", "e2", "e1", "e1"]) == [[0, 1]]
    path = c.save(tmp_path / "toy.json", labels=[0, 1, 0, 0])
    raw = json.loads(path.read_text())
    assert raw["mixed_label_clusters"] == [[0, 1]]
    assert DupClusters.load(path, expected_tweet_ids=["a", "b", "c", "d"]).cluster_of == [0, 0, 2, 3]
    with pytest.raises(ValueError):
        DupClusters.load(path, expected_tweet_ids=["a", "b", "d", "c"])


# ---------------------------------------------------------------------------
# Tiny model + stub tokenizer
# ---------------------------------------------------------------------------
class StubTokenizer:
    """Hashes whitespace tokens into a small vocab; mimics the HF call API."""

    vocab_size = 101

    def __call__(self, texts, truncation=True, max_length=16, padding=True, return_tensors="pt"):
        rows = []
        for t in texts:
            ids = [0] + [3 + (hash(w) % 97) for w in t.split()][: max_length - 2] + [2]
            rows.append(ids)
        width = max(len(r) for r in rows)
        input_ids = torch.ones((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "stub_tokenizer.json").write_text("{}")


def tiny_model(n_classes=4, seed=0):
    from transformers import RobertaConfig, RobertaModel

    torch.manual_seed(seed)
    cfg = RobertaConfig(
        vocab_size=StubTokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=40,
        pad_token_id=1,
    )
    return SourceTextClassifier(RobertaModel(cfg, add_pooling_layer=False), n_classes, 0.1)


def toy_data(n=48, n_classes=4):
    words = ["alpha", "bravo", "charlie", "delta"]
    texts, labels = [], []
    for i in range(n):
        y = i % n_classes
        texts.append(f"{words[y]} story number {i} {words[y]}")
        labels.append(y)
    df = pd.DataFrame(
        {"tweet_id": [str(1000 + i) for i in range(n)], "raw_text": texts, "label": labels}
    )
    idx = list(range(n))
    view = ArmView(
        arm="arm_a",
        dataset="twitter16",
        train_indices=idx[: n // 2],
        val_indices=idx[n // 2 : 3 * n // 4],
        test_indices=idx[3 * n // 4 :],
        labels={i: y for i, y in enumerate(labels)},
        n_classes=n_classes,
        class_names=["non-rumour", "false", "true", "unverified"],
    )
    return df, view


def test_encode_classify_boundary():
    model = tiny_model()
    tok = StubTokenizer()
    enc = tok(["a b c", "d"])
    emb = model.encode(enc["input_ids"], enc["attention_mask"])
    assert emb.shape == (2, model.embedding_dim)
    assert model.classify(emb).shape == (2, 4)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def test_early_stopper():
    s = EarlyStopper(patience=2)
    assert s.update(0.5, 1) and not s.should_stop
    assert not s.update(0.4, 2)
    assert not s.update(0.5, 3)  # equal is not an improvement
    assert s.should_stop and s.best_epoch == 1


def test_class_weights_modes():
    assert compute_class_weights([0, 1, 0, 1], 2, "auto") is None
    w = compute_class_weights([0] * 8 + [1] * 2, 2, "auto")
    assert w is not None and w[1] > w[0]
    assert compute_class_weights([0] * 8 + [1] * 2, 2, "off") is None


def _cfg(**kw):
    base = dict(max_len=16, batch_size=8, micro_batch_size=4, lr=1e-3, max_epochs=3,
                patience=3, amp="float32", eval_batch_size=16, seed=7)
    base.update(kw)
    return FinetuneConfig(**base)


def test_run_finetune_end_to_end(tmp_path):
    df, view = toy_data()
    clean = view.test_indices[:-3]
    result = run_finetune(
        _cfg(max_epochs=2), tmp_path / "run", device="cpu", clean_test_indices=clean,
        df=df, view=view, tokenizer=StubTokenizer(), model=tiny_model(),
    )
    assert result["epochs_run"] == 2
    assert result["test_full"]["n"] == len(view.test_indices)
    assert result["test_clean"]["n"] == len(clean)
    assert len(result["test_full"]["accuracy_wilson_95"]) == 2
    assert all("epoch_seconds" in h for h in result["history"])
    assert (tmp_path / "run" / "artifact" / "config.json").exists()
    assert (tmp_path / "run" / "artifact" / "head.safetensors").exists()
    # bulky resume files are cleaned up after a finished run
    assert not (tmp_path / "run" / "last.pt").exists()
    # re-running a finished run is a no-op returning the stored result
    again = run_finetune(
        _cfg(max_epochs=2), tmp_path / "run", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(),
    )
    assert again["finished_at"] == result["finished_at"]


def test_resume_after_interrupt_matches_uninterrupted(tmp_path):
    df, view = toy_data()
    full = run_finetune(
        _cfg(), tmp_path / "a", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(),
    )
    with pytest.raises(_SimulatedInterrupt):
        run_finetune(
            _cfg(), tmp_path / "b", device="cpu", df=df, view=view,
            tokenizer=StubTokenizer(), model=tiny_model(), _stop_after_epoch=1,
        )
    assert (tmp_path / "b" / "last.pt").exists()
    resumed = run_finetune(
        _cfg(), tmp_path / "b", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(seed=123),  # weights overwritten by resume
    )
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]
    for a, b in zip(full["history"], resumed["history"]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-4, abs=1e-6)
        assert a["val_macro_f1"] == pytest.approx(b["val_macro_f1"])
    assert full["test_full"]["accuracy"] == pytest.approx(resumed["test_full"]["accuracy"])


def test_config_mismatch_refuses_resume(tmp_path):
    df, view = toy_data()
    with pytest.raises(_SimulatedInterrupt):
        run_finetune(
            _cfg(), tmp_path / "r", device="cpu", df=df, view=view,
            tokenizer=StubTokenizer(), model=tiny_model(), _stop_after_epoch=1,
        )
    with pytest.raises(RuntimeError):
        run_finetune(
            _cfg(lr=5e-4), tmp_path / "r", device="cpu", df=df, view=view,
            tokenizer=StubTokenizer(), model=tiny_model(),
        )


def test_test_split_never_used_for_selection(tmp_path, monkeypatch):
    """evaluate_split must see test indices exactly once, after training."""
    import src.finetune as ft

    df, view = toy_data()
    calls = []
    original = ft.evaluate_split

    def spy(model, tok, texts, indices, labels, cfg, device, amp):
        calls.append("test" if list(indices) == list(view.test_indices) else "val")
        return original(model, tok, texts, indices, labels, cfg, device, amp)

    monkeypatch.setattr(ft, "evaluate_split", spy)
    run_finetune(
        _cfg(max_epochs=3), tmp_path / "s", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(),
    )
    assert calls == ["val", "val", "val", "test"]


def test_light_run_keeps_probabilities_instead_of_the_model(tmp_path):
    """light=True trains exactly like a normal run but leaves only result.json
    and probs.json (validation + test probabilities of the best weights)."""
    import numpy as np

    from src.stack import read_probs

    df, view = toy_data()
    full = run_finetune(
        _cfg(), tmp_path / "full", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(),
    )
    light = run_finetune(
        _cfg(), tmp_path / "light", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(), light=True,
    )
    assert sorted(p.name for p in (tmp_path / "light").iterdir()) == ["probs.json", "result.json"]
    assert light["artifact"] is None
    assert light["best_epoch"] == full["best_epoch"]
    for a, b in zip(full["history"], light["history"]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-4, abs=1e-6)
        assert a["val_macro_f1"] == pytest.approx(b["val_macro_f1"])
    assert [p["pred"] for p in light["test_predictions"]] == [p["pred"] for p in full["test_predictions"]]

    val_p, test_p = read_probs(tmp_path / "light" / "probs.json", view.val_indices, view.test_indices, 4)
    assert np.allclose(test_p.max(1), [p["confidence"] for p in light["test_predictions"]], atol=1e-5)
    # the stored validation probabilities are those of the best epoch
    best = light["history"][light["best_epoch"] - 1]
    val_loss = -np.log(val_p[np.arange(len(val_p)), view.val_labels]).mean()
    assert val_loss == pytest.approx(best["val_loss"], abs=1e-3)
    # a finished light run is reused, not retrained
    again = run_finetune(
        _cfg(), tmp_path / "light", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(), light=True,
    )
    assert again["finished_at"] == light["finished_at"]


def test_light_run_scores_validation_once_per_epoch_and_test_once(tmp_path, monkeypatch):
    import src.finetune as ft

    df, view = toy_data()
    calls = []
    original = ft.evaluate_split

    def spy(model, tok, texts, indices, labels, cfg, device, amp):
        calls.append("test" if list(indices) == list(view.test_indices) else "val")
        return original(model, tok, texts, indices, labels, cfg, device, amp)

    monkeypatch.setattr(ft, "evaluate_split", spy)
    run_finetune(
        _cfg(max_epochs=3), tmp_path / "s", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(), light=True,
    )
    assert calls == ["val", "val", "val", "test"]


def test_artifact_roundtrip_predicts_with_confidence(tmp_path):
    """Saved artifact reloads without training code and returns label+confidence."""
    df, view = toy_data()
    run_finetune(
        _cfg(max_epochs=1), tmp_path / "art", device="cpu", df=df, view=view,
        tokenizer=StubTokenizer(), model=tiny_model(),
    )
    art = tmp_path / "art" / "artifact"
    cfg = json.loads((art / "config.json").read_text())
    assert cfg["class_names"] == view.class_names and cfg["normalizer"]["name"] == "normalize_tweet"
    # The stub tokenizer can't be reloaded by AutoTokenizer, so reload the
    # encoder + head directly and reuse the stub for prediction.
    from safetensors.torch import load_file
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(art / "encoder", add_pooling_layer=False)
    model = SourceTextClassifier(encoder, cfg["n_classes"])
    model.head.load_state_dict(load_file(str(art / "head.safetensors")))
    out = predict_texts(model, StubTokenizer(), cfg, ["alpha story", "delta thing"])
    assert len(out) == 2
    assert all(0.0 <= o["confidence"] <= 1.0 and o["label"] in view.class_names for o in out)
