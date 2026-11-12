"""Lexical arm (TF-IDF + logistic regression) and the stacked arm."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.cv import Folds
from src.data import ArmView, build_arm_view, load_dataset
from src.lexical import PROBS_FILE, LexicalConfig, run_lexical, thread_bags, write_probs
from src.propagation import build_tree
from src.stack import choose_weights, combine, read_probs, simplex_grid

WORDS = [["alpha", "apple"], ["bravo", "banana"], ["charlie", "cherry"], ["delta", "date"]]


def toy(n=80):
    """Class k tweets use WORDS[k]; the thread bag carries the class in word id k."""
    rng = np.random.default_rng(0)
    labels = [i % 4 for i in range(n)]
    texts = [f"{WORDS[y][int(rng.integers(2))]} filler{int(rng.integers(5))} news" for y in labels]
    graphs = [
        build_tree(0, {1: 0, 2: 0}, {0: [(y, 1.0)], 1: [(4 + int(rng.integers(6)), 1.0)], 2: [(y, 2.0)]}, 10)[0]
        for y in labels
    ]
    df = pd.DataFrame({"tweet_id": [str(1000 + i) for i in range(n)], "raw_text": texts, "label": labels})
    rows = list(range(n))
    view = ArmView("arm_a", "toy", rows[:48], rows[48:60], rows[60:], dict(zip(rows, labels)), 4, list("abcd"))
    return df, view, texts, graphs


def test_thread_bags_sum_over_nodes():
    _, _, _, graphs = toy(8)
    bags = thread_bags(graphs)
    assert bags.shape == (8, 10)
    assert bags[1, 1] == 3.0 and bags[1].sum() == 4.0  # root 1 + reply 2 on the class word, 1 filler


def test_lexical_end_to_end(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    df, view, texts, graphs = toy()
    fitted, scored = [], []
    fit, proba = LogisticRegression.fit, LogisticRegression.predict_proba
    monkeypatch.setattr(LogisticRegression, "fit", lambda self, X, y, **kw: (fitted.append(X.shape[0]), fit(self, X, y, **kw))[1])
    monkeypatch.setattr(LogisticRegression, "predict_proba", lambda self, X: (scored.append(X.shape[0]), proba(self, X))[1])

    cfg = LexicalConfig(dataset="toy", c_grid=(1.0, 10.0))
    result = run_lexical(cfg, tmp_path / "lex", df=df, view=view, texts=texts, graphs=graphs, split_source="toy split")
    assert fitted == [48, 48]  # training rows only: validation rows are never trained on
    assert scored == [12, 12, 20]  # validation once per C, test exactly once
    assert result["test_full"]["accuracy"] == 1.0 and result["best_val_macro_f1"] == 1.0
    assert result["selected"]["C"] == 1.0  # tie keeps the first (strongest regularisation)
    assert result["split"] == {"source": "toy split", "train": 48, "val": 12, "test": 20}
    assert [p["row"] for p in result["test_predictions"]] == view.test_indices

    val_p, test_p = read_probs(tmp_path / "lex" / PROBS_FILE, view.val_indices, view.test_indices, 4)
    assert val_p.shape == (12, 4) and test_p.argmax(1).tolist() == view.test_labels
    with pytest.raises(ValueError):
        read_probs(tmp_path / "lex" / PROBS_FILE, view.test_indices, view.val_indices, 4)

    again = run_lexical(cfg, tmp_path / "lex", df=df, view=view, texts=texts, graphs=graphs)
    assert again["finished_at"] == result["finished_at"] and len(fitted) == 2
    with pytest.raises(RuntimeError):
        run_lexical(LexicalConfig(dataset="toy", c_grid=(3.0,)), tmp_path / "lex", df=df, view=view, texts=texts, graphs=graphs)


def test_lexical_rejects_overlapping_rows(tmp_path):
    df, view, texts, graphs = toy()
    view.val_indices = view.train_indices[:5]
    with pytest.raises(ValueError):
        run_lexical(LexicalConfig(dataset="toy"), tmp_path / "x", df=df, view=view, texts=texts, graphs=graphs)


def test_lexical_vocabulary_comes_from_training_rows_only():
    from src.lexical import build_features

    texts = ["alpha news", "bravo news", "zulu secret"]
    bags = thread_bags([build_tree(0, {}, {0: [(k, 1.0)]}, 5)[0] for k in range(3)])
    X = build_features(LexicalConfig(), texts, bags, [0, 1])
    Y = build_features(LexicalConfig(), ["alpha news", "bravo news", "yankee hidden"], bags, [0, 1])
    assert X.shape == Y.shape  # words that only occur outside training add no features
    assert (X[:2] != Y[:2]).nnz == 0  # and do not change the training rows


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------
def test_simplex_grid():
    assert len(simplex_grid(2)) == 11 and len(simplex_grid(3)) == 66
    assert all(abs(sum(w) - 1) < 1e-9 for w in simplex_grid(3))
    assert simplex_grid(1) == [(1.0,)]


def _onehot(y, k=4, conf=0.9):
    conf = np.broadcast_to(np.asarray(conf, dtype=float), (len(y),))
    p = np.repeat(((1 - conf) / (k - 1))[:, None], k, axis=1)
    p[np.arange(len(y)), y] = conf
    return p


def test_choose_weights_prefers_the_useful_member():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 200)
    good, noise = _onehot(y), rng.dirichlet(np.ones(4), 200)
    best = choose_weights([noise, good], y, 4)
    assert best["weights"][1] > best["weights"][0] and best["val_macro_f1"] == 1.0
    # two members that are right on different halves: the mix beats either alone
    wrong = (y + 1) % 4
    a = _onehot(np.where(np.arange(200) < 100, y, wrong), conf=np.where(np.arange(200) < 100, 0.9, 0.4))
    b = _onehot(np.where(np.arange(200) < 100, wrong, y), conf=np.where(np.arange(200) < 100, 0.4, 0.9))
    mixed = choose_weights([a, b], y, 4)
    assert 0 < mixed["weights"][0] < 1 and mixed["val_accuracy"] == 1.0
    assert choose_weights([good, good], y, 4)["weights"] == [0.5, 0.5]  # ties -> most even


def test_combine_normalises_and_checks_lengths():
    p = combine([_onehot(np.array([0, 1])), _onehot(np.array([1, 1]))], [0.5, 0.5])
    assert np.allclose(p.sum(1), 1.0) and p.argmax(1).tolist()[1] == 1
    with pytest.raises(ValueError):
        combine([p], [0.5, 0.5])


def test_read_probs_validates(tmp_path):
    path = tmp_path / "p.json"
    write_probs(path, {"val": ([1, 2], _onehot(np.array([0, 1]))), "test": ([3], np.array([[0.5, 0.2, 0.2, 0.2]]))})
    with pytest.raises(ValueError):
        read_probs(path, [1, 2], [3], 4)  # test row does not sum to 1
    write_probs(path, {"val": ([1, 2], _onehot(np.array([0, 1]))), "test": ([3], _onehot(np.array([2])))})
    v, t = read_probs(path, [1, 2], [3], 4)
    assert v.shape == (2, 4) and t.argmax(1).tolist() == [2]
    with pytest.raises(ValueError):
        read_probs(path, [1, 2], [3], 2)


def _fake_members(root, arms):
    """Fake member runs on the real Twitter16 folds: 'lexical' is right on even
    rows, any other arm on odd rows, each confident only where it is right."""
    df = load_dataset("twitter16")
    ids = df["tweet_id"].astype(str).tolist()
    base = build_arm_view(Config(task_arm="arm_a", dataset_name="twitter16"), df=df)
    folds = Folds.load("twitter16", ids)

    def probs(arm, rows):
        y = np.array([base.labels[i] for i in rows])
        right = (np.array(rows) % 2 == 0) == (arm == "lexical")
        return _onehot(np.where(right, y, (y + 1) % 4), conf=np.where(right, 0.9, 0.4))

    for arm in arms:
        for k in range(5):
            view = folds.view(base, k)
            d = root / f"twitter16_4way_{arm}" / f"fold{k}"
            d.mkdir(parents=True)
            pt = probs(arm, view.test_indices)
            write_probs(d / PROBS_FILE, {"val": (view.val_indices, probs(arm, view.val_indices)),
                                         "test": (view.test_indices, pt)})
            preds = pt.argmax(1).tolist()
            acc = float(np.mean(np.array(preds) == np.array(view.test_labels)))
            (d / "result.json").write_text(json.dumps({
                "split": {"source": folds.split_source(k)},
                "test_full": {"accuracy": acc, "macro_f1": acc, "n": len(preds)},
                "best_val_macro_f1": 0.5, "best_epoch": 1,
                "test_predictions": [{"row": i, "label": base.labels[i], "pred": p}
                                     for i, p in zip(view.test_indices, preds)],
            }))
    return df


def test_stack_and_compare_on_real_folds(tmp_path):
    from src.run_cv import cmd_compare, cmd_stack, format_comparison

    df = _fake_members(tmp_path, ("lexical", "graph"))
    args = SimpleNamespace(dataset="twitter16", members=["lexical", "graph"], folds=list(range(5)),
                           device="cpu", ckpt_root=str(tmp_path), tag=None, fresh=False)
    out = cmd_stack(args)
    assert [f["test_accuracy"] for f in out["folds"]] == [1.0] * 5
    assert all(0 < f["weights"]["lexical"] < 1 for f in out["folds"])
    assert cmd_stack(args) == out  # finished folds are reused
    args.members = ["graph", "lexical"]
    with pytest.raises(RuntimeError):
        cmd_stack(args)

    report = cmd_compare("twitter16", str(tmp_path))
    assert report["arms"]["stack"]["pooled"]["accuracy"] == 1.0
    assert report["unfinished_arms"] == []  # arms never started under this root are not listed
    assert report["pairs"]["stack_minus_lexical"]["only_b_correct"] == 0
    assert report["pairs"]["lexical_minus_graph"]["n"] == len(df)
    assert "stack - lexical" in format_comparison(report)


def test_text_stack_needs_nothing_but_lexical_and_text_long(tmp_path):
    """A fresh checkpoint root with only the lexical and text_long folds
    (result.json + probs.json, no saved model) stacks and compares."""
    from src.run_cv import build_parser, cmd_compare, cmd_stack, format_comparison

    df = _fake_members(tmp_path, ("lexical", "text_long"))
    args = build_parser().parse_args(
        ["stack", "--dataset", "twitter16", "--members", "lexical", "text_long", "--tag", "text",
         "--ckpt-root", str(tmp_path)]
    )
    out = cmd_stack(args)
    assert out["run"] == "twitter16_4way_stack_text"
    assert [f["test_accuracy"] for f in out["folds"]] == [1.0] * 5

    report = cmd_compare("twitter16", str(tmp_path))
    assert list(report["arms"]) == ["lexical", "text_long", "stack_text"]
    assert report["unfinished_arms"] == []
    assert set(report["pairs"]) == {"text_long_minus_lexical", "stack_text_minus_lexical",
                                    "stack_text_minus_text_long"}
    assert report["pairs"]["stack_text_minus_text_long"]["n"] == len(df)
    text = format_comparison(report)
    assert "stack_text - lexical" in text and "unfinished" not in text


def test_full_pipeline_stack_of_three_members(tmp_path):
    from src.run_cv import build_parser, cmd_compare, cmd_stack, format_comparison

    _fake_members(tmp_path, ("lexical", "text_long", "graph"))
    args = build_parser().parse_args(
        ["stack", "--dataset", "twitter16", "--members", "lexical", "text_long", "graph", "--tag", "full",
         "--ckpt-root", str(tmp_path)]
    )
    out = cmd_stack(args)
    assert out["run"] == "twitter16_4way_stack_full"
    assert all(abs(sum(f["weights"].values()) - 1) < 1e-9 and len(f["weights"]) == 3 for f in out["folds"])
    assert [f["test_accuracy"] for f in out["folds"]] == [1.0] * 5
    report = cmd_compare("twitter16", str(tmp_path))
    assert "stack_full" in report["arms"] and report["unfinished_arms"] == []
    assert {"stack_full_minus_lexical", "stack_full_minus_text_long", "stack_full_minus_graph"} <= set(report["pairs"])
    assert "stack_full - lexical" in format_comparison(report)


def test_text_long_arm_is_the_text_model_on_a_longer_schedule(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import src.finetune as ft
    from src.run_cv import build_parser, cmd_train

    seen = []

    def fake(cfg, ckpt_dir, **kw):
        seen.append((cfg, ckpt_dir, kw))
        n = len(kw["view"].test_indices)
        return {"split": {"source": kw["split_source"]}, "epochs_run": 30, "best_epoch": 12,
                "best_val_macro_f1": 1.0, "test_full": {"n": n, "accuracy": 1.0, "macro_f1": 1.0}}

    monkeypatch.setattr(ft, "run_finetune", fake)
    args = build_parser().parse_args(
        ["train", "--dataset", "twitter16", "--arm", "text_long", "--light", "--folds", "0", "3",
         "--ckpt-root", str(tmp_path)]
    )
    out = cmd_train(args)
    assert out["run"] == "twitter16_4way_text_long" and out["folds"][1]["best_epoch"] == 12
    cfg, ckpt_dir, kw = seen[1]
    default = ft.FinetuneConfig(seed=45)
    assert (cfg.max_epochs, cfg.patience) == (30, 10) and kw["light"] is True
    assert ckpt_dir == tmp_path / "twitter16_4way_text_long" / "fold3"
    skip = ("max_epochs", "patience")
    assert {k: v for k, v in cfg.to_dict().items() if k not in skip} == {
        k: v for k, v in default.to_dict().items() if k not in skip
    }
    # the stage 1 arm itself is unchanged, and --light is refused for non-text arms
    args = build_parser().parse_args(["train", "--dataset", "twitter16", "--arm", "text", "--folds", "0",
                                      "--ckpt-root", str(tmp_path)])
    cmd_train(args)
    assert seen[-1][0].to_dict() == ft.FinetuneConfig(seed=42).to_dict() and seen[-1][2]["light"] is False
    args = build_parser().parse_args(["train", "--dataset", "twitter16", "--arm", "graph", "--light",
                                      "--ckpt-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        cmd_train(args)


def test_neural_member_is_rescored_from_its_artifact(tmp_path):
    pytest.importorskip("torch")
    from src.run_cv import _neural_probs
    from tests.test_propagation import _cfg, _run, toy_graph_data

    result = _run(_cfg("graph", max_epochs=2), tmp_path / "g")
    _, view, texts, graphs = toy_graph_data()
    probs = _neural_probs("graph", tmp_path / "g" / "artifact",
                          {"val": view.val_indices, "test": view.test_indices}, texts, graphs, "cpu")
    assert probs["val"].shape == (len(view.val_indices), 4)
    assert np.allclose(probs["test"].sum(1), 1.0, atol=1e-5)
    assert probs["test"].argmax(1).tolist() == [p["pred"] for p in result["test_predictions"]]
