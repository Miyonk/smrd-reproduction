"""Cross-validation protocol: folds, per-fold vocabulary, paired comparison."""

import json

import numpy as np
import pytest

from src.config import DATASETS, Config
from src.cv import Folds, make_folds, mcnemar_exact, paired_difference, pool_fold_results
from src.data import ArmView, build_arm_view, load_dataset
from src.propagation import build_tree, restrict_vocab


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
def test_make_folds_keeps_groups_together():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400).tolist()
    groups = rng.integers(0, 150, 400).tolist()
    fold = make_folds(labels, groups)
    assert sorted(set(fold)) == [0, 1, 2, 3, 4]
    seen = {}
    for g, f in zip(groups, fold):
        assert seen.setdefault(g, f) == f
    assert make_folds(labels, groups) == fold  # deterministic


def _toy_folds():
    fold_of_row = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    return Folds("twitter16", fold_of_row, list(range(10)), 5, "abc")


def _toy_view(labels):
    return ArmView("arm_a", "twitter16", [], [], [], labels, 4, ["a", "b", "c", "d"])


def test_fold_view_partitions_rows():
    folds = _toy_folds()
    base = _toy_view({i: i % 4 for i in range(10)})
    tested = []
    for k in range(5):
        v = folds.view(base, k)
        assert v.test_indices == [k, k + 5]
        assert v.val_indices == [(k + 1) % 5, (k + 1) % 5 + 5]
        parts = [set(v.train_indices), set(v.val_indices), set(v.test_indices)]
        assert sum(len(p) for p in parts) == 10 and len(set().union(*parts)) == 10
        tested += v.test_indices
    assert sorted(tested) == list(range(10))
    with pytest.raises(ValueError):
        folds.view(base, 5)


def test_fold_view_respects_arm_exclusions():
    folds = _toy_folds()
    base = _toy_view({i: 0 for i in range(10) if i != 5})  # row 5 not in this arm
    assert folds.view(base, 0).test_indices == [0]


def test_group_leak_detected():
    folds = Folds("twitter16", [0, 1], [7, 7], 5, "x")
    with pytest.raises(ValueError):
        folds.check_no_group_leak()


@pytest.mark.parametrize("dataset", DATASETS)
def test_committed_fold_files_are_valid(dataset):
    df = load_dataset(dataset)
    folds = Folds.load(dataset, df["tweet_id"].astype(str).tolist())
    folds.check_no_group_leak()
    assert len(folds.fold_of_row) == len(df) and set(folds.fold_of_row) == set(range(5))
    with pytest.raises(ValueError):
        Folds.load(dataset, df["tweet_id"].astype(str).tolist()[::-1])


# ---------------------------------------------------------------------------
# Per-fold vocabulary
# ---------------------------------------------------------------------------
def test_restrict_vocab_uses_training_rows_only():
    def tree(bags):
        return build_tree(0, {k: 0 for k in range(1, len(bags))},
                          {k: [(w, 1.0) for w in b] for k, b in enumerate(bags)}, 50)[0]

    graphs = [
        tree([[1, 2], [2, 3]]),      # train: df 1->1, 2->2, 3->1
        tree([[2, 7], [7], [7, 3]]),  # train: df 2->3, 7->3, 3->2
        tree([[9, 9 + 1, 2], [9]]),  # test only: words 9, 10 must be dropped
    ]
    out, info = restrict_vocab(graphs, train_rows=[0, 1], size=2, min_df=2)
    # df over train nodes: 2 -> 3, 7 -> 3, 3 -> 2, 1 -> 1  => top-2 = {2, 7} (tie by id)
    assert info["vocab_size"] == 2
    assert out[0].words.tolist() == [0, 0] and out[0].offsets.tolist() == [0, 1]
    assert out[1].words.tolist() == [0, 1, 1, 1] and out[1].offsets.tolist() == [0, 2, 3]
    assert out[2].words.tolist() == [0] and out[2].offsets.tolist() == [0, 1]
    assert [g.n_nodes for g in out] == [2, 3, 2]
    # min_df drops rare words even when there is room
    out, info = restrict_vocab(graphs, train_rows=[0], size=10, min_df=2)
    assert info["vocab_size"] == 1


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------
def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)
    assert mcnemar_exact(10, 10) == 1.0


def test_paired_difference():
    a = [True] * 8 + [False] * 2
    b = [True] * 5 + [False] * 3 + [True] * 2
    d = paired_difference(a, b)
    assert d["only_a_correct"] == 3 and d["only_b_correct"] == 2
    assert d["accuracy_difference"] == pytest.approx(0.1)
    assert d["difference_ci95"][0] < 0.1 < d["difference_ci95"][1]
    with pytest.raises(ValueError):
        paired_difference(a, b[:-1])


def test_pool_fold_results_requires_each_row_once():
    r0 = {"test_predictions": [{"row": 0, "label": 1, "pred": 1}, {"row": 2, "label": 0, "pred": 1}]}
    r1 = {"test_predictions": [{"row": 1, "label": 2, "pred": 2}]}
    assert pool_fold_results([r0, r1], [0, 1, 2]) == ([0, 1, 2], [1, 2, 0], [1, 2, 1])
    with pytest.raises(ValueError):
        pool_fold_results([r0], [0, 1, 2])
    with pytest.raises(ValueError):
        pool_fold_results([r0, r0], [0, 2])


def test_compare_end_to_end(tmp_path):
    """Fake fold results on the real Twitter16 folds: text is perfect, graph
    gets every 4th row wrong, fusion is unfinished."""
    from src.run_cv import cmd_compare, format_comparison

    df = load_dataset("twitter16")
    base = build_arm_view(Config(task_arm="arm_a", dataset_name="twitter16"), df=df)
    folds = Folds.load("twitter16", df["tweet_id"].astype(str).tolist())
    for arm in ("text", "graph"):
        for k in range(5):
            view = folds.view(base, k)
            preds = []
            for i in view.test_indices:
                y = base.labels[i]
                wrong = arm == "graph" and i % 4 == 0
                preds.append({"row": i, "label": y, "pred": (y + 1) % 4 if wrong else y})
            acc = sum(p["label"] == p["pred"] for p in preds) / len(preds)
            d = tmp_path / f"twitter16_4way_{arm}" / f"fold{k}"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps({
                "split": {"source": folds.split_source(k)},
                "test_full": {"accuracy": acc, "macro_f1": acc, "n": len(preds)},
                "best_val_macro_f1": 0.5, "best_epoch": 1, "test_predictions": preds,
            }))
    out = cmd_compare("twitter16", str(tmp_path))
    assert out["unfinished_arms"] == ["fusion"] and out["n"] == len(df)
    assert out["arms"]["text"]["pooled"]["accuracy"] == 1.0
    pair = out["pairs"]["graph_minus_text"]
    assert pair["only_a_correct"] == 0 and pair["only_b_correct"] == sum(i % 4 == 0 for i in range(len(df)))
    assert pair["mcnemar_p"] < 1e-6
    assert "graph - text" in format_comparison(out)
    # a result from another fold definition is rejected
    bad = tmp_path / "twitter16_4way_text" / "fold0" / "result.json"
    r = json.loads(bad.read_text())
    r["split"]["source"] = "something else"
    bad.write_text(json.dumps(r))
    with pytest.raises(RuntimeError):
        cmd_compare("twitter16", str(tmp_path))
