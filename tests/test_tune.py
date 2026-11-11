"""Stage 3: nested GWO search on toy graph data (CPU, graph arm, seconds)."""

import dataclasses
import json

import pytest

from src.gwo import DEFAULTS, SPACES
from src.run_tune import SEARCH_FILE, tune_fold
from src.train_propagation import ARM_DEFAULTS, PropagationConfig
from tests.test_propagation import N_FEAT, toy_graph_data


def test_space_defaults_are_the_stage2_settings():
    for arm in SPACES:
        cfg = PropagationConfig(arm=arm, **ARM_DEFAULTS[arm])
        assert {k: getattr(cfg, k) for k in DEFAULTS[arm]} == DEFAULTS[arm]


def _setup():
    df, view, texts, graphs = toy_graph_data()
    half = len(view.train_indices) // 2
    a, b = view.train_indices[:half], view.train_indices[half:]
    # inner views use training + validation rows only; the test rows stay out
    inner = [
        ("valA", dataclasses.replace(view, train_indices=b + view.val_indices, val_indices=a, test_indices=a), graphs),
        ("valB", dataclasses.replace(view, train_indices=a + view.val_indices, val_indices=b, test_indices=b), graphs),
    ]
    fixed = dict(arm="graph", n_features=N_FEAT, schedule="constant", warmup_ratio=0.0,
                 max_epochs=2, patience=2, amp="float32", eval_batch_size=16, seed=7,
                 graph_cache_sha256="toy")
    return df, view, texts, graphs, inner, fixed


def _tune(tmp_path, **kw):
    df, view, texts, graphs, inner, fixed = _setup()
    args = dict(fold_dir=tmp_path / "fold0", scratch_dir=tmp_path / "scratch", arm="graph",
                fixed=fixed, final=(view, graphs, {"sha256": "toy"}), inner=inner, df=df,
                texts=texts, population=3, rounds=1, gwo_seed=3, device="cpu",
                split_source="toy split")
    args.update(kw)
    return tune_fold(**args), view


def test_search_then_final_fit(tmp_path, monkeypatch):
    import src.train_propagation as tp

    _, view0, _, _, _, _ = _setup()
    touched, original = [], tp.evaluate_split

    def spy(model, tok, texts, graphs, indices, *a):
        touched.append(list(indices) == list(view0.test_indices))
        return original(model, tok, texts, graphs, indices, *a)

    monkeypatch.setattr(tp, "evaluate_split", spy)
    out, view = _tune(tmp_path)
    assert sum(touched) == 1 and touched[-1]  # test rows scored once, by the final fit only

    search, result = out["search"], out["result"]
    evals = search["evaluations"]
    assert sorted(evals, key=int) == [str(i) for i in range(6)]
    assert evals["0"]["params"] == DEFAULTS["graph"]
    for e in evals.values():
        assert set(e["scores"]) == {"valA", "valB"}
        assert e["fitness"] == pytest.approx(sum(e["scores"].values()) / 2)
    best = search["best"]
    assert best["fitness"] == max(e["fitness"] for e in evals.values())
    assert best["fitness"] >= best["default_fitness"]
    for name, value in best["params"].items():
        assert result["config"][name] == value
    assert result["split"]["source"] == "toy split"
    assert result["test_full"]["n"] == len(view.test_indices)
    assert (tmp_path / "fold0" / "artifact" / "model.safetensors").exists()
    assert not any((tmp_path / "scratch").glob("*"))  # search runs leave nothing behind


def test_search_resumes_from_the_log(tmp_path, monkeypatch):
    import src.train_propagation as tp

    out, _ = _tune(tmp_path)
    path = tmp_path / "fold0" / SEARCH_FILE
    state = json.loads(path.read_text())
    # forget the final fit, the last candidate, and one run of the one before
    (tmp_path / "fold0" / "result.json").unlink()
    del state["evaluations"]["5"]
    del state["evaluations"]["4"]["scores"]["valB"], state["evaluations"]["4"]["fitness"]
    state["best"] = None
    path.write_text(json.dumps(state))

    calls, original = [], tp.run_propagation

    def spy(cfg, ckpt_dir, **kw):
        calls.append(str(ckpt_dir))
        return original(cfg, ckpt_dir, **kw)

    monkeypatch.setattr(tp, "run_propagation", spy)
    again, _ = _tune(tmp_path)
    # 1 missing run of eval 4, both runs of eval 5 (unless a twin of an earlier
    # candidate), and the final fit
    twin = again["search"]["evaluations"]["5"].get("duplicate", False)
    assert len(calls) == (2 if twin else 4)
    assert again["search"]["best"] == out["search"]["best"]
    assert again["search"]["evaluations"]["4"]["scores"] == pytest.approx(
        out["search"]["evaluations"]["4"]["scores"])
    assert again["result"]["test_full"]["accuracy"] == pytest.approx(out["result"]["test_full"]["accuracy"])

    with pytest.raises(RuntimeError):  # different budget, same directory
        _tune(tmp_path, rounds=2)
