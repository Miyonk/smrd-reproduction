"""Grey Wolf Optimizer and search-space encoding (no torch needed)."""

import numpy as np
import pytest

from src.cv import Folds
from src.data import ArmView
from src.gwo import DEFAULTS, SPACES, Dim, decode, encode, gwo_maximize


def _sphere(target):
    return lambda index, x: -float(((x - target) ** 2).sum())


def test_budget_bounds_and_determinism():
    seen = []

    def f(index, x):
        assert index == len(seen) and x.shape == (4,)
        assert (x >= 0).all() and (x <= 1).all()
        seen.append(x.copy())
        return _sphere(0.3)(index, x)

    out = gwo_maximize(f, 4, population=6, rounds=5, seed=1)
    assert out["evaluations"] == len(seen) == 36
    assert [h["round"] for h in out["history"]] == [r for r in range(6) for _ in range(6)]
    again = gwo_maximize(_sphere(0.3), 4, population=6, rounds=5, seed=1)
    assert again["history"] == out["history"]
    other = gwo_maximize(_sphere(0.3), 4, population=6, rounds=5, seed=2)
    assert other["history"][0]["position"] != out["history"][0]["position"]


def test_best_is_the_maximum_and_search_improves():
    out = gwo_maximize(_sphere(0.3), 5, population=8, rounds=15, seed=0)
    fits = [h["fitness"] for h in out["history"]]
    assert out["best_fitness"] == max(fits)
    assert out["history"][out["best_index"]]["position"] == out["best_position"]
    assert out["best_fitness"] > max(fits[:8]) and out["best_fitness"] > -0.01


def test_seed_position_is_wolf_zero_and_never_lost():
    start = [0.3, 0.3, 0.3]
    out = gwo_maximize(_sphere(0.3), 3, population=4, rounds=3, seed=0, seed_positions=[start])
    assert out["history"][0]["position"] == start
    assert out["best_index"] == 0 and out["best_fitness"] == 0.0  # the optimum itself
    with pytest.raises(ValueError):
        gwo_maximize(_sphere(0.3), 3, seed_positions=[[0.1, 0.2, 1.5]])
    with pytest.raises(ValueError):
        gwo_maximize(_sphere(0.3), 3, population=2)


def test_replay_from_cached_fitness_matches():
    """A resumed search (fitness read from a cache) proposes the same positions."""
    first = gwo_maximize(_sphere(0.6), 3, population=5, rounds=4, seed=9)
    cache = {h["index"]: h for h in first["history"]}

    def cached(index, x):
        assert np.allclose(cache[index]["position"], x, atol=1e-12)
        return cache[index]["fitness"]

    assert gwo_maximize(cached, 3, population=5, rounds=4, seed=9)["best_position"] == first["best_position"]


def test_dim_decoding():
    assert Dim("c", "choice", choices=(32, 64, 128, 256)).decode(0.0) == 32
    assert Dim("c", "choice", choices=(32, 64, 128, 256)).decode(1.0) == 256
    assert Dim("c", "choice", choices=(32, 64, 128, 256)).decode(0.49) == 64
    log = Dim("lr", "log", 1e-4, 1e-2)
    assert log.decode(0.0) == 1e-4 and log.decode(1.0) == 1e-2 and log.decode(0.5) == 1e-3
    assert Dim("d", "linear", 0.1, 0.7).decode(0.5) == 0.4
    with pytest.raises(ValueError):
        log.encode(1.0)


@pytest.mark.parametrize("arm", sorted(SPACES))
def test_defaults_roundtrip_through_the_space(arm):
    assert set(DEFAULTS[arm]) == {d.name for d in SPACES[arm]}
    position = encode(arm, DEFAULTS[arm])
    assert all(0.0 <= x <= 1.0 for x in position)
    assert decode(arm, position) == DEFAULTS[arm]
    rng = np.random.default_rng(0)
    for _ in range(50):  # decode is stable under encode
        params = decode(arm, rng.random(len(SPACES[arm])))
        assert decode(arm, encode(arm, params)) == params


def test_inner_view_never_touches_the_held_out_fold():
    fold_of_row = [0, 1, 2, 3, 4] * 4
    folds = Folds("twitter16", fold_of_row, list(range(20)), 5, "abc")
    base = ArmView("arm_a", "twitter16", [], [], [], {i: i % 4 for i in range(20)}, 4, list("abcd"))
    for held_out in range(5):
        for val in range(5):
            if val == held_out:
                with pytest.raises(ValueError):
                    folds.inner_view(base, held_out, val)
                continue
            v = folds.inner_view(base, held_out, val)
            used = set(v.train_indices) | set(v.val_indices) | set(v.test_indices)
            assert not used & set(folds.rows(held_out))
            assert v.val_indices == folds.rows(val) == v.test_indices
            assert not set(v.train_indices) & set(v.val_indices) and len(v.train_indices) == 12
    # the first inner view trains and validates on the same rows as the final fit
    final, inner = folds.view(base, 2), folds.inner_view(base, 2, 3)
    assert (final.train_indices, final.val_indices) == (inner.train_indices, inner.val_indices)
