import json

import numpy as np
import pytest

from src.config import Config
from src.dhoa import (
    DHOAOptimizer,
    Hyperparameters,
    PopulationInfeasibleError,
    decode_coordinates,
    encode_hyperparameters,
)


def _fake_eval(hp: Hyperparameters, evaluation_index: int):
    # Smooth, deterministic objective with a unique-ish optimum.
    fitness = (
        1.0
        - abs(hp.DR - 0.25)
        - abs(hp.NAH - 8) / 100.0
        - abs(hp.NL - 24) / 1000.0
    )
    return {"fitness": fitness, "val_loss": 1.0 - fitness, "status": "EVALUATED"}


def test_decode_ranges():
    cfg = Config(max_layers=None)
    low = decode_coordinates([0, 0, 0, 0, 0], cfg)
    high = decode_coordinates([1, 1, 1, 1, 1], cfg)
    assert low.L == pytest.approx(1e-5)
    assert high.L == pytest.approx(5e-4)
    assert (low.B, low.NL, low.NAH) == (1, 1, 1)
    assert (high.B, high.NL, high.NAH) == (512, 48, 16)
    assert low.DR == pytest.approx(0.1)
    assert high.DR == pytest.approx(0.5)


def test_encode_roundtrip():
    cfg = Config(max_layers=None)
    hp = Hyperparameters(L=1e-4, B=32, NL=28, NAH=12, DR=0.25)
    coords = encode_hyperparameters(hp, cfg)
    decoded = decode_coordinates(coords, cfg)
    assert decoded == hp


def test_dhoa_runs_exactly_36_evaluations(tmp_path):
    cfg = Config(max_layers=None)
    optimizer = DHOAOptimizer(
        cfg, _fake_eval, state_path=tmp_path / "state.json"
    )
    result = optimizer.run()
    assert result["evaluations"] == 36
    assert result["leader"] is not None
    assert result["successor"] is not None
    state = json.loads((tmp_path / "state.json").read_text())
    assert len(state["cache"]) >= 1


def test_dhoa_is_deterministic(tmp_path):
    cfg = Config(max_layers=None)
    first = DHOAOptimizer(cfg, _fake_eval, state_path=tmp_path / "a.json").run()
    second = DHOAOptimizer(cfg, _fake_eval, state_path=tmp_path / "b.json").run()
    assert first["leader"]["hyperparameters"] == second["leader"]["hyperparameters"]
    assert first["leader"]["coords"] == pytest.approx(second["leader"]["coords"])


def test_dhoa_resume_uses_cache(tmp_path):
    cfg = Config(max_layers=None)
    state = tmp_path / "state.json"
    DHOAOptimizer(cfg, _fake_eval, state_path=state).run()
    calls = {"n": 0}

    def counting_eval(hp, evaluation_index):
        calls["n"] += 1
        return _fake_eval(hp, evaluation_index)

    resumed = DHOAOptimizer(cfg, counting_eval, state_path=state).run()
    assert resumed["evaluations"] == 36
    assert calls["n"] == 0  # every position was already cached


def test_all_infeasible_aborts(tmp_path):
    cfg = Config(max_layers=None)

    def infeasible(hp, evaluation_index):
        return {"fitness": None, "val_loss": None, "status": "INFEASIBLE_HARDWARE"}

    with pytest.raises(PopulationInfeasibleError):
        DHOAOptimizer(cfg, infeasible, state_path=tmp_path / "x.json").run()


def test_nl_cap_is_honoured(tmp_path):
    cfg = Config(max_layers=24)
    hp = decode_coordinates([0.5, 0.5, 1.0, 0.5, 0.5], cfg)
    assert hp.NL <= 24
    result = DHOAOptimizer(cfg, _fake_eval, state_path=tmp_path / "c.json").run()
    for item in result["history"]:
        assert item["hyperparameters"]["NL"] <= 24
