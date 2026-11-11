"""
Grey Wolf Optimizer (Mirjalili et al., 2014) and the stage 3 search spaces.

The optimiser works in the unit cube ``[0, 1]^d`` and **maximises**; a search
space maps a position to training hyperparameters.  NumPy only, no torch.

The run is fully determined by ``seed``: candidate ``i`` always gets the same
position as long as the fitness values it depends on are the same.  The caller
can therefore make a search resumable just by caching ``evaluate`` results by
index (``src/run_tune.py`` does this).

Budget: ``population * (rounds + 1)`` evaluations.  The default 6 wolves and
5 update rounds give 36 evaluations, the same budget the paper's DHOA search
gets in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["gwo_maximize", "Dim", "SPACES", "DEFAULTS", "decode", "encode", "space_signature"]


def gwo_maximize(
    evaluate: Callable[[int, np.ndarray], float],
    dim: int,
    population: int = 6,
    rounds: int = 5,
    seed: int = 0,
    seed_positions: Optional[Sequence[Sequence[float]]] = None,
) -> Dict:
    """Maximise ``evaluate(index, position)`` over ``[0, 1]^dim``.

    ``seed_positions`` replace the first wolves of the initial pack (used to
    start from the default hyperparameters, so the search cannot end below
    them on its own fitness).  Returns the best position/fitness and the full
    evaluation history.
    """
    if population < 3:
        raise ValueError("GWO needs at least 3 wolves (alpha, beta, delta)")
    if rounds < 0 or dim < 1:
        raise ValueError("rounds must be >= 0 and dim >= 1")
    rng = np.random.default_rng(seed)
    pack = rng.random((population, dim))
    for k, pos in enumerate(seed_positions or []):
        if k >= population:
            raise ValueError("more seed positions than wolves")
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (dim,) or (pos < 0).any() or (pos > 1).any():
            raise ValueError("seed positions must lie in [0, 1]^dim")
        pack[k] = pos

    history: List[Dict] = []
    leaders: List[Tuple[float, int, np.ndarray]] = []  # (fitness, eval index, position), best first

    def run_pack(round_no: int) -> None:
        for w in range(population):
            index = len(history)
            fitness = float(evaluate(index, pack[w].copy()))
            if not np.isfinite(fitness):
                raise FloatingPointError(f"non-finite fitness for evaluation {index}")
            history.append({"index": index, "round": round_no, "wolf": w,
                            "position": pack[w].tolist(), "fitness": fitness})
            leaders.append((fitness, index, pack[w].copy()))
        # best three ever seen; earlier evaluation wins a tie
        leaders.sort(key=lambda t: (-t[0], t[1]))
        del leaders[3:]

    run_pack(0)
    for t in range(rounds):
        a = 2.0 * (1.0 - t / rounds)
        moved = np.zeros_like(pack)
        for _, _, leader in leaders:
            r1, r2 = rng.random((population, dim)), rng.random((population, dim))
            A, C = 2.0 * a * r1 - a, 2.0 * r2
            moved += leader - A * np.abs(C * leader - pack)
        pack = np.clip(moved / len(leaders), 0.0, 1.0)
        run_pack(t + 1)

    best_fitness, best_index, best_position = leaders[0]
    return {
        "best_position": best_position.tolist(),
        "best_fitness": best_fitness,
        "best_index": best_index,
        "evaluations": len(history),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dim:
    name: str
    kind: str  # log | linear | choice
    low: float = 0.0
    high: float = 1.0
    choices: Tuple = ()

    def decode(self, x: float):
        x = min(max(float(x), 0.0), 1.0)
        if self.kind == "choice":
            return self.choices[min(int(x * len(self.choices)), len(self.choices) - 1)]
        if self.kind == "log":
            value = self.low * (self.high / self.low) ** x
        elif self.kind == "linear":
            value = self.low + (self.high - self.low) * x
        else:
            raise ValueError(f"unknown dimension kind {self.kind!r}")
        return float(f"{value:.4g}")

    def encode(self, value) -> float:
        if self.kind == "choice":
            return (self.choices.index(value) + 0.5) / len(self.choices)
        if self.kind == "log":
            x = np.log(value / self.low) / np.log(self.high / self.low)
        else:
            x = (value - self.low) / (self.high - self.low)
        if not -1e-9 <= x <= 1 + 1e-9:
            raise ValueError(f"{self.name}={value} is outside [{self.low}, {self.high}]")
        return float(min(max(x, 0.0), 1.0))


_GRAPH_DIMS = (
    Dim("lr_graph", "log", 1e-4, 5e-3),
    Dim("weight_decay_graph", "log", 1e-5, 1e-2),
    Dim("graph_hidden", "choice", choices=(32, 64, 128, 256)),
    Dim("graph_out", "choice", choices=(32, 64, 128, 256)),
    Dim("graph_dropout", "linear", 0.1, 0.7),
    Dim("edge_drop", "linear", 0.0, 0.5),
    Dim("label_smoothing", "linear", 0.0, 0.2),
)

# Names are ``PropagationConfig`` fields.  Everything not listed keeps the
# stage 2 value (schedule, epochs, patience, class weights, ...).
SPACES: Dict[str, Tuple[Dim, ...]] = {
    "graph": _GRAPH_DIMS + (Dim("batch_size", "choice", choices=(16, 32, 64, 128)),),
    "fusion": _GRAPH_DIMS + (
        Dim("lr_text", "log", 1e-5, 5e-5),
        Dim("dropout", "linear", 0.0, 0.3),
    ),
}

# Stage 2 values of the searched fields (checked against ``PropagationConfig``
# and ``ARM_DEFAULTS`` in tests/test_tune.py).
DEFAULTS: Dict[str, Dict] = {
    "graph": dict(lr_graph=5e-4, weight_decay_graph=1e-4, graph_hidden=64, graph_out=64,
                  graph_dropout=0.5, edge_drop=0.2, label_smoothing=0.1, batch_size=64),
    "fusion": dict(lr_graph=5e-4, weight_decay_graph=1e-4, graph_hidden=64, graph_out=64,
                   graph_dropout=0.5, edge_drop=0.2, label_smoothing=0.1, lr_text=2e-5, dropout=0.1),
}


def decode(arm: str, position: Sequence[float]) -> Dict:
    dims = SPACES[arm]
    if len(position) != len(dims):
        raise ValueError(f"{arm} space has {len(dims)} dimensions, got {len(position)}")
    return {d.name: d.decode(x) for d, x in zip(dims, position)}


def encode(arm: str, params: Dict) -> List[float]:
    return [d.encode(params[d.name]) for d in SPACES[arm]]


def space_signature(arm: str) -> List[List]:
    """JSON-friendly description, stored with a search so that a changed
    space cannot silently reuse cached evaluations."""
    return [[d.name, d.kind, d.low, d.high, list(d.choices)] for d in SPACES[arm]]
