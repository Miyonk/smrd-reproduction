"""
Deer Hunting Optimization Algorithm (DHOA).

Specification: Section 5 of the technical design.  Equations 4-14 are
implemented in the normalized ``[0, 1]^5`` coordinate space and mapped to the
paper's parameter ranges with the documented T3 protocols.

Primary Table 2 branch mapping (T3, Assumption A-DHOA-BRANCH):
    |l| < 1 and |P| >= 1 -> Strategy 1 (Eq. 7)
    |l| < 1 and |P| <  1 -> Strategy 3 (Eq. 14)
    |l| >= 1             -> Strategy 2 (Eq. 13)

``|P|`` is a vector norm reduced to a scalar; the reduction is configurable
and defaults to the mean absolute value.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import Config, replace_file

LOGGER = logging.getLogger(__name__)

PARAM_NAMES = ("L", "B", "NL", "NAH", "DR")
L_MIN = 1e-5
L_MAX = 5e-4
B_MIN, B_MAX = 1, 512
NL_MIN, NL_MAX = 1, 48
NAH_MIN, NAH_MAX = 1, 16
DR_MIN, DR_MAX = 0.1, 0.5


class PopulationInfeasibleError(RuntimeError):
    """Raised when no candidate in the initial population is feasible."""


@dataclass(frozen=True)
class Hyperparameters:
    L: float
    B: int
    NL: int
    NAH: int
    DR: float

    def as_tuple(self) -> Tuple[float, int, int, int, float]:
        return (
            round(float(self.L), 12),
            int(self.B),
            int(self.NL),
            int(self.NAH),
            round(float(self.DR), 6),
        )

    def as_dict(self) -> Dict[str, float | int]:
        return asdict(self)


@dataclass
class CandidateResult:
    hyperparameters: Hyperparameters
    coords: List[float]
    fitness: Optional[float]
    val_loss: Optional[float]
    status: str
    record: Dict = field(default_factory=dict)

    def rank_key(self) -> Tuple:
        feasible = self.fitness is not None
        return (
            -float(self.fitness) if feasible else math.inf,
            float(self.val_loss) if (feasible and self.val_loss is not None) else math.inf,
            int(self.hyperparameters.NL),
            float(self.hyperparameters.L),
        )


# ---------------------------------------------------------------------------
# Coordinate mapping (T3 - Assumption A-DHOA-SPACE)
# ---------------------------------------------------------------------------
def decode_coordinates(coords: Sequence[float], config: Config) -> Hyperparameters:
    u = np.clip(np.asarray(coords, dtype=np.float64), 0.0, 1.0)
    if u.shape != (5,):
        raise ValueError(f"expected 5 coordinates, got {u.shape}")

    if config.dhoa_lr_mapping == "log":
        log_min, log_max = math.log10(L_MIN), math.log10(L_MAX)
        learning_rate = 10.0 ** (log_min + u[0] * (log_max - log_min))
    else:
        learning_rate = L_MIN + u[0] * (L_MAX - L_MIN)

    batch = int(round(B_MIN + u[1] * (B_MAX - B_MIN)))
    batch = max(B_MIN, min(B_MAX, batch))

    nl_high = config.max_layers if config.max_layers is not None else NL_MAX
    layers = int(round(NL_MIN + u[2] * (nl_high - NL_MIN)))
    layers = max(NL_MIN, min(nl_high, layers))

    heads = int(round(NAH_MIN + u[3] * (NAH_MAX - NAH_MIN)))
    heads = max(NAH_MIN, min(NAH_MAX, heads))

    dropout = DR_MIN + u[4] * (DR_MAX - DR_MIN)
    dropout = max(DR_MIN, min(DR_MAX, dropout))

    return Hyperparameters(
        L=float(learning_rate),
        B=int(batch),
        NL=int(layers),
        NAH=int(heads),
        DR=float(dropout),
    )


def encode_hyperparameters(hp: Hyperparameters, config: Config) -> np.ndarray:
    if config.dhoa_lr_mapping == "log":
        log_min, log_max = math.log10(L_MIN), math.log10(L_MAX)
        u0 = (math.log10(max(hp.L, L_MIN)) - log_min) / (log_max - log_min)
    else:
        u0 = (hp.L - L_MIN) / (L_MAX - L_MIN)
    nl_high = config.max_layers if config.max_layers is not None else NL_MAX
    u = np.array(
        [
            u0,
            (hp.B - B_MIN) / (B_MAX - B_MIN),
            (hp.NL - NL_MIN) / (nl_high - NL_MIN),
            (hp.NAH - NAH_MIN) / (NAH_MAX - NAH_MIN),
            (hp.DR - DR_MIN) / (DR_MAX - DR_MIN),
        ],
        dtype=np.float64,
    )
    return np.clip(u, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class DHOAOptimizer:
    """6 deer x (1 + 5) rounds = 36 evaluations."""

    def __init__(
        self,
        config: Config,
        evaluate_fn: Callable[..., Dict],
        state_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.state_path = Path(state_path) if state_path else config.search_state
        self.rng = np.random.default_rng(config.seed_dhoa)

        self.population: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None
        self.theta0: float = 0.0
        self.history: List[CandidateResult] = []
        self.cache: Dict[Tuple, CandidateResult] = {}
        self.leader: Optional[CandidateResult] = None
        self.successor: Optional[CandidateResult] = None
        self.iteration = 0
        self._load_state()

    # -- state -------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("could not read DHOA state %s; starting fresh", self.state_path)
            return
        for item in raw.get("cache", []):
            result = _result_from_json(item)
            self.cache[result.hyperparameters.as_tuple()] = result
        LOGGER.info("loaded %d cached evaluations from %s", len(self.cache), self.state_path)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": "DHOA",
            "config": {
                "seed_dhoa": self.config.seed_dhoa,
                "num_deer": self.config.num_deer,
                "num_iterations": self.config.num_iterations,
                "max_layers": self.config.max_layers,
                "branch_profile": self.config.dhoa_branch_profile,
            },
            "iteration": self.iteration,
            "evaluations": len(self.history),
            "leader": _result_to_json(self.leader) if self.leader else None,
            "successor": _result_to_json(self.successor) if self.successor else None,
            "cache": [_result_to_json(r) for r in self.cache.values()],
            "history": [_result_to_json(r) for r in self.history],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        replace_file(tmp, self.state_path)

    # -- evaluation --------------------------------------------------------
    def _evaluate(self, coords: np.ndarray) -> CandidateResult:
        hp = decode_coordinates(coords, self.config)
        key = hp.as_tuple()
        if key in self.cache:
            cached = self.cache[key]
            result = CandidateResult(
                hyperparameters=cached.hyperparameters,
                coords=list(map(float, coords)),
                fitness=cached.fitness,
                val_loss=cached.val_loss,
                status="CACHE_HIT",
                record=dict(cached.record, status="CACHE_HIT"),
            )
            self.history.append(result)
            return result

        payload = self.evaluate_fn(hp, evaluation_index=len(self.history))
        result = CandidateResult(
            hyperparameters=hp,
            coords=list(map(float, coords)),
            fitness=payload.get("fitness"),
            val_loss=payload.get("val_loss"),
            status=payload.get("status", "EVALUATED"),
            record=payload.get("record", {}),
        )
        self.cache[key] = result
        self.history.append(result)
        self._save_state()
        return result

    def _ranked_feasible(self) -> List[CandidateResult]:
        seen: Dict[Tuple, CandidateResult] = {}
        for result in self.history:
            if result.fitness is None:
                continue
            key = result.hyperparameters.as_tuple()
            if key not in seen or result.rank_key() < seen[key].rank_key():
                seen[key] = result
        return sorted(seen.values(), key=lambda r: r.rank_key())

    def _update_leader_successor(self, initial: bool = False) -> None:
        ranked = self._ranked_feasible()
        if not ranked:
            if initial:
                raise PopulationInfeasibleError(
                    "all 6 initial candidates are infeasible; emitting "
                    "INFEASIBLE_HARDWARE_ABORT"
                )
            return
        self.leader = ranked[0]
        self.successor = ranked[1] if len(ranked) >= 2 else ranked[0]

    # -- equation helpers --------------------------------------------------
    def _log(self, value: float) -> float:
        if self.config.dhoa_log_base == "10":
            return math.log10(value)
        if self.config.dhoa_log_base == "2":
            return math.log2(value)
        return math.log(value)

    def _p_scalar(self, p: np.ndarray) -> float:
        if self.config.dhoa_p_norm == "l2":
            return float(np.linalg.norm(p))
        if self.config.dhoa_p_norm == "max":
            return float(np.max(np.abs(p)))
        return float(np.mean(np.abs(p)))

    def _strategy(self, deer: np.ndarray, leader: np.ndarray, successor: np.ndarray,
                  t: int, phi: float, theta: float) -> np.ndarray:
        # Eq. 7 / 8 / 9 variables
        z = self.rng.uniform(-1.0, 1.0, size=5)
        w = self.rng.uniform(0.0, 1.0, size=5)
        l = float(self.rng.uniform(0.0, 2.0))
        d = 0.25 * self._log(t + 1.0 / self.config.num_iterations) * z
        p = 2.0 * w
        p_scalar = self._p_scalar(p)

        # Eq. 10 / 11 / 12
        r = float(self.rng.uniform(0.0, 1.0))
        y = (math.pi / 8.0) * r
        x = theta - y
        phi_next = phi + x
        v = phi_next if self.config.dhoa_v_mode == "phi" else float(
            self.rng.uniform(-math.pi / 4.0, math.pi / 4.0)
        )

        if self.config.dhoa_branch_profile == "primary":
            if abs(l) < 1.0:
                if p_scalar >= 1.0:
                    updated = leader - d * l * np.abs(p * leader - deer)  # Eq. 7
                else:
                    updated = successor - d * l * np.abs(p * successor - deer)  # Eq. 14
            else:
                updated = leader - l * np.abs(math.cos(v) * leader - deer)  # Eq. 13
        else:
            # Ablation S10 alternative mapping.
            if abs(l) < 1.0:
                if p_scalar >= 1.0:
                    updated = leader - l * np.abs(math.cos(v) * leader - deer)
                else:
                    updated = successor - d * l * np.abs(p * successor - deer)
            else:
                updated = leader - d * l * np.abs(p * leader - deer)

        return np.clip(updated, 0.0, 1.0), phi_next

    # -- main loop ---------------------------------------------------------
    def run(self) -> Dict:
        n = self.config.num_deer
        self.population = self.rng.random((n, 5))
        self.phi = self.rng.uniform(0.0, 2.0 * math.pi, size=n)
        self.theta0 = float(self.rng.uniform(0.0, 2.0 * math.pi))

        # Initial population (round t=0).
        for k in range(n):
            self._evaluate(self.population[k])
        self._update_leader_successor(initial=True)
        self.iteration = 0
        self._save_state()

        for t in range(1, self.config.num_iterations + 1):
            assert self.leader is not None and self.successor is not None
            leader = np.array(self.leader.coords)
            successor = np.array(self.successor.coords)
            theta = self.theta0
            if self.config.dhoa_theta_mode == "random_walk":
                theta = (self.theta0 + self.rng.normal(0.0, 0.05)) % (2.0 * math.pi)

            updates = []
            for k in range(n):
                updated, phi_next = self._strategy(
                    self.population[k], leader, successor, t, float(self.phi[k]), theta
                )
                self.phi[k] = phi_next
                updates.append(updated)
            self.population = np.vstack(updates)

            for k in range(n):
                self._evaluate(self.population[k])

            self._update_leader_successor()
            self.iteration = t
            self._save_state()
            LOGGER.info(
                "DHOA iteration %d/%d complete; leader accuracy=%s",
                t,
                self.config.num_iterations,
                self.leader.fitness if self.leader else None,
            )

        return self.result()

    def result(self) -> Dict:
        return {
            "leader": _result_to_json(self.leader) if self.leader else None,
            "successor": _result_to_json(self.successor) if self.successor else None,
            "evaluations": len(self.history),
            "distinct_evaluations": len(self.cache),
            "iterations": self.iteration,
            "history": [_result_to_json(r) for r in self.history],
            "state_path": str(self.state_path),
        }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _result_to_json(result: CandidateResult) -> Dict:
    return {
        "hyperparameters": result.hyperparameters.as_dict(),
        "coords": result.coords,
        "fitness": result.fitness,
        "val_loss": result.val_loss,
        "status": result.status,
        "record": result.record,
    }


def _result_from_json(raw: Dict) -> CandidateResult:
    hp = Hyperparameters(**raw["hyperparameters"])
    return CandidateResult(
        hyperparameters=hp,
        coords=list(raw.get("coords", [])),
        fitness=raw.get("fitness"),
        val_loss=raw.get("val_loss"),
        status=raw.get("status", "EVALUATED"),
        record=raw.get("record", {}),
    )
