"""
Stage 3: Grey Wolf hyperparameter search, nested inside the 5-fold protocol.

    python -m src.run_tune search --dataset twitter16 --arm graph --ckpt-root <dir>
    python -m src.run_cv compare --dataset twitter16 --ckpt-root <dir>

For every outer fold ``k`` (``src/cv.py``):

1. **Search.**  GWO (``src/gwo.py``) proposes hyperparameters.  A candidate's
   fitness is its mean validation macro-F1 over ``--inner-vals`` training runs
   that use only the other four folds (validation fold ``k+1``, ``k+2``, ...).
   Fold ``k`` is never read during the search.  Wolf 0 starts at the stage 2
   defaults.
2. **Final fit.**  The best candidate is trained once on the ordinary fold-``k``
   view (same train/val/test rows and seed as the untuned arm) and tested once.

The final fits land in ``<ckpt-root>/<dataset>_<task>_<arm>_gwo/fold<k>/`` in
the same format as every other arm, so ``run_cv compare`` reports
``<arm>_gwo`` next to ``<arm>`` with a paired test.

Resuming: every inner training run is recorded in ``fold<k>/search.json`` as
soon as it finishes; re-running the command replays the search from that file
and continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASETS, replace_file
from src.cv import Folds
from src.data import load_dataset
from src.gwo import DEFAULTS, SPACES, decode, encode, gwo_maximize, space_signature
from src.run_cv import DEFAULT_CKPT_ROOT, DEFAULT_TASK, FOLD_VOCAB_DATASETS, _base_view, _run_dir

LOGGER = logging.getLogger("smrd.tune")
TUNE_ARMS = tuple(SPACES)
SEARCH_FILE = "search.json"


def _save_json(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    replace_file(tmp, path)


def tune_fold(
    *,
    fold_dir: Path | str,
    scratch_dir: Path | str,
    arm: str,
    fixed: Dict,
    final: Tuple,
    inner: Sequence[Tuple],
    df,
    texts: List[str],
    population: int = 6,
    rounds: int = 5,
    gwo_seed: int = 0,
    device: Optional[str] = None,
    split_source: Optional[str] = None,
    identity: Optional[Dict] = None,
    tokenizer=None,
) -> Dict:
    """Search on ``inner`` = ``[(name, view, graphs), ...]``, then fit the best
    candidate on ``final`` = ``(view, graphs, graph_meta)``.

    ``fixed`` holds the ``PropagationConfig`` fields that are not searched.
    Returns ``{"search": <search.json content>, "result": <final result>}``.
    """
    from src.train_propagation import PropagationConfig, run_propagation

    fold_dir, scratch_dir = Path(fold_dir), Path(scratch_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    search_path = fold_dir / SEARCH_FILE
    settings = {
        "arm": arm,
        "population": population,
        "rounds": rounds,
        "gwo_seed": gwo_seed,
        "space": space_signature(arm),
        "inner": [name for name, _, _ in inner],
        "fixed": fixed,
        **(identity or {}),
    }
    settings = json.loads(json.dumps(settings))  # tuples -> lists, as it will be read back
    if search_path.exists():
        state = json.loads(search_path.read_text(encoding="utf-8"))
        if state["settings"] != settings:
            raise RuntimeError(
                f"{search_path} was started with different search settings; "
                "use another --tag or --fresh"
            )
    else:
        state = {"settings": settings, "evaluations": {}, "best": None}
    evals: Dict[str, Dict] = state["evaluations"]
    total = population * (rounds + 1)
    started = time.perf_counter()
    fits_done = [0]

    def evaluate(index: int, position: np.ndarray) -> float:
        params = decode(arm, position)
        entry = evals.get(str(index))
        if entry is None:
            entry = evals[str(index)] = {"position": position.tolist(), "params": params, "scores": {}}
        elif not np.allclose(entry["position"], position, atol=1e-12):
            raise RuntimeError(
                f"{search_path}: cached evaluation {index} does not match the replayed search; "
                "rerun with --fresh"
            )
        # two wolves can decode to the same hyperparameters; train them once
        twin = next(
            (e for i, e in evals.items()
             if int(i) < index and e["params"] == params and "fitness" in e), None,
        )
        if twin is not None and "fitness" not in entry:
            entry.update(scores=dict(twin["scores"]), fitness=twin["fitness"], duplicate=True)
            _save_json(search_path, state)
        for name, view, graphs in inner:
            if name in entry["scores"]:
                continue
            run_dir = scratch_dir / f"eval{index}_{name}"
            shutil.rmtree(run_dir, ignore_errors=True)
            cfg = PropagationConfig(**{**fixed, **params})
            t0 = time.perf_counter()
            train_log = logging.getLogger("src.train_propagation")
            level = train_log.level
            train_log.setLevel(logging.WARNING)  # no per-epoch lines for search runs
            try:
                result = run_propagation(
                    cfg, run_dir, df=df, view=view, texts=texts, graphs=graphs,
                    device=device, tokenizer=tokenizer, save_model=False,
                )
            finally:
                train_log.setLevel(level)
            shutil.rmtree(run_dir, ignore_errors=True)
            entry["scores"][name] = result["best_val_macro_f1"]
            entry.setdefault("seconds", {})[name] = round(time.perf_counter() - t0, 1)
            fits_done[0] += 1
            _save_json(search_path, state)
        if "fitness" not in entry:
            entry["fitness"] = statistics.fmean(entry["scores"][name] for name, _, _ in inner)
            _save_json(search_path, state)
        if fits_done[0]:
            elapsed = time.perf_counter() - started
            per_fit = elapsed / fits_done[0]
            remaining = (total - index - 1) * len(inner) * per_fit
            LOGGER.info(
                "eval %d/%d fitness=%.4f  %s  (%.0fs per fit, about %.0f min left for this fold)",
                index + 1, total, entry["fitness"], params, per_fit, remaining / 60,
            )
        return entry["fitness"]

    out = gwo_maximize(
        evaluate, len(SPACES[arm]), population=population, rounds=rounds, seed=gwo_seed,
        seed_positions=[encode(arm, DEFAULTS[arm])],
    )
    best = evals[str(out["best_index"])]
    state["best"] = {
        "index": out["best_index"],
        "fitness": out["best_fitness"],
        "params": best["params"],
        "default_fitness": evals["0"]["fitness"],
        "default_params": evals["0"]["params"],
        "evaluations": out["evaluations"],
    }
    _save_json(search_path, state)
    LOGGER.info(
        "search done: best fitness %.4f (defaults %.4f) with %s",
        out["best_fitness"], evals["0"]["fitness"], best["params"],
    )

    view, graphs, graph_meta = final
    cfg = PropagationConfig(**{**fixed, **best["params"]})
    result = run_propagation(
        cfg, fold_dir, df=df, view=view, texts=texts, graphs=graphs, device=device,
        tokenizer=tokenizer, graph_meta=graph_meta, split_source=split_source,
    )
    return {"search": state, "result": result}


# ---------------------------------------------------------------------------
def cmd_search(args) -> Dict:
    from src.encoder_preprocess import normalize_tweet
    from src.propagation import load_graphs, restrict_vocab
    from src.train_propagation import ARM_DEFAULTS

    dataset, arm = args.dataset, args.arm
    task = DEFAULT_TASK[dataset]
    df = load_dataset(dataset)
    ids = df["tweet_id"].astype(str).tolist()
    base = _base_view(dataset, task, df)
    folds = Folds.load(dataset, ids)
    folds.check_no_group_leak()
    if not 1 <= args.inner_vals <= folds.n_folds - 1:
        raise SystemExit(f"--inner-vals must be between 1 and {folds.n_folds - 1}")
    run_dir = _run_dir(args.ckpt_root, dataset, task, f"{arm}_gwo", args.tag)
    per_fold_vocab = dataset in FOLD_VOCAB_DATASETS
    graphs, meta = load_graphs(dataset, ids, variant="fullvocab" if per_fold_vocab else None)
    texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]
    searched = {d.name for d in SPACES[arm]}

    def for_view(view):
        if not per_fold_vocab:
            return graphs, None
        return restrict_vocab(graphs, view.train_indices)

    summary = []
    for fold in args.folds:
        fold_dir = run_dir / f"fold{fold}"
        scratch = Path(args.scratch or Path(tempfile.gettempdir()) / "smrd_tune") / run_dir.name / f"fold{fold}"
        if args.fresh:
            shutil.rmtree(fold_dir, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        seed = args.base_seed + fold

        final_view = folds.view(base, fold)
        final_graphs, info = for_view(final_view)
        final_meta = dict(meta)
        if info:
            final_meta["fold_vocab"] = info
        inner = []
        for j in range(args.inner_vals):
            val_fold = (fold + 1 + j) % folds.n_folds
            view = folds.inner_view(base, fold, val_fold)
            inner.append((f"val{val_fold}", view, for_view(view)[0]))

        fixed = {k: v for k, v in ARM_DEFAULTS[arm].items() if k not in searched}
        fixed.update(dataset=dataset, task=task, arm=arm, seed=seed, graph_cache_sha256=meta["sha256"])
        if args.max_epochs:
            fixed["max_epochs"] = args.max_epochs
        LOGGER.info(
            "=== %s fold %d: %d wolves x %d rounds = %d candidates, %d training runs each ===",
            run_dir.name, fold, args.population, args.rounds,
            args.population * (args.rounds + 1), len(inner),
        )
        out = tune_fold(
            fold_dir=fold_dir, scratch_dir=scratch, arm=arm, fixed=fixed,
            final=(final_view, final_graphs, final_meta), inner=inner, df=df, texts=texts,
            population=args.population, rounds=args.rounds, gwo_seed=seed, device=args.device,
            split_source=folds.split_source(fold),
            identity={"folds_sha256": folds.sha256, "held_out_fold": fold},
        )
        shutil.rmtree(scratch, ignore_errors=True)
        best, result = out["search"]["best"], out["result"]
        summary.append({
            "fold": fold,
            "search_fitness_defaults": round(best["default_fitness"], 4),
            "search_fitness_best": round(best["fitness"], 4),
            "best_params": best["params"],
            "test_accuracy": round(result["test_full"]["accuracy"], 4),
            "test_macro_f1": round(result["test_full"]["macro_f1"], 4),
        })
    return {"run": run_dir.name, "folds": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: nested GWO hyperparameter search")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search", help="search + final fit for the requested outer folds")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--arm", choices=TUNE_ARMS, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--population", type=int, default=6, help="wolves")
    p.add_argument("--rounds", type=int, default=5, help="GWO update rounds")
    p.add_argument("--inner-vals", type=int, default=2,
                   help="training runs per candidate, each validating on a different inner fold")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=None, help="override, e.g. 1 for a smoke run (use --tag)")
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    p.add_argument("--scratch", default=None, help="local directory for throw-away search runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--fresh", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    non_default = (args.max_epochs, args.population != 6, args.rounds != 5, args.inner_vals != 2)
    if any(non_default) and not args.tag:
        raise SystemExit("a non-default search budget changes the protocol; pass --tag so it gets its own directory")
    print(json.dumps(cmd_search(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
