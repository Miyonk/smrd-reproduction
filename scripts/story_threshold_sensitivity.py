"""
How much does story-held-out accuracy depend on the story-clustering threshold?

Twitter15/16 have no event ids, so the ``story`` folds (``src/cv.py``) group
source tweets by text similarity.  This rebuilds those folds at several cosine
thresholds (in a temporary directory; the committed fold files are not
touched) and runs the lexical arm on each.  CPU only, a few minutes.

    python scripts/story_threshold_sensitivity.py [--thresholds 0.5 0.4 0.3 0.2 0.1]
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cv import Folds, build_folds_file
from src.data import load_dataset
from src.encoder_preprocess import normalize_tweet
from src.lexical import LexicalConfig, run_lexical, thread_bags
from src.propagation import load_graphs
from src.run_cv import DEFAULT_TASK, _base_view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--datasets", nargs="+", default=["twitter16", "twitter15"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.4, 0.3, 0.2, 0.1])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    tmp = Path(tempfile.mkdtemp(prefix="story_sensitivity_"))
    for ds in args.datasets:
        df = load_dataset(ds)
        ids = df["tweet_id"].astype(str).tolist()
        base = _base_view(ds, DEFAULT_TASK[ds], df)
        graphs, meta = load_graphs(ds, ids)
        bags = thread_bags(graphs)
        texts = [normalize_tweet(t) for t in df["raw_text"].astype(str).tolist()]
        for thr in args.thresholds:
            path = tmp / f"{ds}_story_{thr}.json"
            payload = build_folds_file(ds, df, path=path, split="story", story_threshold=thr)
            folds = Folds.load(ds, ids, path=path, split="story")
            folds.check_no_group_leak()
            sizes = collections.Counter(payload["group_of_row"])
            correct = n = 0
            for k in range(folds.n_folds):
                result = run_lexical(
                    LexicalConfig(dataset=ds, task=DEFAULT_TASK[ds], graph_cache_sha256=meta["sha256"]),
                    tmp / f"{ds}_{thr}" / f"fold{k}", df=df, view=folds.view(base, k), texts=texts, bags=bags,
                    split_source=f"{path.name} fold {k}",
                )
                correct += sum(p["label"] == p["pred"] for p in result["test_predictions"])
                n += len(result["test_predictions"])
            print(
                f"{ds} cosine>={thr}: {len(sizes)} stories, largest {max(sizes.values())}, "
                f"{sum(v for v in sizes.values() if v > 1)} tweets in multi-tweet stories | "
                f"lexical accuracy {correct / n:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
