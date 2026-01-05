"""
Find query songs (prefer test split) where n-gram Jaccard overlap is nonzero,
so the grammar term actually affects ranking (good for qualitative Figure 2).

Reads:
  data/processed/features_v0_energy.csv
  data/processed/chords_tokens/ngrams_cache.pkl

Prints:
  top queries by max overlap and count of overlapping candidates
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("find_good_qual_examples")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a) + len(b) - inter
    return float(inter / uni) if uni > 0 else 0.0


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", type=str, default=str(REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"))
    p.add_argument("--ngrams_cache", type=str, default=str(REPO_ROOT / "data" / "processed" / "chords_tokens" / "ngrams_cache.pkl"))
    p.add_argument("--query_split", type=str, default="test", choices=["test", "val", "train", "all"])
    p.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    p.add_argument("--min_nonzero", type=int, default=1, help="Require at least this many candidates with sim>0")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    features_csv = Path(args.features_csv)
    ngrams_cache = Path(args.ngrams_cache)

    if not features_csv.exists():
        raise FileNotFoundError(f"Missing: {features_csv}")
    if not ngrams_cache.exists():
        raise FileNotFoundError(f"Missing: {ngrams_cache} (run 06_build_ngrams_cache.py)")

    df = pd.read_csv(features_csv)
    if not {"song_id", "split"}.issubset(df.columns):
        raise ValueError("features csv must contain song_id and split")

    with ngrams_cache.open("rb") as f:
        payload = pickle.load(f)

    # supports both {"n":..., "ngrams": {...}} and raw dict forms
    if isinstance(payload, dict) and "ngrams" in payload:
        n = int(payload.get("n", 0))
        grams_map: Dict[int, Set[Tuple[str, ...]]] = payload["ngrams"]
    else:
        n = -1
        grams_map = payload

    df["song_id"] = df["song_id"].astype(int)

    if args.query_split == "all":
        q_ids = df["song_id"].tolist()
    else:
        q_ids = df[df["split"] == args.query_split]["song_id"].tolist()

    if args.candidates == "trainval":
        c_ids = df[df["split"].isin(["train", "val"])]["song_id"].tolist()
    else:
        c_ids = df["song_id"].tolist()

    LOG.info("Loaded n-grams cache: n=%s", str(n))
    LOG.info("Queries: %d (%s), Candidates: %d (%s)", len(q_ids), args.query_split, len(c_ids), args.candidates)

    rows = []
    c_ids_arr = np.array(c_ids, dtype=np.int32)

    for qid in q_ids:
        qg = grams_map.get(int(qid), set())
        if not qg:
            continue

        # exclude self if present
        cand = c_ids_arr[c_ids_arr != int(qid)]

        max_sim = 0.0
        count_nonzero = 0
        best_sid = None

        for sid in cand:
            sg = grams_map.get(int(sid), set())
            if not sg:
                continue
            sim = jaccard(qg, sg)
            if sim > 0.0:
                count_nonzero += 1
                if sim > max_sim:
                    max_sim = sim
                    best_sid = int(sid)

        if count_nonzero >= args.min_nonzero:
            rows.append(
                {
                    "query_song_id": int(qid),
                    "max_sim_ngram": float(max_sim),
                    "count_nonzero": int(count_nonzero),
                    "best_match_song_id": int(best_sid) if best_sid is not None else -1,
                }
            )

    if not rows:
        LOG.info("No queries found with nonzero overlaps under current settings.")
        LOG.info("If n=4 is too strict, try rebuilding ngrams with n=3 and rerun this script.")
        return

    out = pd.DataFrame(rows).sort_values(["max_sim_ngram", "count_nonzero"], ascending=False)
    print(out.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
