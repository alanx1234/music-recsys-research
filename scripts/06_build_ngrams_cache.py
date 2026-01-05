"""
Build per-song n-gram sets from token sequences for grammar similarity.

Input:
  data/processed/chords_tokens/chords_tokens_cache.pkl

Output:
  data/processed/chords_tokens/ngrams_cache.pkl

N-gram definition:
  Given token sequence t0..t{L-1}, build set of tuples:
    (t_i, ..., t_{i+n-1}) for i=0..L-n
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import pickle
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TOK_DIR_DEFAULT = REPO_ROOT / "data" / "processed" / "chords_tokens"
IN_CACHE_DEFAULT = TOK_DIR_DEFAULT / "chords_tokens_cache.pkl"
OUT_CACHE_DEFAULT = TOK_DIR_DEFAULT / "ngrams_cache.pkl"

LOG = logging.getLogger("build_ngrams_cache")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def make_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_cache", type=str, default=str(IN_CACHE_DEFAULT))
    parser.add_argument("--out_cache", type=str, default=str(OUT_CACHE_DEFAULT))
    parser.add_argument("--n", type=int, default=4, help="n-gram size (e.g., 3 or 4).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.in_cache)
    out_path = Path(args.out_cache)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Exists (use --overwrite): {out_path}")

    with in_path.open("rb") as f:
        cache: Dict[int, Dict[str, object]] = pickle.load(f)

    ngrams_cache: Dict[int, Set[Tuple[str, ...]]] = {}
    n_empty = 0
    for song_id, d in cache.items():
        tokens = d.get("tokens", [])
        if not isinstance(tokens, list):
            tokens = list(tokens)
        grams = make_ngrams(tokens, args.n)
        if len(grams) == 0:
            n_empty += 1
        ngrams_cache[int(song_id)] = grams

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "n": args.n,
                "ngrams": ngrams_cache,
            },
            f,
        )

    LOG.info("Built n-grams: n=%d", args.n)
    LOG.info("Songs total: %d", len(ngrams_cache))
    LOG.info("Songs with empty n-gram set: %d", n_empty)
    LOG.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()