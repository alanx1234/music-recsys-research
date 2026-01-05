"""
Parse Chordino CSV outputs into key-invariant tokens + transition entropy (CTE).

Input:
  data/processed/chords_chordino_simplechord/{song_id}.csv
  CSV schema: time,chord

Output:
  data/processed/chords_tokens/
    - sequences.csv    (song_id, tonic_pc, n_tokens, tokens_str)
    - transitions.csv  (song_id, prev_tok, next_tok, count)
    - cte.csv          (song_id, n_transitions, cte)
    - chords_tokens_cache.pkl  (dict cache for fast iteration)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import math
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR_DEFAULT = REPO_ROOT / "data" / "processed" / "chords_chordino_simplechord"
OUT_DIR_DEFAULT = REPO_ROOT / "data" / "processed" / "chords_tokens"

LOG = logging.getLogger("parse_chords_and_tokens")



NOTE_TO_PC: Dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}
# Normalize common flats + edge cases to sharps (canonical)
ENHARMONIC: Dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B",  "Fb": "E",  "E#": "F",  "B#": "C",
}

ROOT_RE = re.compile(r"^\s*([A-G])([#b]?)(.*)\s*$")


def normalize_root(root: str) -> Optional[str]:
    """
    Normalize a root like 'Ab' -> 'G#'. Returns canonical sharp spelling.
    """
    root = root.strip()
    if root in ENHARMONIC:
        root = ENHARMONIC[root]
    return root if root in NOTE_TO_PC else None


def parse_root_and_quality(chord_raw: str) -> Optional[Tuple[int, str]]:
    """
    Parse chord label into (root_pc, quality_class).

    Quality classes (compact):
      - maj, min, dom, dim, aug, sus, hdim

    Returns None if chord should be skipped (e.g., 'N' or malformed).
    """
    if chord_raw is None:
        return None

    chord = chord_raw.strip()
    if chord == "" or chord.upper() == "N":
        return None

    # Strip inversion: "A/G" -> "A"
    chord = chord.split("/")[0].strip()
    if chord == "" or chord.upper() == "N":
        return None

    m = ROOT_RE.match(chord)
    if not m:
        return None

    letter, accidental, suffix = m.group(1), m.group(2), m.group(3)
    root = normalize_root(f"{letter}{accidental}")
    if root is None:
        return None

    root_pc = NOTE_TO_PC[root]
    suf = (suffix or "").strip()

    # Determine quality class (robust heuristics for Chordino labels)
    suf_lower = suf.lower()

    # half-diminished patterns (common in chordino): m7b5, min7b5, etc.
    if "m7b5" in suf_lower or "min7b5" in suf_lower:
        qual = "hdim"
    elif "dim" in suf_lower:
        qual = "dim"
    elif "aug" in suf_lower:
        qual = "aug"
    elif "sus" in suf_lower:
        qual = "sus"
    else:
        # maj7 explicitly contains "maj"
        if suf_lower.startswith("m") and not suf_lower.startswith("maj"):
            qual = "min"
        else:
            qual = "maj"

        # dominant 7: 'Bb7' (not maj7, not m7)
        if "7" in suf_lower and qual == "maj" and "maj7" not in suf_lower:
            qual = "dom"

    return root_pc, qual


def rel_token(root_pc: int, tonic_pc: int, qual: str) -> str:
    """
    Relative-root token: interval in semitones from tonic + quality.
    Example: tonic=C(0), chord=G(7) => "7:dom" or "7:maj" depending on qual.
    """
    interval = (root_pc - tonic_pc) % 12
    return f"{interval}:{qual}"


def dedup_consecutive(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    out = [tokens[0]]
    for t in tokens[1:]:
        if t != out[-1]:
            out.append(t)
    return out


def compute_cte(tokens: List[str]) -> Tuple[int, float, Counter[Tuple[str, str]]]:
    """
    Compute Chord Transition Entropy (CTE) for one song.

    Formula (spec):
      CTE = - sum_i pi(i) * sum_j p(j|i) log p(j|i)

    where:
      transitions: x_t -> x_{t+1}
      p(j|i) estimated from counts
      pi(i) = frequency of state i (as previous state), i.e. outgoing_count_i / total_transitions

    Returns:
      n_transitions, cte_value, transition_counts
    """
    if len(tokens) < 2:
        return 0, 0.0, Counter()

    trans = Counter(zip(tokens[:-1], tokens[1:]))
    out_counts = Counter()
    total = 0
    for (a, b), c in trans.items():
        out_counts[a] += c
        total += c

    if total == 0:
        return 0, 0.0, trans

    cte = 0.0
    for a, out_c in out_counts.items():
        pi = out_c / total
        # entropy of outgoing distribution from a
        h = 0.0
        for (aa, bb), c in trans.items():
            if aa != a:
                continue
            p = c / out_c
            h -= p * math.log(p)
        cte += pi * h

    return total, float(cte), trans

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=str(IN_DIR_DEFAULT))
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all).")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate consecutive identical tokens.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_path = out_dir / "sequences.csv"
    trans_path = out_dir / "transitions.csv"
    cte_path = out_dir / "cte.csv"
    cache_path = out_dir / "chords_tokens_cache.pkl"

    if not args.overwrite:
        for p in [seq_path, trans_path, cte_path, cache_path]:
            if p.exists():
                raise FileExistsError(f"Output exists (use --overwrite): {p}")

    files = sorted(in_dir.glob("*.csv"))
    if args.limit > 0:
        files = files[: args.limit]

    LOG.info("Input:  %s (%d files)", in_dir, len(files))
    LOG.info("Output: %s", out_dir)
    LOG.info("dedup=%s", args.dedup)

    sequences_rows: List[Dict[str, object]] = []
    transitions_rows: List[Dict[str, object]] = []
    cte_rows: List[Dict[str, object]] = []

    # cache: song_id -> dict
    cache: Dict[int, Dict[str, object]] = {}

    n_ok = 0
    n_empty = 0

    for idx, fp in enumerate(files, start=1):
        try:
            song_id = int(fp.stem)
        except ValueError:
            LOG.warning("Skipping non-numeric filename: %s", fp.name)
            continue

        df = pd.read_csv(fp)
        if not {"time", "chord"}.issubset(df.columns):
            LOG.warning("Bad schema (expected time,chord): %s", fp.name)
            continue

        parsed: List[Tuple[int, str]] = []
        for chord_raw in df["chord"].astype(str).tolist():
            out = parse_root_and_quality(chord_raw)
            if out is None:
                continue
            parsed.append(out)

        if not parsed:
            n_empty += 1
            cache[song_id] = {
                "song_id": song_id,
                "tonic_pc": None,
                "tokens": [],
                "n_tokens": 0,
                "n_transitions": 0,
                "cte": 0.0,
            }
            sequences_rows.append({
                "song_id": song_id,
                "tonic_pc": "",
                "n_tokens": 0,
                "tokens_str": "",
            })
            cte_rows.append({
                "song_id": song_id,
                "n_transitions": 0,
                "cte": 0.0,
            })
            continue

        tonic_pc = parsed[0][0]  
        tokens = [rel_token(pc, tonic_pc, qual) for (pc, qual) in parsed]
        if args.dedup:
            tokens = dedup_consecutive(tokens)

        n_trans, cte_val, trans_counts = compute_cte(tokens)

        # Save per-song rows
        sequences_rows.append({
            "song_id": song_id,
            "tonic_pc": tonic_pc,
            "n_tokens": len(tokens),
            "tokens_str": " ".join(tokens),
        })

        cte_rows.append({
            "song_id": song_id,
            "n_transitions": n_trans,
            "cte": cte_val,
        })

        for (a, b), c in trans_counts.items():
            transitions_rows.append({
                "song_id": song_id,
                "prev_tok": a,
                "next_tok": b,
                "count": int(c),
            })

        cache[song_id] = {
            "song_id": song_id,
            "tonic_pc": tonic_pc,
            "tokens": tokens,
            "n_tokens": len(tokens),
            "n_transitions": n_trans,
            "cte": cte_val,
        }

        n_ok += 1
        if idx % 200 == 0:
            LOG.info("Processed %d/%d files...", idx, len(files))

    pd.DataFrame(sequences_rows).sort_values("song_id").to_csv(seq_path, index=False)
    pd.DataFrame(transitions_rows).sort_values(["song_id", "prev_tok", "next_tok"]).to_csv(trans_path, index=False)
    pd.DataFrame(cte_rows).sort_values("song_id").to_csv(cte_path, index=False)

    with cache_path.open("wb") as f:
        pickle.dump(cache, f)

    LOG.info("Done.")
    LOG.info("Songs ok: %d", n_ok)
    LOG.info("Songs empty (no parsed chords): %d", n_empty)
    LOG.info("Wrote: %s", seq_path)
    LOG.info("Wrote: %s", trans_path)
    LOG.info("Wrote: %s", cte_path)
    LOG.info("Wrote: %s", cache_path)


if __name__ == "__main__":
    main()