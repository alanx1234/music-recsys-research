"""
Batch extract chord sequences using Chordino (Vamp) via Sonic Annotator.

Input:
  data/raw/tracks.csv

Output:
  data/processed/chords_chordino_simplechord/{song_id}.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import pandas as pd
import re
import subprocess
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = REPO_ROOT / "data" / "raw" / "tracks.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "chords_chordino_simplechord"

PLUGIN = "vamp:nnls-chroma:chordino:simplechord"

ROW_RE = re.compile(
    r'^\s*(?:"[^"]*"|)?\s*,?\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"([^"]*)"\s*$'
)

def run_sonic(sonic_exe: str, audio_path: Path) -> List[Tuple[float, str]]:
    cmd = [
        sonic_exe,
        "--csv-force",
        "-d", PLUGIN,
        "-w", "csv",
        str(audio_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "sonic-annotator failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RETURN_CODE: {p.returncode}\n"
            "---- STDOUT ----\n"
            f"{p.stdout}\n"
            "---- STDERR ----\n"
            f"{p.stderr}\n"
        )

    stem = audio_path.with_suffix("").name
    vamp_csv = audio_path.with_name(f"{stem}_vamp_nnls-chroma_chordino_simplechord.csv")

    if not vamp_csv.exists():
        raise RuntimeError(f"Expected vamp output CSV not found: {vamp_csv}")

    rows: List[Tuple[float, str]] = []
    for line in vamp_csv.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        t = float(m.group(1))
        chord = m.group(2)
        rows.append((t, chord))

    # clean up temp file so raw/audio doesn't fill up
    vamp_csv.unlink(missing_ok=True)

    return rows



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sonic",
        type=str,
        default="sonic-annotator",
        help="Path to sonic-annotator.exe or 'sonic-annotator' if on PATH.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N songs (0 = all).")
    args = parser.parse_args()

    if not TRACKS_PATH.exists():
        raise FileNotFoundError(f"Missing: {TRACKS_PATH}")

    df = pd.read_csv(TRACKS_PATH)
    if not {"song_id", "audio_path"}.issubset(df.columns):
        raise ValueError("tracks.csv must contain song_id and audio_path columns")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(df) if args.limit <= 0 else min(len(df), args.limit)
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, row in df.head(total).iterrows():
        sid = int(row["song_id"])
        audio_rel = Path(str(row["audio_path"]))
        audio_abs = REPO_ROOT / audio_rel

        out_path = OUT_DIR / f"{sid}.csv"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            rows = run_sonic(args.sonic, audio_abs)

            with out_path.open("w", encoding="utf-8", newline="") as f:
                f.write("time,chord\n")
                for t, chord in rows:
                    f.write(f"{t:.9f},{chord}\n")

            n_ok += 1
            if n_ok % 50 == 0:
                print(f"[OK] extracted {n_ok} songs (skipped={n_skip}, failed={n_fail})")
        except subprocess.CalledProcessError as e:
            n_fail += 1
            err_path = OUT_DIR / f"{sid}.error.txt"

            msg = []
            msg.append(f"CMD: {' '.join(cmd for cmd in e.cmd)}" if isinstance(e.cmd, list) else f"CMD: {e.cmd}")
            msg.append(f"RETURN_CODE: {e.returncode}")
            msg.append("---- OUTPUT (stdout+stderr) ----")
            msg.append(e.output or "<empty>")
            err_path.write_text("\n".join(msg), encoding="utf-8")

            print(f"[FAIL] song_id={sid} -> {err_path.name}")

    print("\nDone.")
    print(f"Processed: {total}")
    print(f"OK:        {n_ok}")
    print(f"Skipped:   {n_skip}")
    print(f"Failed:    {n_fail}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()