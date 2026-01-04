from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANN_DIR = REPO_ROOT / "data" / "raw" / "deam" / "annotations"
AUDIO_DIR = REPO_ROOT / "data" / "raw" / "deam" / "audio"
OUT_PATH = REPO_ROOT / "data" / "raw" / "tracks.csv"


def main() -> None:
    if not ANN_DIR.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ANN_DIR}")
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio dir not found: {AUDIO_DIR}")

    ann_files = sorted(ANN_DIR.glob("static_annotations_averaged_songs_*.csv"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation CSVs found in {ANN_DIR}")

    dfs = []
    for f in ann_files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    ann = pd.concat(dfs, ignore_index=True)

    required = ["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]
    missing = [c for c in required if c not in ann.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}.\nFound columns: {list(ann.columns)}"
        )

    ann["song_id"] = ann["song_id"].astype(int)
    audio_abs = ann["song_id"].apply(lambda sid: AUDIO_DIR / f"{sid}.mp3")
    ann["audio_path_abs"] = audio_abs
    ann["has_audio"] = ann["audio_path_abs"].apply(lambda p: p.exists())

    before = len(ann)
    ann = ann[ann["has_audio"]].copy()
    after = len(ann)

    out = pd.DataFrame({
        "song_id": ann["song_id"].astype(int),
        "audio_path": ann["audio_path_abs"].apply(lambda p: p.relative_to(REPO_ROOT).as_posix()),
        "valence": ann["valence_mean"].astype(float),
        "arousal": ann["arousal_mean"].astype(float),
        "valence_std": ann["valence_std"].astype(float),
        "arousal_std": ann["arousal_std"].astype(float),
    }).sort_values("song_id").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote: {OUT_PATH}")
    print(f"Annotation rows: {before}")
    print(f"With audio:       {after}")
    print(f"song_id range:    {out.song_id.min()} .. {out.song_id.max()}")
    print("First 5 rows:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()