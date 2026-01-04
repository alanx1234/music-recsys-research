from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import librosa


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = REPO_ROOT / "data" / "raw" / "tracks.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
OUT_DIR = REPO_ROOT / "data" / "processed"
OUT_PT = OUT_DIR / "features_v0_energy.pt"
OUT_CSV = OUT_DIR / "features_v0_energy.csv"

# energy extraction settings 
SR = None            
HOP_LENGTH = 512
FRAME_LENGTH = 2048
EPS = 1e-12          


def read_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(int(line))
    return ids


def assign_split(song_id: int, train: set[int], val: set[int], test: set[int]) -> str:
    if song_id in train:
        return "train"
    if song_id in val:
        return "val"
    if song_id in test:
        return "test"
    return "unknown"


def compute_log_rms_stats(audio_path: Path) -> tuple[float, float]:
    """
    Load audio and compute mean/std of log RMS energy over frames.

    Returns:
      (log_rms_mean, log_rms_std) as floats.
    """
    y, sr = librosa.load(audio_path.as_posix(), sr=SR, mono=True)

    if y is None or len(y) == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]  # shape: [T]
    
    log_rms = np.log(rms + EPS)  # shape: [T]

    mu = float(np.mean(log_rms, dtype=np.float64))
    sd = float(np.std(log_rms, dtype=np.float64))
    if not (math.isfinite(mu) and math.isfinite(sd)):
        raise ValueError(f"Non-finite RMS stats for {audio_path}: mean={mu}, std={sd}")
    return mu, sd


def main() -> None:
    if not TRACKS_PATH.exists():
        raise FileNotFoundError(f"tracks.csv not found: {TRACKS_PATH}")

    train_ids = read_ids(SPLITS_DIR / "train.txt")
    val_ids = read_ids(SPLITS_DIR / "val.txt")
    test_ids = read_ids(SPLITS_DIR / "test.txt")

    df = pd.read_csv(TRACKS_PATH)
    required = {"song_id", "audio_path", "valence", "arousal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tracks.csv missing columns: {sorted(missing)}")

    rows = []
    bad = 0

    for i, r in df.iterrows():
        sid = int(r["song_id"])
        split = assign_split(sid, train_ids, val_ids, test_ids)
        if split == "unknown":
            continue  

        audio_path = REPO_ROOT / str(r["audio_path"])
        try:
            log_rms_mean, log_rms_std = compute_log_rms_stats(audio_path)
        except Exception as e:
            bad += 1
            print(f"[WARN] failed song_id={sid} path={audio_path}: {e}")
            continue

        rows.append({
            "song_id": sid,
            "split": split,
            "audio_path": str(r["audio_path"]),
            "valence": float(r["valence"]),
            "arousal": float(r["arousal"]),
            "log_rms_mean": float(log_rms_mean),
            "log_rms_std": float(log_rms_std),
        })

    out_df = pd.DataFrame(rows).sort_values(["split", "song_id"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    blob = {
        "schema": {
            "features": ["log_rms_mean", "log_rms_std"],
            "targets": ["valence", "arousal"],
            "splits": ["train", "val", "test"],
            "energy": {
                "sr": SR,
                "frame_length": FRAME_LENGTH,
                "hop_length": HOP_LENGTH,
                "eps": EPS,
                "log": "natural",
            },
        },
        "data": out_df,
    }
    torch.save(blob, OUT_PT)

    print(f"Wrote: {OUT_PT}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Rows kept: {len(out_df)} (dropped {bad} due to errors)")
    print(out_df.groupby("split")[["song_id"]].count().rename(columns={"song_id": "n"}))
    print("Feature stats:")
    print(out_df[["log_rms_mean", "log_rms_std"]].describe().loc[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()