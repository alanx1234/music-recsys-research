from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PT = REPO_ROOT / "data" / "processed" / "features_v0_energy.pt"


def assert_no_nan_inf(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise ValueError(f"{name} contains NaN/Inf at indices {bad[:10]} (showing up to 10)")


def main() -> None:
    if not IN_PT.exists():
        raise FileNotFoundError(f"Missing cache file: {IN_PT}")

    blob = torch.load(IN_PT, map_location="cpu", weights_only=False)

    if "data" not in blob:
        raise ValueError(f"Unexpected format in {IN_PT}: missing key 'data'")

    df = blob["data"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Unexpected type for blob['data']: {type(df)}")

    required = {"song_id", "split", "valence", "arousal", "log_rms_mean", "log_rms_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Loaded: {IN_PT}")
    print(f"Rows: {len(df)}")
    print("Split counts:")
    print(df.groupby("split")["song_id"].count().rename("n").to_string())

    n_unique = df["song_id"].nunique()
    if n_unique != len(df):
        dupes = df[df["song_id"].duplicated(keep=False)].sort_values("song_id")
        raise ValueError(f"Duplicate song_id rows found (showing first 10):\n{dupes.head(10)}")
    print(f"Unique song_ids: {n_unique} ")

    allowed_splits = {"train", "val", "test"}
    bad_splits = set(df["split"].unique()) - allowed_splits
    if bad_splits:
        raise ValueError(f"Found unexpected split labels: {bad_splits}")
    print("Split labels OK ")

    for col in ["valence", "arousal", "log_rms_mean", "log_rms_std"]:
        arr = df[col].to_numpy(dtype=np.float64)
        assert_no_nan_inf(col, arr)
    vmin, vmax = float(df["valence"].min()), float(df["valence"].max())
    amin, amax = float(df["arousal"].min()), float(df["arousal"].max())
    print(f"Valence range: {vmin:.3f} .. {vmax:.3f}")
    print(f"Arousal range: {amin:.3f} .. {amax:.3f}")

    if vmin < 0 or vmax > 10 or amin < 0 or amax > 10:
        print("[WARN] Valence/Arousal outside expected ~[1,9] range. Not fatal, but check labels.")

    em_min, em_max = float(df["log_rms_mean"].min()), float(df["log_rms_mean"].max())
    es_min, es_max = float(df["log_rms_std"].min()), float(df["log_rms_std"].max())
    print(f"log_rms_mean range: {em_min:.3f} .. {em_max:.3f}")
    print(f"log_rms_std  range: {es_min:.3f} .. {es_max:.3f}")

    if es_max > 5.0:
        top = df.sort_values("log_rms_std", ascending=False).head(5)[
            ["song_id", "split", "log_rms_mean", "log_rms_std", "audio_path"]
        ]
        print("\n[NOTE] Very large log_rms_std values detected (top 5):")
        print(top.to_string(index=False))
        print("This can happen with long quiet sections or sharp silence→loud transitions.\n")

    def summarize(split: str) -> dict[str, float]:
        sub = df[df["split"] == split]
        return {
            "n": float(len(sub)),
            "V_mean": float(sub["valence"].mean()),
            "A_mean": float(sub["arousal"].mean()),
            "Emean_mean": float(sub["log_rms_mean"].mean()),
            "Estd_mean": float(sub["log_rms_std"].mean()),
        }

    summ = {s: summarize(s) for s in ["train", "val", "test"]}
    print("Per-split means (quick skew check):")
    for s, d in summ.items():
        print(
            f"  {s:5s} n={int(d['n']):4d}  "
            f"V={d['V_mean']:.3f} A={d['A_mean']:.3f}  "
            f"log_rms_mean={d['Emean_mean']:.3f} log_rms_std={d['Estd_mean']:.3f}"
        )

    v = df["valence"].to_numpy(dtype=np.float64)
    a = df["arousal"].to_numpy(dtype=np.float64)
    em = df["log_rms_mean"].to_numpy(dtype=np.float64)
    es = df["log_rms_std"].to_numpy(dtype=np.float64)

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print("\nCorrelations (global):")
    print(f"  corr(log_rms_mean, arousal) = {corr(em, a):.3f}")
    print(f"  corr(log_rms_std,  arousal) = {corr(es, a):.3f}")
    print(f"  corr(log_rms_mean, valence) = {corr(em, v):.3f}")
    print(f"  corr(log_rms_std,  valence) = {corr(es, v):.3f}")

if __name__ == "__main__":
    main()