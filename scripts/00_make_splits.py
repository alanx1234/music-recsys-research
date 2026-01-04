from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = REPO_ROOT / "data" / "raw" / "tracks.csv"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

N_BINS = 5
RARE_THRESHOLD = 3  


def make_strata(df: pd.DataFrame) -> pd.Series:
    """
    Create a stratification label per song by binning valence and arousal.
    Uses quantile bins so bins are reasonably balanced.
    """
    v = df["valence"].astype(float)
    a = df["arousal"].astype(float)

    # quantile bins; duplicates='drop' handles ties
    v_bin = pd.qcut(v, q=N_BINS, labels=False, duplicates="drop")
    a_bin = pd.qcut(a, q=N_BINS, labels=False, duplicates="drop")

    strata = v_bin.astype(str) + "_" + a_bin.astype(str)
    counts = strata.value_counts()
    rare = counts[counts < RARE_THRESHOLD].index
    strata = strata.where(~strata.isin(rare), other="rare")

    return strata


def write_ids(path: Path, ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{int(sid)}\n")


def main() -> None:
    if not TRACKS_PATH.exists():
        raise FileNotFoundError(f"tracks.csv not found: {TRACKS_PATH}")

    df = pd.read_csv(TRACKS_PATH)
    required = {"song_id", "valence", "arousal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tracks.csv missing columns: {sorted(missing)}")

    df = df.sort_values("song_id").reset_index(drop=True)

    strata = make_strata(df)

    # --- First split: test vs (train+val)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC, random_state=SEED
    )
    idx_all = np.arange(len(df))
    trainval_idx, test_idx = next(sss1.split(idx_all, strata))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    strata_trainval = make_strata(df_trainval)  # recompute strata within trainval

    # --- Second split: val vs train within trainval
    val_size_within_trainval = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_within_trainval, random_state=SEED
    )
    idx_tv = np.arange(len(df_trainval))
    train_idx2, val_idx2 = next(sss2.split(idx_tv, strata_trainval))

    train_ids = df_trainval.iloc[train_idx2]["song_id"].to_numpy()
    val_ids = df_trainval.iloc[val_idx2]["song_id"].to_numpy()
    test_ids = df.iloc[test_idx]["song_id"].to_numpy()

    write_ids(SPLITS_DIR / "train.txt", train_ids)
    write_ids(SPLITS_DIR / "val.txt", val_ids)
    write_ids(SPLITS_DIR / "test.txt", test_ids)

    print(f"Wrote splits to: {SPLITS_DIR}")
    print(f"Counts: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Fractions: train={len(train_ids)/len(df):.3f}, val={len(val_ids)/len(df):.3f}, test={len(test_ids)/len(df):.3f}")

    def mean_va(ids: np.ndarray) -> tuple[float, float]:
        sub = df[df["song_id"].isin(ids)]
        return float(sub["valence"].mean()), float(sub["arousal"].mean())

    tr_v, tr_a = mean_va(train_ids)
    va_v, va_a = mean_va(val_ids)
    te_v, te_a = mean_va(test_ids)
    print(f"Mean (V,A): train=({tr_v:.2f},{tr_a:.2f}) val=({va_v:.2f},{va_a:.2f}) test=({te_v:.2f},{te_a:.2f})")


if __name__ == "__main__":
    main()