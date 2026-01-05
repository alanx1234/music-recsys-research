"""
Evaluate retrieval tradeoffs (qualitative metrics) for VA-only vs symbolic-augmented scoring.

Splits:
  data/splits/train.txt
  data/splits/val.txt
  data/splits/test.txt

Inputs:
  data/processed/features_v0_energy.pt
  data/processed/chords_tokens/cte.csv
  data/processed/chords_tokens/ngrams_cache.pkl
  experiments/.../run_###/model.pt

Optional:
  experiments/.../run_###/preds.csv (may be test-only)

Outputs (in run_dir):
  retrieval_tradeoff_qual.json
  retrieval_tradeoff_qual.csv
  preds_all.csv  (cached predictions for all songs)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PT = REPO_ROOT / "data" / "processed" / "features_v0_energy.pt"

SPLITS_DIR = REPO_ROOT / "data" / "splits"
TRAIN_TXT = SPLITS_DIR / "train.txt"
VAL_TXT = SPLITS_DIR / "val.txt"
TEST_TXT = SPLITS_DIR / "test.txt"

CTE_CSV = REPO_ROOT / "data" / "processed" / "chords_tokens" / "cte.csv"
NGRAMS_CACHE = REPO_ROOT / "data" / "processed" / "chords_tokens" / "ngrams_cache.pkl"

LOG = logging.getLogger("eval_tradeoff_qual_metrics")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _torch_load_any(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_ids(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    ids: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return np.array(ids, dtype=np.int64)


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    train_ids = set(read_ids(TRAIN_TXT).tolist())
    val_ids = set(read_ids(VAL_TXT).tolist())
    test_ids = set(read_ids(TEST_TXT).tolist())

    def which_split(sid: int) -> str:
        if sid in train_ids:
            return "train"
        if sid in val_ids:
            return "val"
        if sid in test_ids:
            return "test"
        return "unknown"

    out = df.copy()
    out["split"] = out["song_id"].astype(int).map(which_split)
    n_unknown = int((out["split"] == "unknown").sum())
    if n_unknown > 0:
        LOG.warning("Found %d songs not present in split txt files. Dropping them.", n_unknown)
        out = out[out["split"] != "unknown"].copy()
    return out


def load_features_df(path: Path) -> pd.DataFrame:
    pack = _torch_load_any(path)

    if isinstance(pack, dict) and "data" in pack:
        df = pack["data"]
    elif isinstance(pack, dict) and "df" in pack:
        df = pack["df"]
    elif isinstance(pack, pd.DataFrame):
        df = pack
    else:
        raise TypeError(f"Unrecognized .pt format at {path}")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("features_v0_energy.pt did not contain a DataFrame")

    need = {"song_id", "valence", "arousal"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"features DataFrame missing {missing}")

    out = df.copy()
    out["song_id"] = out["song_id"].astype(int)
    out["valence"] = out["valence"].astype(float)
    out["arousal"] = out["arousal"].astype(float)
    return out


def load_cte(cte_csv: Path) -> pd.DataFrame:
    if not cte_csv.exists():
        raise FileNotFoundError(f"Missing: {cte_csv}")
    df = pd.read_csv(cte_csv)
    if not {"song_id", "cte"}.issubset(df.columns):
        raise ValueError(f"cte.csv must contain song_id, cte. Have: {list(df.columns)}")
    df["song_id"] = df["song_id"].astype(int)
    df["cte"] = df["cte"].astype(float)
    return df[["song_id", "cte"]]


def load_ngrams(cache_path: Path, expected_n: Optional[int] = None) -> Tuple[int, Dict[int, set]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing: {cache_path}")
    with cache_path.open("rb") as f:
        pack = pickle.load(f)
    n = int(pack.get("n", 0))
    grams = pack.get("ngrams", {})
    grams = {int(k): set(v) for k, v in grams.items()}
    if expected_n is not None and n != expected_n:
        LOG.warning("ngrams_cache n=%d but expected n=%d (continuing).", n, expected_n)
    return n, grams


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    uni = len(a | b)
    return float(inter) / float(uni) if uni > 0 else 0.0


def parse_k_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("k_list is empty")
    return sorted(set(out))


def try_load_preds_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = {"song_id", "pred_valence", "pred_arousal"}
    if not need.issubset(df.columns):
        return None
    df = df.copy()
    df["song_id"] = df["song_id"].astype(int)
    df["pred_valence"] = df["pred_valence"].astype(float)
    df["pred_arousal"] = df["pred_arousal"].astype(float)
    return df[["song_id", "pred_valence", "pred_arousal"]]


class AcousticMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_all_from_model(run_dir: Path, feats: pd.DataFrame) -> pd.DataFrame:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    ckpt = _torch_load_any(model_path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("model.pt missing model_state_dict")

    x_cols = list(ckpt.get("x_cols", []))
    if not x_cols:
        raise ValueError("model.pt missing x_cols")
    for c in x_cols:
        if c not in feats.columns:
            raise ValueError(f"features missing required x_col: {c}")

    state = ckpt["model_state_dict"]
    w0 = state["net.0.weight"]
    d_hidden = int(w0.shape[0])
    d_in = int(w0.shape[1])

    scaler_mean = torch.tensor(ckpt["scaler_mean"], dtype=torch.float32)
    scaler_std = torch.tensor(ckpt["scaler_std"], dtype=torch.float32).clamp_min(1e-6)

    model = AcousticMLP(d_in=d_in, d_hidden=d_hidden)
    model.load_state_dict(state)
    model.eval()

    x = feats[x_cols].to_numpy(dtype=np.float32)
    xt = torch.from_numpy(x)
    xt = (xt - scaler_mean) / scaler_std

    yhat = model(xt).cpu().numpy()

    out = feats[["song_id"]].copy()
    out["pred_valence"] = yhat[:, 0].astype(float)
    out["pred_arousal"] = yhat[:, 1].astype(float)
    return out


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    p.add_argument("--query_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--k_list", type=str, default="5,10,20")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--ngram_n", type=int, default=4)
    args = p.parse_args()

    run_dir = REPO_ROOT / Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run_dir: {run_dir}")

    k_list = parse_k_list(args.k_list)
    beta = float(args.beta)
    gamma = float(args.gamma)

    feats = load_features_df(FEATURES_PT)
    feats = add_split_column(feats)

    cte = load_cte(CTE_CSV)
    n_cache, grams = load_ngrams(NGRAMS_CACHE, expected_n=args.ngram_n)
    LOG.info("Loaded n-grams cache: n=%d", n_cache)

    preds_all_path = run_dir / "preds_all.csv"
    preds = try_load_preds_csv(preds_all_path)

    if preds is None:
        preds = try_load_preds_csv(run_dir / "preds.csv")

    if preds is None or preds["song_id"].nunique() < feats["song_id"].nunique() * 0.9:
        LOG.info("Generating preds_all.csv from model.pt (preds.csv is missing or incomplete).")
        preds = predict_all_from_model(run_dir, feats)
        preds.to_csv(preds_all_path, index=False)
        LOG.info("Wrote: %s", preds_all_path)

    df = feats.merge(preds, on="song_id", how="inner").merge(cte, on="song_id", how="left")
    df["cte"] = df["cte"].fillna(0.0).astype(float)

    queries_df = df[df["split"] == args.query_split].copy()
    if args.candidates == "all":
        cand_df = df.copy()
    else:
        cand_df = df[df["split"].isin(["train", "val"])].copy()

    LOG.info("Queries: %d (%s)", len(queries_df), args.query_split)
    LOG.info("Candidates: %d (%s)", len(cand_df), args.candidates)
    LOG.info("beta=%.4f gamma=%.4f", beta, gamma)

    if len(queries_df) == 0:
        raise RuntimeError("No queries found. Check split txt files and --query_split.")
    if len(cand_df) == 0:
        raise RuntimeError("Candidate set empty. Check split txt files and --candidates.")

    cand_ids = cand_df["song_id"].to_numpy(dtype=np.int64)
    cand_pred = cand_df[["pred_valence", "pred_arousal"]].to_numpy(dtype=np.float32)
    cand_gt = cand_df[["valence", "arousal"]].to_numpy(dtype=np.float32)
    cand_cte = cand_df["cte"].to_numpy(dtype=np.float32)

    variants = [
        ("va_only", 0.0, 0.0),
        ("va_plus_cte", beta, 0.0),
        ("va_plus_ngram", 0.0, gamma),
        ("full", beta, gamma),
    ]

    rows: List[Dict[str, object]] = []

    for variant, b_cte, g_ng in variants:
        for K in k_list:
            sum_gt = 0.0
            sum_cte = 0.0
            sum_ng = 0.0
            n_total = 0

            for _, q in queries_df.iterrows():
                qid = int(q["song_id"])
                q_pred = np.array([q["pred_valence"], q["pred_arousal"]], dtype=np.float32)
                q_gt = np.array([q["valence"], q["arousal"]], dtype=np.float32)
                q_cte = float(q["cte"])
                q_grams = grams.get(qid, set())

                mask = cand_ids != qid
                ids = cand_ids[mask]
                pred = cand_pred[mask]
                gt = cand_gt[mask]
                cte_v = cand_cte[mask]

                d_va = np.linalg.norm(pred - q_pred[None, :], axis=1)
                score = -d_va

                if b_cte != 0.0:
                    d_cte = np.abs(cte_v - q_cte)
                    score += (-b_cte * d_cte)

                if g_ng != 0.0:
                    sim = np.zeros(len(ids), dtype=np.float32)
                    for i, sid in enumerate(ids):
                        sim[i] = jaccard(q_grams, grams.get(int(sid), set()))
                    score += (g_ng * sim)
                else:
                    sim = np.zeros(len(ids), dtype=np.float32)

                topk = np.argpartition(score, -K)[-K:]
                topk = topk[np.argsort(score[topk])[::-1]]

                gt_dist = np.linalg.norm(gt[topk] - q_gt[None, :], axis=1)
                sum_gt += float(gt_dist.sum())
                sum_cte += float(np.abs(cte_v[topk] - q_cte).sum())
                sum_ng += float(sim[topk].sum())
                n_total += int(K)

            rows.append(
                {
                    "variant": variant,
                    "beta_cte": float(b_cte),
                    "gamma_ngram": float(g_ng),
                    "K": int(K),
                    "mean_gt_va_dist": sum_gt / n_total,
                    "mean_abs_cte_diff": sum_cte / n_total,
                    "mean_ngram_sim": sum_ng / n_total,
                    "n_queries": int(len(queries_df)),
                    "n_candidates": int(len(cand_df) - 1),
                }
            )

    out_json = run_dir / "retrieval_tradeoff_qual.json"
    out_csv = run_dir / "retrieval_tradeoff_qual.csv"

    payload = {
        "run_dir": str(Path(args.run_dir)),
        "candidates": args.candidates,
        "k_list": k_list,
        "beta": beta,
        "gamma": gamma,
        "ngram_n": int(n_cache),
        "n_queries": int(len(queries_df)),
        "n_candidates": int(len(cand_df) - 1),
        "rows": rows,
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    LOG.info("Wrote: %s", out_json)
    LOG.info("Wrote: %s", out_csv)


if __name__ == "__main__":
    main()
