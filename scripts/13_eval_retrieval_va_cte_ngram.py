"""
Evaluate retrieval with:
- predicted VA distance (primary)
- complexity similarity via CTE (Layer 2)
- grammar similarity via n-gram Jaccard on token sequences (Layer 1)

Inputs:
  experiments/acoustic_mlp/run_###/model.pt
  data/processed/features_v0_energy.csv          (GT VA + splits + energy features)
  data/processed/chords_tokens/chords_tokens_cache.pkl  (CTE + tokens)
  data/processed/chords_tokens/ngrams_cache.pkl         (optional precomputed n-grams)

Outputs:
  experiments/acoustic_mlp/run_###/retrieval_metrics_symbolic.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import math
import pickle
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "acoustic_mlp"
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"
TOK_CACHE = REPO_ROOT / "data" / "processed" / "chords_tokens" / "chords_tokens_cache.pkl"
NGRAMS_CACHE = REPO_ROOT / "data" / "processed" / "chords_tokens" / "ngrams_cache.pkl"

LOG = logging.getLogger("eval_retrieval_va_cte_ngram")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def find_latest_run() -> Path:
    runs = sorted([p for p in EXP_ROOT.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {EXP_ROOT}")
    return runs[-1]


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [2], b: [N,2] -> [N]
    d = b - a[None, :]
    return np.sqrt(np.sum(d * d, axis=1))


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a) + len(b) - inter
    return float(inter / uni) if uni > 0 else 0.0


def precision_at_k(rel: np.ndarray, k: int) -> float:
    return float(np.mean(rel[:k])) if k > 0 else 0.0


def recall_at_k(rel: np.ndarray, k: int, total_rel: int) -> float:
    if total_rel <= 0:
        return 0.0
    return float(np.sum(rel[:k]) / total_rel)


def ideal_dcg_at_k(total_relevant: int, k: int) -> float:
    m = min(total_relevant, k)
    if m <= 0:
        return 0.0
    denom = np.log2(np.arange(2, m + 2))
    return float(np.sum(np.ones(m) / denom))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="", help="Optional: experiments/acoustic_mlp/run_###")
    parser.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    parser.add_argument("--relevant_quantile", type=float, default=0.05)
    parser.add_argument("--k_list", type=str, default="5,10,20")
    parser.add_argument("--beta_cte", type=float, default=0.25, help="Weight for |CTE_q - CTE_s| penalty.")
    parser.add_argument("--gamma_ngram", type=float, default=0.50, help="Weight for n-gram Jaccard bonus.")
    parser.add_argument("--use_ngrams_cache", action="store_true", help="Use precomputed n-grams cache if present.")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seed_everything(args.seed)

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_CSV}")
    if not TOK_CACHE.exists():
        raise FileNotFoundError(f"Missing: {TOK_CACHE}")

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run_dir: {run_dir}")

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

    # --- Load features + model and run inference for ALL songs (same as baseline) ---
    df_feat = pd.read_csv(FEATURES_CSV)
    required_feat = {"song_id", "split", "valence", "arousal"}
    if not required_feat.issubset(df_feat.columns):
        raise ValueError(f"features CSV missing: {sorted(required_feat - set(df_feat.columns))}")

    import torch
    import torch.nn as nn

    class AcousticMLP(nn.Module):
        def __init__(self, d_in: int = 2, d_hidden: int = 64, dropout: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    scaler_mean = ckpt["scaler_mean"].float()  # Shape: [2]
    scaler_std = ckpt["scaler_std"].float().clamp_min(1e-6)  # Shape: [2]
    x_cols = ckpt.get("x_cols", ["log_rms_mean", "log_rms_std"])

    model = AcousticMLP(d_in=2, d_hidden=64, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_all = torch.from_numpy(df_feat[x_cols].to_numpy(np.float32))          # Shape: [N,2]
    X_all = (X_all - scaler_mean[None, :]) / scaler_std[None, :]            # Shape: [N,2]
    with torch.no_grad():
        Yhat_all = model(X_all).numpy()                                     # Shape: [N,2]

    df_all = df_feat[["song_id", "split", "valence", "arousal"]].copy()
    df_all["pred_valence"] = Yhat_all[:, 0]
    df_all["pred_arousal"] = Yhat_all[:, 1]

    with TOK_CACHE.open("rb") as f:
        tok_cache: Dict[int, Dict[str, object]] = pickle.load(f)

    # map song_id -> cte (default 0.0 if missing)
    cte_map: Dict[int, float] = {}
    tokens_map: Dict[int, List[str]] = {}
    for sid, d in tok_cache.items():
        sid_i = int(sid)
        cte_map[sid_i] = float(d.get("cte", 0.0))
        tokens = d.get("tokens", [])
        tokens_map[sid_i] = list(tokens) if isinstance(tokens, list) else list(tokens)

    ngrams_map: Dict[int, Set[Tuple[str, ...]]] = {}

    if args.use_ngrams_cache:
        if not NGRAMS_CACHE.exists():
            raise FileNotFoundError(f"Missing ngrams cache (run script 06): {NGRAMS_CACHE}")
        with NGRAMS_CACHE.open("rb") as f:
            payload = pickle.load(f)
        ngrams_map = payload["ngrams"]
        n = int(payload["n"])
        LOG.info("Loaded n-grams cache: n=%d", n)
    else:
        n = 4
        for sid, toks in tokens_map.items():
            if len(toks) < n:
                ngrams_map[sid] = set()
            else:
                ngrams_map[sid] = {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}
        LOG.info("Built n-grams on the fly: n=%d", n)

    df_q = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    if args.candidates == "trainval":
        df_cand = df_all[df_all["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    else:
        df_cand = df_all.copy().reset_index(drop=True)

    cand_ids = df_cand["song_id"].to_numpy(np.int32)
    cand_gt = df_cand[["valence", "arousal"]].to_numpy(np.float64)                  # Shape: [Nc,2]
    cand_pred = df_cand[["pred_valence", "pred_arousal"]].to_numpy(np.float64)      # Shape: [Nc,2]
    cand_cte = np.array([cte_map.get(int(s), 0.0) for s in cand_ids], dtype=np.float64)

    LOG.info("Queries: %d (test)", len(df_q))
    LOG.info("Candidates: %d (%s)", len(df_cand), args.candidates)
    LOG.info("beta_cte=%.4f gamma_ngram=%.4f n=%d", args.beta_cte, args.gamma_ngram, n)

    metrics_sum = {K: {"p": 0.0, "r": 0.0, "ndcg": 0.0} for K in k_list}
    n_queries = len(df_q)

    for _, row in df_q.iterrows():
        qid = int(row["song_id"])
        q_gt = np.array([row["valence"], row["arousal"]], dtype=np.float64)         # Shape: [2]
        q_pred = np.array([row["pred_valence"], row["pred_arousal"]], dtype=np.float64)  # Shape: [2]
        q_cte = float(cte_map.get(qid, 0.0))
        q_grams = ngrams_map.get(qid, set())


        mask = cand_ids != qid
        ids = cand_ids[mask]
        gt = cand_gt[mask]
        pred = cand_pred[mask]
        cte = cand_cte[mask]

        d_va = l2(q_pred, pred)                             # Shape: [N]
        d_cte = np.abs(cte - q_cte)                         # Shape: [N]

        # Grammar similarity per candidate 
        sim_g = np.zeros_like(d_va)
        for i in range(len(ids)):
            sim_g[i] = jaccard(q_grams, ngrams_map.get(int(ids[i]), set()))

        score = (-d_va) - args.beta_cte * d_cte + args.gamma_ngram * sim_g
        order = np.argsort(-score)  # descending
        ids = ids[order]
        gt = gt[order]

        d_gt = l2(q_gt, gt)
        thresh = np.quantile(d_gt, args.relevant_quantile)
        rel = (d_gt <= thresh)
        total_rel = int(np.sum(rel))

        for K in k_list:
            p = precision_at_k(rel, K)
            r = recall_at_k(rel, K, total_rel)

            rel_k = rel[:K].astype(np.float64)
            denom = np.log2(np.arange(2, len(rel_k) + 2))
            dcg = float(np.sum(rel_k / denom)) if len(rel_k) else 0.0
            idcg = ideal_dcg_at_k(total_rel, K)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            metrics_sum[K]["p"] += p
            metrics_sum[K]["r"] += r
            metrics_sum[K]["ndcg"] += ndcg

    metrics_avg = {f"P@{K}": metrics_sum[K]["p"] / n_queries for K in k_list}
    metrics_avg.update({f"R@{K}": metrics_sum[K]["r"] / n_queries for K in k_list})
    metrics_avg.update({f"NDCG@{K}": metrics_sum[K]["ndcg"] / n_queries for K in k_list})

    out = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT).as_posix()),
        "candidates": args.candidates,
        "k_list": k_list,
        "relevant_quantile": args.relevant_quantile,
        "beta_cte": args.beta_cte,
        "gamma_ngram": args.gamma_ngram,
        "ngram_n": n,
        "seed": args.seed,
        "n_queries": int(n_queries),
        "n_candidates": int(len(df_cand)),
        "metrics": metrics_avg,
    }

    out_path = run_dir / "retrieval_metrics_symbolic.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    LOG.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()