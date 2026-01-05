"""
Grid-sweep beta_cte and gamma_ngram for symbolic retrieval and write a CSV.

Reads:
  experiments/acoustic_mlp/run_###/model.pt
  data/processed/features_v0_energy.csv
  data/processed/chords_tokens/chords_tokens_cache.pkl
  data/processed/chords_tokens/ngrams_cache.pkl

Writes:
  experiments/acoustic_mlp/run_###/retrieval_sweep_symbolic.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import logging
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

LOG = logging.getLogger("sweep_symbolic_weights")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def find_latest_run() -> Path:
    runs = sorted([p for p in EXP_ROOT.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {EXP_ROOT}")
    return runs[-1]


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def eval_once(
    df_all: pd.DataFrame,
    cte_map: Dict[int, float],
    ngrams_map: Dict[int, Set[Tuple[str, ...]]],
    candidates: str,
    relevant_quantile: float,
    k_list: List[int],
    beta_cte: float,
    gamma_ngram: float,
) -> Dict[str, float]:
    df_q = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    if candidates == "trainval":
        df_cand = df_all[df_all["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    else:
        df_cand = df_all.copy().reset_index(drop=True)

    cand_ids = df_cand["song_id"].to_numpy(np.int32)
    cand_gt = df_cand[["valence", "arousal"]].to_numpy(np.float64)
    cand_pred = df_cand[["pred_valence", "pred_arousal"]].to_numpy(np.float64)
    cand_cte = np.array([cte_map.get(int(s), 0.0) for s in cand_ids], dtype=np.float64)

    metrics_sum = {K: {"p": 0.0, "r": 0.0, "ndcg": 0.0} for K in k_list}
    n_queries = len(df_q)

    for _, row in df_q.iterrows():
        qid = int(row["song_id"])
        q_gt = np.array([row["valence"], row["arousal"]], dtype=np.float64)
        q_pred = np.array([row["pred_valence"], row["pred_arousal"]], dtype=np.float64)
        q_cte = float(cte_map.get(qid, 0.0))
        q_grams = ngrams_map.get(qid, set())

        mask = cand_ids != qid
        ids = cand_ids[mask]
        gt = cand_gt[mask]
        pred = cand_pred[mask]
        cte = cand_cte[mask]

        d_va = l2(q_pred, pred)
        d_cte = np.abs(cte - q_cte)

        sim_g = np.zeros_like(d_va)
        for i in range(len(ids)):
            sim_g[i] = jaccard(q_grams, ngrams_map.get(int(ids[i]), set()))

        score = (-d_va) - beta_cte * d_cte + gamma_ngram * sim_g
        order = np.argsort(-score)
        gt = gt[order]

        d_gt = l2(q_gt, gt)
        thresh = np.quantile(d_gt, relevant_quantile)
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

    out = {}
    for K in k_list:
        out[f"P@{K}"] = metrics_sum[K]["p"] / n_queries
        out[f"R@{K}"] = metrics_sum[K]["r"] / n_queries
        out[f"NDCG@{K}"] = metrics_sum[K]["ndcg"] / n_queries
    return out


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    parser.add_argument("--relevant_quantile", type=float, default=0.05)
    parser.add_argument("--k_list", type=str, default="5,10,20")
    parser.add_argument("--beta_list", type=str, default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--gamma_list", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    seed_everything(args.seed)
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    beta_list = [float(x) for x in args.beta_list.split(",") if x.strip()]
    gamma_list = [float(x) for x in args.gamma_list.split(",") if x.strip()]

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

    df_feat = pd.read_csv(FEATURES_CSV)

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
    scaler_mean = ckpt["scaler_mean"].float()
    scaler_std = ckpt["scaler_std"].float().clamp_min(1e-6)
    x_cols = ckpt.get("x_cols", ["log_rms_mean", "log_rms_std"])

    model = AcousticMLP(d_in=2, d_hidden=64, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_all = torch.from_numpy(df_feat[x_cols].to_numpy(np.float32))
    X_all = (X_all - scaler_mean[None, :]) / scaler_std[None, :]
    with torch.no_grad():
        Yhat_all = model(X_all).numpy()

    df_all = df_feat[["song_id", "split", "valence", "arousal"]].copy()
    df_all["pred_valence"] = Yhat_all[:, 0]
    df_all["pred_arousal"] = Yhat_all[:, 1]

    with TOK_CACHE.open("rb") as f:
        tok_cache: Dict[int, Dict[str, object]] = pickle.load(f)

    cte_map: Dict[int, float] = {int(sid): float(d.get("cte", 0.0)) for sid, d in tok_cache.items()}

    with NGRAMS_CACHE.open("rb") as f:
        payload = pickle.load(f)
    ngrams_map = payload["ngrams"]
    n = int(payload["n"])

    out_csv = run_dir / "retrieval_sweep_symbolic.csv"
    LOG.info("Sweeping %d betas x %d gammas (n=%d) -> %s", len(beta_list), len(gamma_list), n, out_csv)

    rows = []
    for beta in beta_list:
        for gamma in gamma_list:
            m = eval_once(
                df_all=df_all,
                cte_map=cte_map,
                ngrams_map=ngrams_map,
                candidates=args.candidates,
                relevant_quantile=args.relevant_quantile,
                k_list=k_list,
                beta_cte=beta,
                gamma_ngram=gamma,
            )
            row = {"beta_cte": beta, "gamma_ngram": gamma}
            row.update(m)
            rows.append(row)
            LOG.info("beta=%.3f gamma=%.3f -> %s", beta, gamma, json.dumps(m))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    LOG.info("Wrote: %s", out_csv)


if __name__ == "__main__":
    main()
