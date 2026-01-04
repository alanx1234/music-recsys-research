"""
Evaluate emotion-conditioned retrieval using predicted (V,A) distances.

Inputs:
  experiments/acoustic_mlp/run_###/preds.csv   (must include test predictions)
  data/processed/features_v0_energy.csv        (for GT V/A for all songs + splits)

Outputs:
  experiments/acoustic_mlp/run_###/retrieval_metrics.json

Method:
- Queries: test split songs
- Candidates: all songs (train+val+test) excluding the query itself
- Score: -|| yhat_q - yhat_s ||_2
- Relevance: GT nearest-neighbors by quantile (e.g., top 5% closest by GT distance)
- Metrics: Precision@K, Recall@K, NDCG@K (averaged over queries)
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


K_LIST = [5, 10, 20]
RELEVANT_QUANTILE = 0.05  # relevant = closest 5% by GT distance
CANDIDATES = "trainval"        # "all" or "trainval" (exclude test from candidates)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "acoustic_mlp"
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [2], b: [N,2] -> [N]
    d = b - a[None, :]
    return np.sqrt(np.sum(d * d, axis=1))


def precision_at_k(relevant: np.ndarray, k: int) -> float:
    topk = relevant[:k]
    return float(np.mean(topk)) if k > 0 else 0.0


def recall_at_k(relevant: np.ndarray, k: int, n_rel: int) -> float:
    if n_rel == 0:
        return 0.0
    return float(np.sum(relevant[:k]) / n_rel)


def ndcg_at_k(relevant: np.ndarray, k: int) -> float:
    # Binary relevance NDCG@k
    rel = relevant[:k].astype(np.float64)
    if rel.size == 0:
        return 0.0
    # DCG
    denom = np.log2(np.arange(2, rel.size + 2))
    dcg = np.sum(rel / denom)
    # IDCG (best possible: all relevant first)
    n_rel = int(np.sum(rel))  # note: only within top-k; but IDCG should use min(total_rel, k)

    return float(dcg)


def ideal_dcg_at_k(total_relevant: int, k: int) -> float:
    m = min(total_relevant, k)
    if m <= 0:
        return 0.0
    denom = np.log2(np.arange(2, m + 2))
    return float(np.sum(np.ones(m) / denom))


def find_latest_run() -> Path:
    runs = sorted([p for p in EXP_ROOT.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {EXP_ROOT}")
    return runs[-1]


def main() -> None:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_CSV}")

    run_dir = find_latest_run()
    preds_path = run_dir / "preds.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing preds.csv: {preds_path}")

    df_feat = pd.read_csv(FEATURES_CSV)
    df_pred = pd.read_csv(preds_path)
    required_feat = {"song_id", "split", "valence", "arousal"}
    if not required_feat.issubset(df_feat.columns):
        raise ValueError(f"features CSV missing columns: {sorted(required_feat - set(df_feat.columns))}")

    required_pred = {"song_id", "pred_valence", "pred_arousal"}
    if not required_pred.issubset(df_pred.columns):
        raise ValueError(f"preds CSV missing columns: {sorted(required_pred - set(df_pred.columns))}")

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt for full inference: {model_path}")

    import torch
    import torch.nn as nn

    class AcousticMLP(nn.Module):
        def __init__(self, d_in: int = 2, d_hidden: int = 64, dropout: float = 0.1):
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
    scaler_mean = ckpt["scaler_mean"].float()  # [2]
    scaler_std = ckpt["scaler_std"].float().clamp_min(1e-6)  # [2]
    x_cols = ckpt.get("x_cols", ["log_rms_mean", "log_rms_std"])

    model = AcousticMLP(d_in=2, d_hidden=64, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Inference for all songs
    X_all = torch.from_numpy(df_feat[x_cols].to_numpy(np.float32))  # [N,2]
    X_all = (X_all - scaler_mean[None, :]) / scaler_std[None, :]    # [N,2]
    with torch.no_grad():
        Yhat_all = model(X_all).numpy()  # [N,2]

    df_all = df_feat[["song_id", "split", "valence", "arousal"]].copy()
    df_all["pred_valence"] = Yhat_all[:, 0]
    df_all["pred_arousal"] = Yhat_all[:, 1]

    # Define query set: test
    df_q = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)

    # Define candidate pool
    if CANDIDATES == "trainval":
        df_cand = df_all[df_all["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    elif CANDIDATES == "all":
        df_cand = df_all.copy().reset_index(drop=True)
    else:
        raise ValueError("CANDIDATES must be 'all' or 'trainval'")

    cand_ids = df_cand["song_id"].to_numpy(np.int32)
    cand_gt = df_cand[["valence", "arousal"]].to_numpy(np.float64)          # [Nc,2]
    cand_pred = df_cand[["pred_valence", "pred_arousal"]].to_numpy(np.float64)  # [Nc,2]

    metrics_sum = {k: {"p": 0.0, "r": 0.0, "ndcg": 0.0} for k in K_LIST}
    n_queries = len(df_q)

    for _, row in df_q.iterrows():
        qid = int(row["song_id"])
        q_gt = np.array([row["valence"], row["arousal"]], dtype=np.float64)        # [2]
        q_pred = np.array([row["pred_valence"], row["pred_arousal"]], dtype=np.float64)  # [2]

        # Candidate mask to exclude self if candidate pool includes it
        mask = cand_ids != qid
        ids = cand_ids[mask]
        gt = cand_gt[mask]
        pred = cand_pred[mask]

        # Rank by predicted VA distance (ascending)
        d_pred = l2(q_pred, pred)                 # [N]
        order = np.argsort(d_pred)
        ids = ids[order]
        gt = gt[order]

        # Determine relevance by GT distance quantile
        d_gt = l2(q_gt, gt)                       # [N] in ranked order
        thresh = np.quantile(d_gt, RELEVANT_QUANTILE)
        rel = (d_gt <= thresh)                    # [N] boolean relevance in ranked list
        total_rel = int(np.sum(rel))

        for K in K_LIST:
            p = precision_at_k(rel, K)
            r = recall_at_k(rel, K, total_rel)

            # NDCG@K
            rel_k = rel[:K].astype(np.float64)
            denom = np.log2(np.arange(2, len(rel_k) + 2))
            dcg = float(np.sum(rel_k / denom)) if len(rel_k) else 0.0
            idcg = ideal_dcg_at_k(total_rel, K)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            metrics_sum[K]["p"] += p
            metrics_sum[K]["r"] += r
            metrics_sum[K]["ndcg"] += ndcg

    metrics_avg = {}
    for K in K_LIST:
        metrics_avg[f"P@{K}"] = metrics_sum[K]["p"] / n_queries
        metrics_avg[f"R@{K}"] = metrics_sum[K]["r"] / n_queries
        metrics_avg[f"NDCG@{K}"] = metrics_sum[K]["ndcg"] / n_queries

    out = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT).as_posix()),
        "candidates": CANDIDATES,
        "k_list": K_LIST,
        "relevant_quantile": RELEVANT_QUANTILE,
        "n_queries": n_queries,
        "n_candidates": int(len(df_cand)),
        "metrics": metrics_avg,
    }

    out_path = run_dir / "retrieval_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {out_path}")
    print("Retrieval metrics (averaged over queries):")
    for k, v in metrics_avg.items():
        print(f"  {k:8s} = {v:.4f}")


if __name__ == "__main__":
    main()