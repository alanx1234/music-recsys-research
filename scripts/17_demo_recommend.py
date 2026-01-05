"""
Demo: print top-K recommendations with a "why" breakdown.

Score(q,s) = -||VAhat_q - VAhat_s||_2  - beta*|CTE_q-CTE_s|  + gamma*Jaccard(ngrams_q, ngrams_s)

Reads:
  experiments/acoustic_mlp/run_###/model.pt
  data/processed/features_v0_energy.csv
  data/processed/chords_tokens/chords_tokens_cache.pkl
  data/processed/chords_tokens/ngrams_cache.pkl (optional)

Writes (optional):
  experiments/acoustic_mlp/run_###/demo_{song_id}_{tag}.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("demo_recommend")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_latest_run(exp_root: Path) -> Path:
    runs = sorted([p for p in exp_root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {exp_root}")
    return runs[-1]


def l2(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    d = X - q[None, :]
    return np.sqrt(np.sum(d * d, axis=1))


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a) + len(b) - inter
    return float(inter / uni) if uni > 0 else 0.0


def build_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--song_id", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    p.add_argument("--beta_cte", type=float, default=0.25)
    p.add_argument("--gamma_ngram", type=float, default=0.5)
    p.add_argument("--use_ngrams_cache", action="store_true")
    p.add_argument("--write_csv", action="store_true")
    p.add_argument("--tag", type=str, default="", help="Optional tag for output filename.")
    args = p.parse_args()

    exp_root = REPO_ROOT / "experiments" / "acoustic_mlp"
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(exp_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run_dir: {run_dir}")

    features_csv = REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"
    tok_cache_path = REPO_ROOT / "data" / "processed" / "chords_tokens" / "chords_tokens_cache.pkl"
    ngrams_cache_path = REPO_ROOT / "data" / "processed" / "chords_tokens" / "ngrams_cache.pkl"
    model_path = run_dir / "model.pt"

    for path in [features_csv, tok_cache_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    df_feat = pd.read_csv(features_csv)
    if not {"song_id", "split", "valence", "arousal"}.issubset(df_feat.columns):
        raise ValueError("features_v0_energy.csv must contain song_id, split, valence, arousal")

    import torch
    import torch.nn as nn

    class AcousticMLP(nn.Module):
        def __init__(self, d_in: int, d_hidden: int, dropout: float):
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
    state = ckpt["model_state_dict"]
    x_cols = ckpt.get("x_cols", ["log_rms_mean", "log_rms_std"])

    if any(c not in df_feat.columns for c in x_cols):
        raise ValueError(f"Missing feature columns in CSV. Need: {x_cols}")

    w0 = state.get("net.0.weight", None)
    if w0 is None:
        raise KeyError("Unexpected checkpoint format (missing net.0.weight).")
    d_hidden = int(w0.shape[0])
    d_in = int(w0.shape[1])

    scaler_mean = ckpt["scaler_mean"].float()
    scaler_std = ckpt["scaler_std"].float().clamp_min(1e-6)

    model = AcousticMLP(d_in=d_in, d_hidden=d_hidden, dropout=0.0)
    model.load_state_dict(state)
    model.eval()

    X_all = torch.from_numpy(df_feat[x_cols].to_numpy(np.float32))
    X_all = (X_all - scaler_mean[None, :]) / scaler_std[None, :]
    with torch.no_grad():
        Yhat_all = model(X_all).numpy()

    df_all = df_feat[["song_id", "split", "valence", "arousal"]].copy()
    df_all["song_id"] = df_all["song_id"].astype(int)
    df_all["pred_valence"] = Yhat_all[:, 0]
    df_all["pred_arousal"] = Yhat_all[:, 1]

    with tok_cache_path.open("rb") as f:
        tok_cache: Dict[int, Dict[str, object]] = pickle.load(f)

    cte_map: Dict[int, float] = {}
    tokens_map: Dict[int, List[str]] = {}
    for sid, d in tok_cache.items():
        sid_i = int(sid)
        cte_map[sid_i] = float(d.get("cte", 0.0))
        toks = d.get("tokens", [])
        if not isinstance(toks, list):
            toks = list(toks)
        tokens_map[sid_i] = toks

    if args.use_ngrams_cache:
        if not ngrams_cache_path.exists():
            raise FileNotFoundError(f"Missing: {ngrams_cache_path} (run 06_build_ngrams_cache.py)")
        with ngrams_cache_path.open("rb") as f:
            payload = pickle.load(f)
        n = int(payload["n"])
        ngrams_map: Dict[int, Set[Tuple[str, ...]]] = payload["ngrams"]
    else:
        n = 4
        ngrams_map = {sid: build_ngrams(toks, n) for sid, toks in tokens_map.items()}

    qid = int(args.song_id)
    if qid not in set(df_all["song_id"].tolist()):
        raise ValueError(f"song_id={qid} not found in {features_csv}")

    q_row = df_all[df_all["song_id"] == qid].iloc[0]
    q_pred = np.array([q_row["pred_valence"], q_row["pred_arousal"]], dtype=np.float64)
    q_cte = float(cte_map.get(qid, 0.0))
    q_grams = ngrams_map.get(qid, set())

    if args.candidates == "trainval":
        df_cand = df_all[df_all["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    else:
        df_cand = df_all.copy().reset_index(drop=True)

    cand_ids = df_cand["song_id"].to_numpy(np.int32)
    cand_pred = df_cand[["pred_valence", "pred_arousal"]].to_numpy(np.float64)
    cand_cte = np.array([cte_map.get(int(s), 0.0) for s in cand_ids], dtype=np.float64)

    mask = cand_ids != qid
    cand_ids = cand_ids[mask]
    cand_pred = cand_pred[mask]
    cand_cte = cand_cte[mask]

    d_va = l2(q_pred, cand_pred)
    d_cte = np.abs(cand_cte - q_cte)

    sim_g = np.zeros_like(d_va)
    for i in range(len(cand_ids)):
        sim_g[i] = jaccard(q_grams, ngrams_map.get(int(cand_ids[i]), set()))

    term_va = -d_va
    term_cte = -args.beta_cte * d_cte
    term_ng = args.gamma_ngram * sim_g
    score = term_va + term_cte + term_ng

    order = np.argsort(-score)
    top = order[: max(args.k, 1)]

    out = pd.DataFrame(
        {
            "rank": np.arange(1, len(top) + 1),
            "song_id": cand_ids[top].astype(int),
            "score": score[top],
            "term_va": term_va[top],
            "term_cte": term_cte[top],
            "term_ngram": term_ng[top],
            "d_va": d_va[top],
            "d_cte": d_cte[top],
            "sim_ngram": sim_g[top],
        }
    )

    LOG.info(
        "Query song_id=%d | split=%s | pred(V,A)=(%.3f, %.3f) | CTE=%.3f | n=%d",
        qid,
        str(q_row["split"]),
        float(q_pred[0]),
        float(q_pred[1]),
        q_cte,
        n,
    )
    LOG.info(
        "Scoring: beta_cte=%.4f gamma_ngram=%.4f | candidates=%s",
        args.beta_cte,
        args.gamma_ngram,
        args.candidates,
    )

    print(out.to_string(index=False))

    if args.write_csv:
        tag = f"_{args.tag}" if args.tag else ""
        out_path = run_dir / f"demo_{qid}{tag}.csv"
        out.to_csv(out_path, index=False)
        LOG.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
