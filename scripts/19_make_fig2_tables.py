from __future__ import annotations

from pathlib import Path
import argparse
import logging
import math
import pickle
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"
TOK_CACHE = REPO_ROOT / "data" / "processed" / "chords_tokens" / "chords_tokens_cache.pkl"
NGRAMS_CACHE = REPO_ROOT / "data" / "processed" / "chords_tokens" / "ngrams_cache.pkl"

LOG = logging.getLogger("make_fig2_qual_tables")


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


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a[None, :]
    return np.sqrt(np.sum(d * d, axis=1))


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a) + len(b) - inter
    return float(inter / uni) if uni > 0 else 0.0


def load_preds_from_model(run_dir: Path, df_feat: pd.DataFrame) -> pd.DataFrame:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing: {model_path}")

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

    X = torch.from_numpy(df_feat[x_cols].to_numpy(np.float32))
    X = (X - scaler_mean[None, :]) / scaler_std[None, :]
    with torch.no_grad():
        yhat = model(X).numpy()

    out = df_feat[["song_id", "split"]].copy()
    out["pred_valence"] = yhat[:, 0]
    out["pred_arousal"] = yhat[:, 1]
    return out


def load_symbolic() -> Tuple[Dict[int, float], Dict[int, List[str]]]:
    if not TOK_CACHE.exists():
        raise FileNotFoundError(f"Missing: {TOK_CACHE}")
    with TOK_CACHE.open("rb") as f:
        tok_cache: Dict[int, Dict[str, object]] = pickle.load(f)

    cte_map: Dict[int, float] = {}
    tokens_map: Dict[int, List[str]] = {}
    for sid, d in tok_cache.items():
        sid_i = int(sid)
        cte_map[sid_i] = float(d.get("cte", 0.0))
        toks = d.get("tokens", [])
        tokens_map[sid_i] = list(toks) if isinstance(toks, list) else list(toks)

    return cte_map, tokens_map


def load_ngrams(tokens_map: Dict[int, List[str]], n: int, use_cache: bool) -> Dict[int, Set[Tuple[str, ...]]]:
    if use_cache:
        if not NGRAMS_CACHE.exists():
            raise FileNotFoundError(f"Missing n-grams cache (run script 06): {NGRAMS_CACHE}")
        with NGRAMS_CACHE.open("rb") as f:
            payload = pickle.load(f)
        n_cache = int(payload["n"])
        if n_cache != n:
            raise ValueError(f"ngrams_cache.pkl has n={n_cache}, but requested n={n}")
        return payload["ngrams"]

    ngrams_map: Dict[int, Set[Tuple[str, ...]]] = {}
    for sid, toks in tokens_map.items():
        if len(toks) < n:
            ngrams_map[sid] = set()
        else:
            ngrams_map[sid] = {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}
    return ngrams_map


def score_topk(
    q_sid: int,
    q_pred: np.ndarray,
    q_cte: float,
    q_ngrams: Set[Tuple[str, ...]],
    cand_ids: np.ndarray,
    cand_pred: np.ndarray,
    cand_cte: np.ndarray,
    cand_ngrams: Dict[int, Set[Tuple[str, ...]]],
    beta_cte: float,
    gamma_ngram: float,
    k: int,
) -> pd.DataFrame:
    d_va = l2(q_pred, cand_pred)  # [Nc]
    d_cte = np.abs(cand_cte - q_cte)  # [Nc]

    if gamma_ngram != 0.0:
        sim = np.zeros(len(cand_ids), dtype=np.float64)
        for i, sid in enumerate(cand_ids):
            sim[i] = jaccard(q_ngrams, cand_ngrams.get(int(sid), set()))
    else:
        sim = np.zeros(len(cand_ids), dtype=np.float64)

    term_va = -d_va
    term_cte = -beta_cte * d_cte
    term_ng = gamma_ngram * sim
    score = term_va + term_cte + term_ng

    kk = min(k, len(cand_ids))
    idx = np.argpartition(-score, kk - 1)[:kk]
    idx = idx[np.argsort(-score[idx])]

    df = pd.DataFrame(
        {
            "rank": np.arange(1, kk + 1, dtype=np.int64),
            "song_id": cand_ids[idx].astype(np.int64),
            "score": score[idx],
            "d_va": d_va[idx],
            "sim_ngram": sim[idx],
            "d_cte": d_cte[idx],
        }
    )
    return df


def pick_ngram_query(
    q_ids: np.ndarray,
    cand_ids: np.ndarray,
    ngrams_map: Dict[int, Set[Tuple[str, ...]]],
    min_nonzero: int,
) -> int:
    best_sid = int(q_ids[0])
    best_max = -1.0
    best_nz = -1

    for q in q_ids:
        q_set = ngrams_map.get(int(q), set())
        sims = []
        nz = 0
        m = 0.0
        for sid in cand_ids:
            s = jaccard(q_set, ngrams_map.get(int(sid), set()))
            if s > 0:
                nz += 1
            if s > m:
                m = s
        if nz >= min_nonzero:
            if (m > best_max) or (math.isclose(m, best_max) and nz > best_nz):
                best_max = m
                best_nz = nz
                best_sid = int(q)

    LOG.info("Picked n-gram query song_id=%d (max_sim=%.3f nonzero=%d)", best_sid, best_max, best_nz)
    return best_sid


def pick_cte_query(
    q_ids: np.ndarray,
    q_pred_map: Dict[int, np.ndarray],
    q_cte_map: Dict[int, float],
    q_ng_map: Dict[int, Set[Tuple[str, ...]]],
    cand_ids: np.ndarray,
    cand_pred: np.ndarray,
    cand_cte: np.ndarray,
    cand_ng_map: Dict[int, Set[Tuple[str, ...]]],
    beta_cte: float,
    k: int,
) -> int:
    best_sid = int(q_ids[0])
    best_delta = -1.0

    for q in q_ids:
        q = int(q)
        df_va = score_topk(
            q, q_pred_map[q], q_cte_map[q], q_ng_map[q],
            cand_ids, cand_pred, cand_cte, cand_ng_map,
            beta_cte=0.0, gamma_ngram=0.0, k=k
        )
        df_cte = score_topk(
            q, q_pred_map[q], q_cte_map[q], q_ng_map[q],
            cand_ids, cand_pred, cand_cte, cand_ng_map,
            beta_cte=beta_cte, gamma_ngram=0.0, k=k
        )

        mean_va = float(df_va["d_cte"].mean()) if len(df_va) else 0.0
        mean_cte = float(df_cte["d_cte"].mean()) if len(df_cte) else 0.0
        delta = mean_va - mean_cte
        changed = len(set(df_va["song_id"].tolist()).difference(set(df_cte["song_id"].tolist())))

        if changed < max(2, k // 2):
            continue
        if delta > best_delta:
            best_delta = delta
            best_sid = q

    LOG.info("Picked CTE query song_id=%d (mean_dcte_drop=%.3f)", best_sid, best_delta)
    return best_sid


def draw_table(ax, df: pd.DataFrame, title: str) -> None:
    import matplotlib.pyplot as plt

    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    df2 = df.copy()
    df2["rank"] = df2["rank"].astype(int)
    df2["song_id"] = df2["song_id"].astype(int)

    fmt = {
        "score": "{:.3f}",
        "d_va": "{:.3f}",
        "sim_ngram": "{:.3f}",
        "d_cte": "{:.3f}",
    }
    for c, f in fmt.items():
        if c in df2.columns:
            df2[c] = df2[c].map(lambda x: f.format(float(x)))

    cols = ["rank", "song_id", "score", "d_va", "sim_ngram", "d_cte"]
    cell_text = df2[cols].values.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="", help="e.g., experiments/acoustic_mlp/run_001")
    p.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--beta_cte", type=float, default=0.10)
    p.add_argument("--gamma_ngram", type=float, default=0.75)
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--use_ngrams_cache", action="store_true")
    p.add_argument("--min_nonzero", type=int, default=5)
    p.add_argument("--out", type=str, default="", help="Optional output path (png).")
    args = p.parse_args()

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_CSV}")

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(REPO_ROOT / "experiments" / "acoustic_mlp")
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run_dir: {run_dir}")

    df_feat = pd.read_csv(FEATURES_CSV)
    need = {"song_id", "split", "valence", "arousal"}
    if not need.issubset(df_feat.columns):
        raise ValueError(f"features_v0_energy.csv missing: {sorted(need - set(df_feat.columns))}")

    df_pred = load_preds_from_model(run_dir, df_feat)

    cte_map, tokens_map = load_symbolic()
    ngrams_map = load_ngrams(tokens_map, n=args.ngram_n, use_cache=args.use_ngrams_cache)

    df_all = df_pred.copy()
    df_all["cte"] = df_all["song_id"].map(lambda s: float(cte_map.get(int(s), 0.0)))

    df_q = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
    if args.candidates == "trainval":
        df_c = df_all[df_all["split"].isin(["train", "val"])].copy().reset_index(drop=True)
    else:
        df_c = df_all.copy().reset_index(drop=True)

    q_ids = df_q["song_id"].to_numpy(np.int64)
    cand_ids = df_c["song_id"].to_numpy(np.int64)
    cand_pred = df_c[["pred_valence", "pred_arousal"]].to_numpy(np.float64)
    cand_cte = df_c["cte"].to_numpy(np.float64)

    q_pred_map = {int(r.song_id): np.array([r.pred_valence, r.pred_arousal], dtype=np.float64) for r in df_q.itertuples()}
    q_cte_map = {int(r.song_id): float(r.cte) for r in df_q.itertuples()}
    q_ng_map = {int(sid): ngrams_map.get(int(sid), set()) for sid in q_ids}

    cand_ng_map = {int(sid): ngrams_map.get(int(sid), set()) for sid in cand_ids}

    q_ngram = pick_ngram_query(q_ids, cand_ids, ngrams_map, min_nonzero=args.min_nonzero)
    q_cte = pick_cte_query(
        q_ids, q_pred_map, q_cte_map, q_ng_map,
        cand_ids, cand_pred, cand_cte, cand_ng_map,
        beta_cte=args.beta_cte, k=args.k
    )

    def mk(qsid: int, beta: float, gamma: float) -> pd.DataFrame:
        qsid = int(qsid)
        return score_topk(
            qsid,
            q_pred_map[qsid],
            q_cte_map[qsid],
            q_ng_map[qsid],
            cand_ids,
            cand_pred,
            cand_cte,
            cand_ng_map,
            beta_cte=beta,
            gamma_ngram=gamma,
            k=args.k,
        )

    df_ngr_va = mk(q_ngram, beta=0.0, gamma=0.0)
    df_ngr_ng = mk(q_ngram, beta=0.0, gamma=args.gamma_ngram)
    df_ngr_full = mk(q_ngram, beta=args.beta_cte, gamma=args.gamma_ngram)

    df_cte_va = mk(q_cte, beta=0.0, gamma=0.0)
    df_cte_cte = mk(q_cte, beta=args.beta_cte, gamma=0.0)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, wspace=0.15, hspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    draw_table(ax1, df_ngr_va, f"Query {q_ngram} | VA only")
    draw_table(ax2, df_ngr_ng, f"Query {q_ngram} | + n-gram (γ={args.gamma_ngram:.2f})")
    draw_table(ax3, df_ngr_full, f"Query {q_ngram} | full (β={args.beta_cte:.2f}, γ={args.gamma_ngram:.2f})")
    draw_table(ax4, df_cte_va, f"Query {q_cte} | VA only")
    draw_table(ax5, df_cte_cte, f"Query {q_cte} | + CTE (β={args.beta_cte:.2f}, γ=0)")

    out_path = Path(args.out) if args.out else (run_dir / "fig2_qual_tables.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    LOG.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
