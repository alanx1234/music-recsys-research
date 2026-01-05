"""
For each query song (test split), rank candidate songs by:
  score = -||yhat_q - yhat_s||2  - beta * |cte_q - cte_s| + gamma * jaccard(ngrams_q, ngrams_s)

Then report, for each variant and each K:
  - mean GT (valence, arousal) distance over top-K
  - mean |CTE diff| over top-K
  - mean n-gram similarity over top-K (optional but useful)

Inputs:
  experiments/.../preds.csv
  data/processed/features_v0_energy.csv
  data/processed/chords_tokens/cte.csv
  data/processed/chords_tokens/ngrams_cache.pkl

Outputs:
  experiments/.../table2_retrieval_distance.csv
  experiments/.../table2_retrieval_distance.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"
CTE_CSV = REPO_ROOT / "data" / "processed" / "chords_tokens" / "cte.csv"
NGRAMS_CACHE = REPO_ROOT / "data" / "processed" / "chords_tokens" / "ngrams_cache.pkl"

LOG = logging.getLogger("make_table2_retrieval_distance")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _pick_cols(df: pd.DataFrame, candidates: List[Tuple[str, ...]]) -> Tuple[str, str]:
    cols = set(df.columns)
    for a, b in candidates:
        if a in cols and b in cols:
            return a, b
    raise ValueError(f"Could not find required column pair among: {candidates}. Have: {sorted(df.columns)}")


def jaccard(a: Iterable[Tuple[str, ...]] | set, b: Iterable[Tuple[str, ...]] | set) -> float:
    a = set(a)
    b = set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter) / float(uni) if uni > 0 else 0.0


def load_ngrams(path: Path) -> Tuple[int, Dict[int, set]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "ngrams" not in obj:
        raise ValueError(f"Unexpected ngrams_cache format: {path}")
    n = int(obj.get("n", 0))
    grams = obj["ngrams"]
    grams = {int(k): v for k, v in grams.items()}
    return n, grams


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g., experiments/acoustic_mlp/run_001")
    ap.add_argument("--preds_csv", type=str, default="", help="Override preds.csv path (else uses run_dir/preds.csv)")
    ap.add_argument("--candidates", type=str, default="trainval", choices=["trainval", "all"])
    ap.add_argument("--k_list", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--beta", type=float, default=0.1, help="CTE weight")
    ap.add_argument("--gamma", type=float, default=0.75, help="n-gram weight")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    preds_path = Path(args.preds_csv) if args.preds_csv else (run_dir / "preds.csv")
    out_csv = run_dir / "table2_retrieval_distance.csv"
    out_json = run_dir / "table2_retrieval_distance.json"

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing: {preds_path}")
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_CSV}")
    if not CTE_CSV.exists():
        raise FileNotFoundError(f"Missing: {CTE_CSV}")
    if not NGRAMS_CACHE.exists():
        raise FileNotFoundError(f"Missing: {NGRAMS_CACHE}")

    preds = pd.read_csv(preds_path)
    feats = pd.read_csv(FEATURES_CSV)
    cte = pd.read_csv(CTE_CSV)

    y_col_v, y_col_a = _pick_cols(
        feats,
        [
            ("valence", "arousal"),
            ("y_valence", "y_arousal"),
            ("gt_valence", "gt_arousal"),
            ("V", "A"),
            ("v", "a"),
        ],
    )
    yhat_col_v, yhat_col_a = _pick_cols(
        preds,
        [
            ("pred_valence", "pred_arousal"),
            ("yhat_valence", "yhat_arousal"),
            ("valence_hat", "arousal_hat"),
            ("pred_v", "pred_a"),
        ],
    )

    if "song_id" not in feats.columns or "song_id" not in preds.columns:
        raise ValueError("Both features_v0_energy.csv and preds.csv must contain song_id")

    split_col = "split" if "split" in feats.columns else ("split" if "split" in preds.columns else None)
    if split_col is None:
        raise ValueError("Need a split column in features_v0_energy.csv (preferred) or preds.csv")

    feats = feats[["song_id", split_col, y_col_v, y_col_a]].copy()
    feats.rename(columns={split_col: "split", y_col_v: "gt_v", y_col_a: "gt_a"}, inplace=True)

    preds = preds[["song_id", yhat_col_v, yhat_col_a]].copy()
    preds.rename(columns={yhat_col_v: "pred_v", yhat_col_a: "pred_a"}, inplace=True)

    cte = cte[["song_id", "cte"]].copy()

    df = feats.merge(preds, on="song_id", how="inner").merge(cte, on="song_id", how="left")
    df["song_id"] = df["song_id"].astype(int)

    n, grams = load_ngrams(NGRAMS_CACHE)
    LOG.info("Loaded n-grams cache: n=%d", n)

    queries = df[df["split"] == "test"].copy()
    if args.candidates == "trainval":
        cand_df = df[df["split"].isin(["train", "val"])].copy()
    else:
        cand_df = df.copy()

    q_ids = queries["song_id"].to_numpy()
    c_ids = cand_df["song_id"].to_numpy()

    LOG.info("Queries: %d (test)", len(q_ids))
    LOG.info("Candidates: %d (%s)", len(c_ids), args.candidates)

    cand_pred = cand_df[["pred_v", "pred_a"]].to_numpy(dtype=np.float32)
    cand_gt = cand_df[["gt_v", "gt_a"]].to_numpy(dtype=np.float32)
    cand_cte = cand_df["cte"].to_numpy(dtype=np.float32)

    variants = [
        ("va_only", 0.0, 0.0),
        ("va_plus_cte", float(args.beta), 0.0),
        ("va_plus_ngram", 0.0, float(args.gamma)),
        ("full", float(args.beta), float(args.gamma)),
    ]

    rows_out: List[dict] = []

    for name, beta, gamma in variants:
        for K in args.k_list:
            gt_dists: List[float] = []
            cte_diffs: List[float] = []
            sims: List[float] = []

            for _, q in queries.iterrows():
                qid = int(q["song_id"])
                q_pred = np.array([q["pred_v"], q["pred_a"]], dtype=np.float32)
                q_gt = np.array([q["gt_v"], q["gt_a"]], dtype=np.float32)
                q_cte = float(q["cte"]) if pd.notna(q["cte"]) else np.nan
                q_grams = grams.get(qid, set())

                d_va = np.linalg.norm(cand_pred - q_pred[None, :], axis=1)  # [Nc]
                d_cte = np.abs(cand_cte - q_cte) if np.isfinite(q_cte) else np.zeros_like(d_va)

                if gamma != 0.0:
                    sim_ng = np.array([jaccard(q_grams, grams.get(int(sid), set())) for sid in c_ids], dtype=np.float32)
                else:
                    sim_ng = np.zeros_like(d_va, dtype=np.float32)

                score = (-d_va) - (beta * d_cte) + (gamma * sim_ng)

                if args.candidates == "all":
                    score = score.copy()
                    score[c_ids == qid] = -np.inf

                top_idx = np.argsort(-score)[:K]

                top_gt = cand_gt[top_idx]
                top_cte = cand_cte[top_idx]
                top_sim = sim_ng[top_idx]

                gt_dist_k = np.linalg.norm(top_gt - q_gt[None, :], axis=1)
                gt_dists.append(float(np.mean(gt_dist_k)))

                if np.isfinite(q_cte):
                    cte_diffs.append(float(np.mean(np.abs(top_cte - q_cte))))
                else:
                    cte_diffs.append(float("nan"))

                sims.append(float(np.mean(top_sim)))

            rows_out.append(
                {
                    "variant": name,
                    "beta_cte": beta,
                    "gamma_ngram": gamma,
                    "K": int(K),
                    "mean_gt_va_dist": float(np.nanmean(gt_dists)),
                    "mean_abs_cte_diff": float(np.nanmean(cte_diffs)),
                    "mean_ngram_sim": float(np.nanmean(sims)),
                    "n_queries": int(len(q_ids)),
                    "n_candidates": int(len(c_ids)),
                }
            )

    out_df = pd.DataFrame(rows_out)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    payload = {
        "run_dir": str(run_dir).replace("\\", "/"),
        "candidates": args.candidates,
        "k_list": list(map(int, args.k_list)),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "ngram_n": int(n),
        "n_queries": int(len(q_ids)),
        "n_candidates": int(len(c_ids)),
        "rows": rows_out,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOG.info("Wrote: %s", out_csv)
    LOG.info("Wrote: %s", out_json)


if __name__ == "__main__":
    main()
