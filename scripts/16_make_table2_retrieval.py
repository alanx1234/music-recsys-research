"""
Make Table 2 (retrieval metrics) from the symbolic sweep CSV.

Reads:
  experiments/acoustic_mlp/run_###/retrieval_sweep_symbolic.csv

Writes:
  experiments/acoustic_mlp/run_###/retrieval_table2.csv

Rows:
  - va_only              (beta=0, gamma=0)
  - va_plus_cte_best     (gamma=0, best beta by select_metric)
  - va_plus_ngram_best   (beta=0, best gamma by select_metric)
  - full_best            (best (beta,gamma) by select_metric)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("make_table2_retrieval")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def find_latest_run(exp_root: Path) -> Path:
    runs = sorted([p for p in exp_root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {exp_root}")
    return runs[-1]


def pick_best(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        raise ValueError(f"Missing metric column '{metric}' in sweep CSV. Have: {list(df.columns)}")
    return df.sort_values(metric, ascending=False).iloc[0]


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="", help="Optional: experiments/acoustic_mlp/run_### (default: latest)")
    p.add_argument("--select_metric", type=str, default="NDCG@10", help="Which metric to maximize (e.g., NDCG@10)")
    args = p.parse_args()

    exp_root = REPO_ROOT / "experiments" / "acoustic_mlp"
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(exp_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run_dir: {run_dir}")

    sweep_path = run_dir / "retrieval_sweep_symbolic.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing sweep CSV: {sweep_path} (run script 14)")

    df = pd.read_csv(sweep_path)

    need = {"beta_cte", "gamma_ngram", args.select_metric}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Sweep CSV missing columns: {sorted(missing)}")

    df["beta_cte"] = df["beta_cte"].astype(float)
    df["gamma_ngram"] = df["gamma_ngram"].astype(float)

    # va_only row (must exist from sweep)
    va_only = df[(df["beta_cte"] == 0.0) & (df["gamma_ngram"] == 0.0)]
    if len(va_only) != 1:
        raise RuntimeError(f"Expected exactly 1 va_only row, found {len(va_only)}")
    va_only = va_only.iloc[0]

    # va + cte (gamma=0), best beta
    df_cte = df[df["gamma_ngram"] == 0.0].copy()
    va_plus_cte_best = pick_best(df_cte, args.select_metric)

    # va + ngram (beta=0), best gamma
    df_ng = df[df["beta_cte"] == 0.0].copy()
    va_plus_ngram_best = pick_best(df_ng, args.select_metric)

    # full best
    full_best = pick_best(df, args.select_metric)

    rows = []
    for name, s in [
        ("va_only", va_only),
        ("va_plus_cte_best", va_plus_cte_best),
        ("va_plus_ngram_best", va_plus_ngram_best),
        ("full_best", full_best),
    ]:
        row = {"variant": name, "beta_cte": float(s["beta_cte"]), "gamma_ngram": float(s["gamma_ngram"])}
        # keep all metric columns (P@k/R@k/NDCG@k)
        for c in df.columns:
            if c in ("beta_cte", "gamma_ngram"):
                continue
            if c.startswith(("P@", "R@", "NDCG@")):
                row[c] = float(s[c])
        rows.append(row)

    out_path = run_dir / "retrieval_table2.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    LOG.info("Wrote: %s", out_path)

    # also print the chosen params (useful for locking into fig/demo)
    for r in rows:
        LOG.info("%s: beta=%.3f gamma=%.3f", r["variant"], r["beta_cte"], r["gamma_ngram"])


if __name__ == "__main__":
    main()
