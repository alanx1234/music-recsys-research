"""
Create Table 1 (prediction MAE/MSE) from run metrics.json.

Reads:
  experiments/acoustic_mlp/run_###/metrics.json
  experiments/symbolic_lstm/run_###/metrics.json
  experiments/hybrid_lstm_energy/run_###/metrics.json

Writes:
  experiments/prediction_table1.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_metrics(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    test = obj["metrics"]["test"]
    return {
        "run_dir": obj.get("run_dir", str(path.parent)),
        "mse": float(test["mse"]),
        "mae": float(test["mae"]),
        "mae_valence": float(test["mae_valence"]),
        "mae_arousal": float(test["mae_arousal"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acoustic", type=str, required=True, help="Path to acoustic metrics.json")
    p.add_argument("--symbolic", type=str, required=True, help="Path to symbolic metrics.json")
    p.add_argument("--hybrid", type=str, required=True, help="Path to hybrid metrics.json")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "experiments" / "prediction_table1.csv"))
    args = p.parse_args()

    rows = []
    rows.append({"model": "acoustic_only", **load_metrics(Path(args.acoustic))})
    rows.append({"model": "symbolic_only", **load_metrics(Path(args.symbolic))})
    rows.append({"model": "hybrid", **load_metrics(Path(args.hybrid))})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
