"""
Acoustic-only baseline (v0): predict (valence, arousal) from [log_rms_mean, log_rms_std].

Inputs:
  data/processed/features_v0_energy.pt  

Outputs (run directory):
  experiments/acoustic_mlp/run_###/metrics.json
  experiments/acoustic_mlp/run_###/preds.csv
  experiments/acoustic_mlp/run_###/model.pt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PT = REPO_ROOT / "data" / "processed" / "features_v0_energy.pt"
EXP_ROOT = REPO_ROOT / "experiments" / "acoustic_mlp"

SEED = 42

# Training hyperparams 
HIDDEN = 64
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20  # early stop epochs without val improvement

X_COLS = ["log_rms_mean", "log_rms_std"]
Y_COLS = ["valence", "arousal"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Standardizer:
    mean: torch.Tensor  # shape: [D]
    std: torch.Tensor   # shape: [D]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D] -> [N, D]
        return (x - self.mean) / self.std


class AcousticMLP(nn.Module):
    def __init__(self, d_in: int = 2, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),   # Output: [B, d_hidden]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),  # Output: [B, d_hidden]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),      # Output: [B, 2] for (V,A)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2] -> yhat: [B, 2]
        return self.net(x)


def make_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.glob("run_*") if p.is_dir()]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_")[-1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    run_dir = base / f"run_{nxt:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def to_tensor(df: pd.DataFrame, cols: list[str], device: torch.device) -> torch.Tensor:
    arr = df[cols].to_numpy(dtype=np.float32)
    t = torch.from_numpy(arr).to(device)
    return t


def mae(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # both [N, 2] -> scalar
    return torch.mean(torch.abs(yhat - y))


@torch.no_grad()
def eval_split(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    yhat = model(x)  # [N,2]
    mse = torch.mean((yhat - y) ** 2).item()
    mae_all = torch.mean(torch.abs(yhat - y)).item()
    mae_v = torch.mean(torch.abs(yhat[:, 0] - y[:, 0])).item()
    mae_a = torch.mean(torch.abs(yhat[:, 1] - y[:, 1])).item()
    return {"mse": mse, "mae": mae_all, "mae_valence": mae_v, "mae_arousal": mae_a}


def main() -> None:
    set_seed(SEED)

    if not CACHE_PT.exists():
        raise FileNotFoundError(f"Missing cache: {CACHE_PT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    blob = torch.load(CACHE_PT, map_location="cpu", weights_only=False)
    df = blob["data"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame in cache, got: {type(df)}")

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    x_train = to_tensor(df_train, X_COLS, device=device)  # [Ntr,2]
    y_train = to_tensor(df_train, Y_COLS, device=device)  # [Ntr,2]
    x_val = to_tensor(df_val, X_COLS, device=device)      # [Nva,2]
    y_val = to_tensor(df_val, Y_COLS, device=device)      # [Nva,2]
    x_test = to_tensor(df_test, X_COLS, device=device)    # [Nte,2]
    y_test = to_tensor(df_test, Y_COLS, device=device)    # [Nte,2]

    feat_mean = torch.mean(x_train, dim=0)                # [2]
    feat_std = torch.std(x_train, dim=0).clamp_min(1e-6)  # [2]
    scaler = Standardizer(mean=feat_mean, std=feat_std)

    x_train_s = scaler.transform(x_train)  # [Ntr,2]
    x_val_s = scaler.transform(x_val)      # [Nva,2]
    x_test_s = scaler.transform(x_test)    # [Nte,2]

    model = AcousticMLP(d_in=2, d_hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_train = x_train_s.shape[0]
    indices = np.arange(n_train)

    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        np.random.shuffle(indices)

        losses = []
        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            xb = x_train_s[batch_idx]  # [B,2]
            yb = y_train[batch_idx]    # [B,2]

            yhat = model(xb)           # [B,2]
            loss = F.mse_loss(yhat, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_metrics = eval_split(model, x_train_s, y_train)
        val_metrics = eval_split(model, x_val_s, y_val)

        print(
            f"Epoch {epoch:03d} | "
            f"train_mae={train_metrics['mae']:.4f} val_mae={val_metrics['mae']:.4f} | "
            f"val(V,A)=({val_metrics['mae_valence']:.4f},{val_metrics['mae_arousal']:.4f})"
        )

        # Early stopping on val MAE
        if val_metrics["mae"] + 1e-6 < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": feat_mean.detach().cpu(),
                "scaler_std": feat_std.detach().cpu(),
            }
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val_mae={best_val_mae:.4f})")
                break

    elapsed = time.time() - t0

    if best_state is None:
        raise RuntimeError("Training failed to produce a best_state (unexpected).")

    model.load_state_dict(best_state["model"])
    feat_mean = best_state["scaler_mean"].to(device)
    feat_std = best_state["scaler_std"].to(device)
    scaler = Standardizer(mean=feat_mean, std=feat_std)

    train_metrics = eval_split(model, scaler.transform(x_train), y_train)
    val_metrics = eval_split(model, scaler.transform(x_val), y_val)
    test_metrics = eval_split(model, scaler.transform(x_test), y_test)

    run_dir = make_run_dir(EXP_ROOT)
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "preds.csv"
    model_path = run_dir / "model.pt"

    model.eval()
    with torch.no_grad():
        yhat_test = model(scaler.transform(x_test))  # [Nte,2]
    yhat_test_np = yhat_test.detach().cpu().numpy()

    out_preds = df_test[["song_id", "split", "audio_path", "valence", "arousal"]].copy()
    out_preds["pred_valence"] = yhat_test_np[:, 0]
    out_preds["pred_arousal"] = yhat_test_np[:, 1]
    out_preds.to_csv(preds_path, index=False)

    payload = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT).as_posix()),
        "device": str(device),
        "seed": SEED,
        "features": X_COLS,
        "targets": Y_COLS,
        "model": {
            "type": "AcousticMLP",
            "hidden": HIDDEN,
            "dropout": DROPOUT,
        },
        "train": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "best_epoch": best_epoch,
            "elapsed_sec": elapsed,
        },
        "standardization": {
            "mean": best_state["scaler_mean"].tolist(),
            "std": best_state["scaler_std"].tolist(),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": best_state["scaler_mean"],
            "scaler_std": best_state["scaler_std"],
            "x_cols": X_COLS,
            "y_cols": Y_COLS,
        },
        model_path,
    )

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {preds_path}")
    print(f"  {model_path}")
    print("\nFinal test metrics:")
    print(f"  MAE overall:  {test_metrics['mae']:.4f}")
    print(f"  MAE valence:  {test_metrics['mae_valence']:.4f}")
    print(f"  MAE arousal:  {test_metrics['mae_arousal']:.4f}")


if __name__ == "__main__":
    main()