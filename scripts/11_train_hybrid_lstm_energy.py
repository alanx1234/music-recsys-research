"""
Hybrid predictor (Layers 3-4): LSTM(tokens) + MLP(energy) -> (valence, arousal).

Inputs:
  data/processed/features_v0_energy.csv
  data/processed/chords_tokens/chords_tokens_cache.pkl

Outputs (run directory):
  experiments/hybrid_lstm_energy/run_###/metrics.json
  experiments/hybrid_lstm_energy/run_###/preds.csv
  experiments/hybrid_lstm_energy/run_###/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("train_hybrid_lstm_energy")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.glob("run_*") if p.is_dir()]
    nums: List[int] = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_")[-1]))
        except Exception:
            pass
    nxt = (max(nums) + 1) if nums else 1
    run_dir = base / f"run_{nxt:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int

    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in toks]


def build_vocab(all_tokens: List[List[str]], min_freq: int) -> Vocab:
    counts: Dict[str, int] = {}
    for seq in all_tokens:
        for t in seq:
            counts[t] = counts.get(t, 0) + 1

    itos = ["<PAD>", "<UNK>"]
    for tok, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            itos.append(tok)

    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_id=0, unk_id=1)


class HybridDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        token_map: Dict[int, List[str]],
        vocab: Vocab,
        x_cols: List[str],
        max_len: int,
        min_len: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.token_map = token_map
        self.vocab = vocab
        self.x_cols = x_cols
        self.max_len = int(max_len)
        self.min_len = int(min_len)

        keep = []
        for i, r in self.df.iterrows():
            sid = int(r["song_id"])
            toks = self.token_map.get(sid, [])
            if len(toks) >= self.min_len:
                keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, int]:
        r = self.df.iloc[idx]
        sid = int(r["song_id"])

        y = torch.tensor([float(r["valence"]), float(r["arousal"])], dtype=torch.float32)  # Shape: [2]
        x_energy = torch.tensor([float(r[c]) for c in self.x_cols], dtype=torch.float32)  # Shape: [2]

        toks = self.token_map.get(sid, [])
        ids = self.vocab.encode(toks[: self.max_len])
        length = len(ids)
        x_ids = torch.tensor(ids, dtype=torch.long)  # Shape: [T]

        return x_ids, length, x_energy, y, sid


def collate_batch(batch: List[Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ids, lens, x_e, y, sids = zip(*batch)

    lens_t = torch.tensor(lens, dtype=torch.long)        # Shape: [B]
    x_e_t = torch.stack(list(x_e), dim=0)                # Shape: [B, 2]
    y_t = torch.stack(list(y), dim=0)                    # Shape: [B, 2]
    sids_t = torch.tensor(sids, dtype=torch.long)        # Shape: [B]

    max_len = int(max(lens))
    x_pad = torch.full((len(x_ids), max_len), pad_id, dtype=torch.long)  # Shape: [B, Tmax]
    for i, x in enumerate(x_ids):
        x_pad[i, : x.shape[0]] = x

    return x_pad, lens_t, x_e_t, y_t, sids_t


class HybridModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        energy_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # Shape: [B, T] -> [B, T, E]
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )  # h_n Shape: [L, B, Hs]

        self.energy_mlp = nn.Sequential(
            nn.Linear(2, energy_hidden),  # Shape: [B, 2] -> [B, He]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden, energy_hidden),  # Shape: [B, He] -> [B, He]
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + energy_hidden, lstm_hidden + energy_hidden),  # Shape: [B, Hs+He] -> [B, Hs+He]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden + energy_hidden, 2),  # Shape: [B, Hs+He] -> [B, 2]
        )

    def forward(self, x_ids: torch.Tensor, lengths: torch.Tensor, x_energy: torch.Tensor) -> torch.Tensor:
        assert x_ids.ndim == 2
        assert lengths.ndim == 1
        assert x_energy.ndim == 2 and x_energy.shape[1] == 2

        emb = self.emb(x_ids)  # Shape: [B, T] -> [B, T, E]
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # Shape: [L, B, Hs]
        h_sym = h_n[-1]                  # Shape: [B, Hs]

        h_eng = self.energy_mlp(x_energy)  # Shape: [B, He]
        h = torch.cat([h_sym, h_eng], dim=1)  # Shape: [B, Hs+He]
        out = self.head(h)                   # Shape: [B, 2]
        return out


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys: List[torch.Tensor] = []
    yh: List[torch.Tensor] = []

    for x_ids, lengths, x_e, y, _sids in loader:
        x_ids = x_ids.to(device)
        lengths = lengths.to(device)
        x_e = x_e.to(device)
        y = y.to(device)

        yhat = model(x_ids, lengths, x_e)  # Shape: [B, 2]
        ys.append(y)
        yh.append(yhat)

    Y = torch.cat(ys, dim=0)   # Shape: [N, 2]
    YH = torch.cat(yh, dim=0)  # Shape: [N, 2]

    mse = torch.mean((YH - Y) ** 2).item()
    mae_all = torch.mean(torch.abs(YH - Y)).item()
    mae_v = torch.mean(torch.abs(YH[:, 0] - Y[:, 0])).item()
    mae_a = torch.mean(torch.abs(YH[:, 1] - Y[:, 1])).item()
    return {"mse": mse, "mae": mae_all, "mae_valence": mae_v, "mae_arousal": mae_a}


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", type=str, default=str(REPO_ROOT / "data" / "processed" / "features_v0_energy.csv"))
    p.add_argument("--tokens_cache", type=str, default=str(REPO_ROOT / "data" / "processed" / "chords_tokens" / "chords_tokens_cache.pkl"))
    p.add_argument("--exp_root", type=str, default=str(REPO_ROOT / "experiments" / "hybrid_lstm_energy"))

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--min_freq", type=int, default=1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--min_len", type=int, default=1)

    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--energy_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--grad_clip", type=float, default=1.0)

    args = p.parse_args()
    seed_everything(args.seed)

    features_csv = Path(args.features_csv)
    tokens_cache = Path(args.tokens_cache)
    exp_root = Path(args.exp_root)

    if not features_csv.exists():
        raise FileNotFoundError(f"Missing: {features_csv}")
    if not tokens_cache.exists():
        raise FileNotFoundError(f"Missing: {tokens_cache}")

    df = pd.read_csv(features_csv)
    req = {"song_id", "split", "valence", "arousal", "log_rms_mean", "log_rms_std"}
    if not req.issubset(df.columns):
        raise ValueError(f"features csv missing columns: {sorted(req - set(df.columns))}")

    with tokens_cache.open("rb") as f:
        cache = pickle.load(f)
    token_map: Dict[int, List[str]] = {int(sid): list(d.get("tokens", [])) for sid, d in cache.items()}

    x_cols = ["log_rms_mean", "log_rms_std"]

    train_df = df[df["split"] == "train"].copy()
    feat_mean = train_df[x_cols].to_numpy(np.float64).mean(axis=0)
    feat_std = train_df[x_cols].to_numpy(np.float64).std(axis=0)
    feat_std = np.maximum(feat_std, 1e-6)

    df[x_cols] = (df[x_cols] - feat_mean) / feat_std

    train_ids = df[df["split"] == "train"]["song_id"].astype(int).tolist()
    train_tokens = [token_map.get(sid, [])[: args.max_len] for sid in train_ids if len(token_map.get(sid, [])) >= args.min_len]
    vocab = build_vocab(train_tokens, min_freq=args.min_freq)

    LOG.info("Energy standardization: mean=%s std=%s", feat_mean.tolist(), feat_std.tolist())
    LOG.info("Vocab size=%d (min_freq=%d)", len(vocab.itos), args.min_freq)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    ds_train = HybridDataset(df_train, token_map, vocab, x_cols=x_cols, max_len=args.max_len, min_len=args.min_len)
    ds_val = HybridDataset(df_val, token_map, vocab, x_cols=x_cols, max_len=args.max_len, min_len=args.min_len)
    ds_test = HybridDataset(df_test, token_map, vocab, x_cols=x_cols, max_len=args.max_len, min_len=args.min_len)

    LOG.info("Dataset sizes: train=%d val=%d test=%d", len(ds_train), len(ds_val), len(ds_test))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Device: %s", device)

    collate = lambda b: collate_batch(b, pad_id=vocab.pad_id)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = HybridModel(
        vocab_size=len(vocab.itos),
        emb_dim=args.emb_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        energy_hidden=args.energy_hidden,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()

        for step, (x_ids, lengths, x_e, y, _sids) in enumerate(dl_train, start=1):
            x_ids = x_ids.to(device)
            lengths = lengths.to(device)
            x_e = x_e.to(device)
            y = y.to(device)

            if epoch == 1 and step == 1:
                LOG.info("First batch: energy_mean=%.4f energy_std=%.4f",
                         float(x_e.mean().item()), float(x_e.std(unbiased=False).item()))

            yhat = model(x_ids, lengths, x_e)  # Shape: [B, 2]
            loss = F.mse_loss(yhat, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {loss.item()}")

            opt.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite grad_norm: {float(grad_norm)}")

            opt.step()

        train_m = eval_loader(model, dl_train, device)
        val_m = eval_loader(model, dl_val, device)

        LOG.info(
            "Epoch %03d | train_mae=%.4f val_mae=%.4f | val(V,A)=(%.4f,%.4f)",
            epoch, train_m["mae"], val_m["mae"], val_m["mae_valence"], val_m["mae_arousal"]
        )

        if val_m["mae"] + 1e-6 < best_val:
            best_val = val_m["mae"]
            best_epoch = epoch
            best_state = {"model_state_dict": model.state_dict()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                LOG.info("Early stopping at epoch %d (best_epoch=%d best_val_mae=%.4f)", epoch, best_epoch, best_val)
                break

    elapsed = time.time() - t0
    if best_state is None:
        raise RuntimeError("Training failed to produce a best_state.")

    model.load_state_dict(best_state["model_state_dict"])

    train_m = eval_loader(model, dl_train, device)
    val_m = eval_loader(model, dl_val, device)
    test_m = eval_loader(model, dl_test, device)

    run_dir = make_run_dir(exp_root)
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "preds.csv"
    model_path = run_dir / "model.pt"

    model.eval()
    rows = []
    with torch.no_grad():
        for x_ids, lengths, x_e, y, sids in dl_test:
            x_ids = x_ids.to(device)
            lengths = lengths.to(device)
            x_e = x_e.to(device)

            yhat = model(x_ids, lengths, x_e).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            sids = sids.detach().cpu().numpy()

            for i in range(len(sids)):
                rows.append({
                    "song_id": int(sids[i]),
                    "split": "test",
                    "valence": float(y[i, 0]),
                    "arousal": float(y[i, 1]),
                    "pred_valence": float(yhat[i, 0]),
                    "pred_arousal": float(yhat[i, 1]),
                })
    pd.DataFrame(rows).sort_values("song_id").to_csv(preds_path, index=False)

    payload = {
        "run_dir": str(run_dir.relative_to(REPO_ROOT).as_posix()),
        "seed": int(args.seed),
        "device": str(device),
        "model": {
            "type": "HybridModel",
            "emb_dim": int(args.emb_dim),
            "lstm_hidden": int(args.lstm_hidden),
            "lstm_layers": int(args.lstm_layers),
            "energy_hidden": int(args.energy_hidden),
            "dropout": float(args.dropout),
        },
        "data": {
            "features_csv": str(features_csv.relative_to(REPO_ROOT).as_posix()),
            "tokens_cache": str(tokens_cache.relative_to(REPO_ROOT).as_posix()),
            "x_cols": x_cols,
            "max_len": int(args.max_len),
            "min_len": int(args.min_len),
            "min_freq": int(args.min_freq),
            "vocab_size": int(len(vocab.itos)),
        },
        "standardization": {"mean": feat_mean.tolist(), "std": feat_std.tolist()},
        "train": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "grad_clip": float(args.grad_clip),
            "best_epoch": int(best_epoch),
            "elapsed_sec": float(elapsed),
        },
        "metrics": {"train": train_m, "val": val_m, "test": test_m},
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_itos": vocab.itos,
            "pad_id": vocab.pad_id,
            "unk_id": vocab.unk_id,
            "max_len": int(args.max_len),
            "min_len": int(args.min_len),
            "energy_mean": feat_mean.astype(np.float32),
            "energy_std": feat_std.astype(np.float32),
            "x_cols": x_cols,
        },
        model_path,
    )

    LOG.info("Wrote: %s", metrics_path)
    LOG.info("Wrote: %s", preds_path)
    LOG.info("Wrote: %s", model_path)
    LOG.info("Final test: MAE=%.4f | MAE_V=%.4f MAE_A=%.4f", test_m["mae"], test_m["mae_valence"], test_m["mae_arousal"])


if __name__ == "__main__":
    main()
