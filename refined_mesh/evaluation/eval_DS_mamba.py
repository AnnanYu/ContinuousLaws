#!/usr/bin/env python3
"""Evaluate an S6 model to infer mu from 1D Van der Pol oscillator time series.

Expected NPZ keys:
    x_train, y_train, x_val, y_val, x_test, y_test
where x_* has shape (N, L, 1) and y_* has shape (N, 1)
"""

import os
import argparse
import random
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.s4.mamba_VDP import MambaBlock as Mamba


# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split(".")[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
elif tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NPZTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str, split: str):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        x_key = f"x_{split}"
        y_key = f"y_{split}"
        if x_key not in data or y_key not in data:
            raise KeyError(f"Missing {x_key}/{y_key} in {npz_path}")

        self.x = data[x_key].astype(np.float32)   # (N, L, 1)
        self.y = data[y_key].astype(np.float32)   # (N, d_out)

        if not (self.x.ndim == 3 and self.x.shape[-1] == 1):
            raise ValueError(f"Expected x shape (N,L,1), got {self.x.shape}")

        # Accept (N,) or (N, d_out); convert (N,) -> (N,1)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        if self.y.ndim != 2:
            raise ValueError(f"Expected y shape (N,) or (N,d_out), got {self.y.shape}")

    @property
    def d_output(self) -> int:
        return int(self.y.shape[1])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # x: (L, 1), y: (d_output,)
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


class S6Regressor(nn.Module):
    def __init__(self, d_input=1, d_output=1, d_model=128, n_layers=4, dropout=0.1, prenorm=False, scale_dt=1.0):
        super().__init__()
        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s6_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s6_layers.append(
                Mamba(d_model=d_model, scale_dt=scale_dt)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_output),
        )

    def forward(self, x):
        # x: (B, L, 1)
        x = self.encoder(x)              # (B, L, d_model)
        # x = x.transpose(-1, -2)          # (B, d_model, L)

        for layer, norm, dropout in zip(self.s6_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)

            z = layer(z)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x)

        # x = x.transpose(-1, -2)          # (B, L, d_model)
        x = x.mean(dim=1)                # (B, d_model)
        y = self.decoder(x)
        return y


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    sse = 0.0
    sae = 0.0
    n = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        err = pred - y
        sse += torch.sum(err * err).item()
        sae += torch.sum(torch.abs(err)).item()
        n += y.numel()
    rmse = (sse / max(n, 1)) ** 0.5
    mae = sae / max(n, 1)
    return mae, rmse


def main():
    p = argparse.ArgumentParser(description="Evaluate Mamba on dynamical systems data.")

    # Data
    p.add_argument("--data", type=str, required=True, help="Path to .npz produced by generate_vdp_dataset.py")
    p.add_argument("--scale_dt", type=float, required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)

    # Model
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prenorm", action="store_true")

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="checkpoint_vdp")
    p.add_argument("--seed", type=int, default=37)

    args = p.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cudnn.benchmark = True

    # Data
    testset = NPZTimeSeriesDataset(args.data, "test")
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    # Model
    model = S6Regressor(
        d_input=1, d_output=testset.d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        scale_dt=args.scale_dt,
    ).to(device)

    # Checkpoint paths
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")

    if os.path.exists(best_path):
        best = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best["model"])
        model.to(device)
        test_mae, test_rmse = evaluate(model, testloader, device)
        print(f"test MAE {test_mae:.6f} | test RMSE {test_rmse:.6f}")
    else:
        print("invalid checkpoint path!")


if __name__ == "__main__":
    main()
