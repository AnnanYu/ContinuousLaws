import os
import argparse
import random
import numpy as np
from tqdm.auto import tqdm
import wandb
# from fla.layers.gated_deltanet import GatedDeltaNet
# from fla.layers.delta_net import DeltaNet
from fla.layers.gla import GatedLinearAttention
from fla.layers.linear_attn import LinearAttention
from fla.layers.mamba3 import Mamba3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from lit_gpt.gated_delta_net import GatedDeltaNet

MODELS = ["gated_deltanet", "delta_net", "gla", "linear_attn", "mamba3"]


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

        self.x = data[x_key].astype(np.float32)   # (N, L, d_input)
        self.y = data[y_key].astype(np.float32)   # (N, d_out)

        if not (self.x.ndim == 3):
            raise ValueError(f"Expected x shape (N,L,d_input), got {self.x.shape}")

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
        x = torch.from_numpy(self.x[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y


def build_layer(
    model: str,
    d_model: int,
    num_heads: int,
    expand_k: float,
    expand_v: float,
    head_dim: int,
    state_size: int,
    layer_idx: int,
) -> nn.Module:
    if model == "gated_deltanet":
        # key_dim = num_heads * head_dim; no expand_k
        return GatedDeltaNet(
            hidden_size=d_model,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            layer_idx=layer_idx,
        )
    elif model == "delta_net":
        return GatedDeltaNet(
            hidden_size=d_model,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            layer_idx=layer_idx,
            use_mamba_gate=False,
        )
    elif model == "gla":
        return GatedLinearAttention(
            hidden_size=d_model,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            layer_idx=layer_idx,
        )
    elif model == "linear_attn":
        return LinearAttention(
            hidden_size=d_model,
            num_heads=1,
            expand_k=expand_k,
            expand_v=expand_v,
            layer_idx=layer_idx,
        )
    elif model == "mamba3":
        return Mamba3(
            hidden_size=d_model,
            state_size=state_size,
            head_dim=head_dim,
            layer_idx=layer_idx,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


class SequenceRegressor(nn.Module):
    def __init__(
        self,
        model: str = "gated_deltanet",
        d_input: int = 1,
        d_output: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        head_dim: int = 32,
        state_size: int = 128,
        dropout: float = 0.1,
        prenorm: bool = False,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(build_layer(
                model=model,
                d_model=d_model,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                head_dim=head_dim,
                state_size=state_size,
                layer_idx=i,
            ))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_input)
        x = self.encoder(x)                        # (B, L, d_model)

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z)

            z, _, _ = layer(z)                     # (B, L, d_model)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x)

        x = x.mean(dim=1)                          # (B, d_model)
        return self.decoder(x)                     # (B, d_output)


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
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data", type=str, required=True, help="Path to the .npz dataset")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=128)

    # Model
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prenorm", action="store_true")
    p.add_argument("--model", type=str, default="gated_deltanet", choices=MODELS,
                   help="Sequence layer to use (default: gated_deltanet)")
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--expand_k", type=float, default=1.0,
                   help="Key expansion ratio (used by delta_net, gla, linear_attn)")
    p.add_argument("--expand_v", type=float, default=1.0,
                   help="Value expansion ratio (used by delta_net, gla, linear_attn, gated_deltanet)")
    p.add_argument("--head_dim", type=int, default=16,
                   help="Head dimension (used by mamba3)")
    p.add_argument("--state_size", type=int, default=128,
                   help="SSM state size (used by mamba3)")
    p.add_argument("--d_input", type=int, default=16)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="Checkpoint directory (default: checkpoint_<model>)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=37)

    # Wandb
    p.add_argument("--wandb_project", type=str, default="continuous_benchmarking")
    p.add_argument("--wandb_name", type=str, default=None)

    args = p.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = f"checkpoint_{args.model}"

    set_seed(args.seed)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cudnn.benchmark = True

    if args.model == "mamba3" and device != "cuda":
        raise RuntimeError("mamba3 requires a CUDA device.")

    trainset = NPZTimeSeriesDataset(args.data, "train")
    valset = NPZTimeSeriesDataset(args.data, "val")
    testset = NPZTimeSeriesDataset(args.data, "test")

    d_output = trainset.d_output
    print(f"Model: {args.model} | Inferred d_output = {d_output} from dataset labels.")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    model = SequenceRegressor(
        model=args.model,
        d_input=args.d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        expand_k=args.expand_k,
        expand_v=args.expand_v,
        head_dim=args.head_dim,
        state_size=args.state_size,
        dropout=args.dropout,
        prenorm=args.prenorm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best.pt")
    last_path = os.path.join(args.ckpt_dir, "last.pt")

    best_val_rmse = float("inf")
    start_epoch = 0

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        best_val_rmse = ckpt.get("best_val_rmse", best_val_rmse)
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resumed from {last_path} at epoch {start_epoch} (best val RMSE {best_val_rmse:.6f}).")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        n_seen = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs-1}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.numel()
            n_seen += y.numel()
            pbar.set_postfix(train_mse=running / max(n_seen, 1))

        scheduler.step()

        train_rmse = (running / max(n_seen, 1)) ** 0.5
        val_mae, val_rmse = evaluate(model, valloader, device)
        test_mae, test_rmse = evaluate(model, testloader, device)
        print(
            f"[Epoch {epoch}] val MAE {val_mae:.6f} | val RMSE {val_rmse:.6f} "
            f"|| test MAE {test_mae:.6f} | test RMSE {test_rmse:.6f}"
        )
        if args.wandb_project:
            wandb.log({
                "train/rmse": train_rmse,
                "val/mae": val_mae,
                "val/rmse": val_rmse,
                "test/mae": test_mae,
                "test/rmse": test_rmse,
            }, step=epoch)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "best_val_rmse": best_val_rmse,
                "args": vars(args),
            },
            last_path,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_val_rmse": best_val_rmse,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"New best checkpoint saved (val RMSE {best_val_rmse:.6f}) to {best_path}")

    if os.path.exists(best_path):
        best = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best["model"])
        model.to(device)
        val_mae, val_rmse = evaluate(model, valloader, device)
        test_mae, test_rmse = evaluate(model, testloader, device)
        print(
            f"[Best] val MAE {val_mae:.6f} | val RMSE {val_rmse:.6f} "
            f"|| test MAE {test_mae:.6f} | test RMSE {test_rmse:.6f}"
        )
        if args.wandb_project:
            wandb.summary["best/val_rmse"] = val_rmse
            wandb.summary["best/val_mae"] = val_mae
            wandb.summary["best/test_rmse"] = test_rmse
            wandb.summary["best/test_mae"] = test_mae

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
