#!/usr/bin/env python3
"""Generate train/val/test datasets for parameter inference on the (autonomous) Van der Pol oscillator.

Dynamics:
    x'' - mu (1 - x^2) x' + x = 0
Convert to first order:
    x' = v
    v' = mu (1 - x^2) v - x

We integrate with RK4 at step size delta and return a length-L sequence of x values.

Outputs:
    Saves an .npz with keys:
        x_train, y_train, x_val, y_val, x_test, y_test, meta_json
    where x_* has shape (N, L, 1) and y_* has shape (N, 1).
"""

import argparse
import json
from pathlib import Path
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("No tqdm")
    tqdm = None


def progress(iterable, **kwargs):
    """tqdm wrapper with graceful fallback if tqdm isn't installed."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def rk4_step(x, v, mu, dt):
    """One RK4 step for the Van der Pol first-order system."""
    # x' = v
    # v' = mu (1 - x^2) v - x
    def f(x_, v_):
        dx = v_
        dv = mu * (1.0 - x_ * x_) * v_ - x_
        return dx, dv

    k1x, k1v = f(x, v)
    k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
    k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
    k4x, k4v = f(x + dt * k3x, v + dt * k3v)

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_next, v_next


def simulate_vdp(mu, L, delta, x0, v0, burn_in=0, noise_std=0.0, rng=None):
    """Simulate and return an x-sequence of length L (after optional burn-in)."""
    rng = np.random.default_rng() if rng is None else rng

    x = float(x0)
    v = float(v0)

    for _ in range(int(burn_in)):
        x, v = rk4_step(x, v, float(mu), float(delta))

    xs = np.empty((L,), dtype=np.float32)
    for t in range(L):
        x, v = rk4_step(x, v, float(mu), float(delta))
        xs[t] = x

    if noise_std and noise_std > 0:
        xs = xs + rng.normal(0.0, float(noise_std), size=xs.shape).astype(np.float32)

    return xs


def make_split(n, L, delta, mu_min, mu_max, x0_range, v0_range, burn_in, noise_std, seed, split_name="split"):
    rng = np.random.default_rng(seed)

    mus = rng.uniform(mu_min, mu_max, size=(n,)).astype(np.float32)
    x0s = rng.uniform(x0_range[0], x0_range[1], size=(n,)).astype(np.float32)
    v0s = rng.uniform(v0_range[0], v0_range[1], size=(n,)).astype(np.float32)

    X = np.empty((n, L, 1), dtype=np.float32)
    Y = mus.reshape(n, 1).astype(np.float32)

    for i in progress(range(n), total=n, desc=f"Generating {split_name}", unit="traj"):
        xs = simulate_vdp(
            mu=float(mus[i]),
            L=L,
            delta=delta,
            x0=float(x0s[i]),
            v0=float(v0s[i]),
            burn_in=burn_in,
            noise_std=noise_std,
            rng=rng,
        )
        X[i, :, 0] = xs

    return X, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output .npz path")
    p.add_argument("--L", type=int, required=True, help="Sequence length")
    p.add_argument("--delta", type=float, required=True, help="Integration step size")

    # Parameter distribution (mu)
    p.add_argument("--mu_min", type=float, default=0.1)
    p.add_argument("--mu_max", type=float, default=5.0)

    # Dataset sizes
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)

    # Initial condition ranges
    p.add_argument("--x0_range", type=float, nargs=2, default=(-2.0, 2.0))
    p.add_argument("--v0_range", type=float, nargs=2, default=(-2.0, 2.0))

    # Simulation options
    p.add_argument("--burn_in", type=int, default=0, help="Number of steps to discard before recording")
    p.add_argument("--noise_std", type=float, default=0.0, help="Additive Gaussian noise on x(t)")

    p.add_argument("--seed", type=int, default=37)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = make_split(
        n=args.n_train, L=args.L, delta=args.delta,
        mu_min=args.mu_min, mu_max=args.mu_max,
        x0_range=args.x0_range, v0_range=args.v0_range,
        burn_in=args.burn_in, noise_std=args.noise_std,
        seed=args.seed + 0,
        split_name="train"
    )
    x_val, y_val = make_split(
        n=args.n_val, L=args.L, delta=args.delta,
        mu_min=args.mu_min, mu_max=args.mu_max,
        x0_range=args.x0_range, v0_range=args.v0_range,
        burn_in=args.burn_in, noise_std=args.noise_std,
        seed=args.seed + 1,
        split_name="val"
    )
    x_test, y_test = make_split(
        n=args.n_test, L=args.L, delta=args.delta,
        mu_min=args.mu_min, mu_max=args.mu_max,
        x0_range=args.x0_range, v0_range=args.v0_range,
        burn_in=args.burn_in, noise_std=args.noise_std,
        seed=args.seed + 2,
        split_name="test"
    )

    meta = {
        "system": "van_der_pol",
        "obs": "x(t) only",
        "label": "mu",
        "L": int(args.L),
        "delta": float(args.delta),
        "mu_min": float(args.mu_min),
        "mu_max": float(args.mu_max),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "x0_range": list(map(float, args.x0_range)),
        "v0_range": list(map(float, args.v0_range)),
        "burn_in": int(args.burn_in),
        "noise_std": float(args.noise_std),
        "seed": int(args.seed),
    }

    np.savez_compressed(
        out_path,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        meta_json=json.dumps(meta),
    )
    print(f"Saved: {out_path}")
    print("Shapes:",
          f"x_train {x_train.shape}, y_train {y_train.shape} | ",
          f"x_val {x_val.shape}, y_val {y_val.shape} | ",
          f"x_test {x_test.shape}, y_test {y_test.shape}")


if __name__ == "__main__":
    main()
