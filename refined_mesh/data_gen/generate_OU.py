#!/usr/bin/env python3
"""Generate train/val/test datasets for parameter inference on the Ornstein–Uhlenbeck (OU) process.

SDE:
    dX_t = -theta * X_t dt + sigma dW_t

Discrete-time simulation (Euler–Maruyama):
    X_{t+1} = X_t + (-theta * X_t) * delta + sigma * sqrt(delta) * eps_t
    eps_t ~ N(0,1) i.i.d.

Notes:
- This script simulates *process noise* (noise is cumulative because it changes the state each step).
- Optional burn-in can be used to approach the stationary distribution (mean 0, var = sigma^2/(2 theta)).

Outputs:
    Saves an .npz with keys:
        x_train, y_train, x_val, y_val, x_test, y_test, meta_json
    where x_* has shape (N, L, 1) and y_* has shape (N, 2) (theta, sigma).
"""

import argparse
import json
from pathlib import Path
import numpy as np


def simulate_ou(theta, sigma, L, delta, x0, burn_in=0, meas_noise_std=0.0, rng=None):
    """Simulate OU path and return x-sequence of length L (after optional burn-in)."""
    rng = np.random.default_rng() if rng is None else rng

    theta = float(theta)
    sigma = float(sigma)
    delta = float(delta)

    x = float(x0)

    # Burn-in steps (discard)
    for _ in range(int(burn_in)):
        eps = rng.normal(0.0, 1.0)
        x = x + (-theta * x) * delta + sigma * np.sqrt(delta) * eps

    xs = np.empty((L,), dtype=np.float32)
    for t in range(L):
        eps = rng.normal(0.0, 1.0)
        x = x + (-theta * x) * delta + sigma * np.sqrt(delta) * eps
        xs[t] = x

    # Optional measurement noise (not cumulative)
    if meas_noise_std and meas_noise_std > 0:
        xs = xs + rng.normal(0.0, float(meas_noise_std), size=xs.shape).astype(np.float32)

    return xs


def sample_x0(rng, theta, sigma, mode, x0_range):
    """Sample x0 given a mode."""
    if mode == "zero":
        return 0.0
    if mode == "uniform":
        return float(rng.uniform(x0_range[0], x0_range[1]))
    if mode == "stationary":
        # Stationary OU: N(0, sigma^2 / (2 theta))
        var = (sigma * sigma) / (2.0 * theta)
        return float(rng.normal(0.0, np.sqrt(var)))
    raise ValueError(f"Unknown x0_mode: {mode}")


def make_split(n, L, delta, theta_min, theta_max, sigma_min, sigma_max,
               x0_mode, x0_range, burn_in, meas_noise_std, normalize, seed):
    rng = np.random.default_rng(seed)

    thetas = rng.uniform(theta_min, theta_max, size=(n,)).astype(np.float32)
    sigmas = rng.uniform(sigma_min, sigma_max, size=(n,)).astype(np.float32)

    X = np.empty((n, L, 1), dtype=np.float32)
    Y = np.stack([thetas], axis=1).astype(np.float32)  # (N, 1)

    for i in range(n):
        theta = float(thetas[i])
        sigma = float(sigmas[i])
        x0 = sample_x0(rng, theta=theta, sigma=sigma, mode=x0_mode, x0_range=x0_range)

        xs = simulate_ou(
            theta=theta,
            sigma=sigma,
            L=L,
            delta=delta,
            x0=x0,
            burn_in=burn_in,
            meas_noise_std=meas_noise_std,
            rng=rng,
        )

        if normalize:
            m = float(xs.mean())
            s = float(xs.std())
            xs = (xs - m) / (s + 1e-6)

        X[i, :, 0] = xs

    return X, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output .npz path")
    p.add_argument("--L", type=int, required=True, help="Sequence length")
    p.add_argument("--delta", type=float, required=True, help="Time step size (dt)")

    # Parameter distribution
    p.add_argument("--theta_min", type=float, default=0.2)
    p.add_argument("--theta_max", type=float, default=2.0)
    p.add_argument("--sigma_min", type=float, default=0.1)
    p.add_argument("--sigma_max", type=float, default=1.0)

    # Dataset sizes
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)

    # Initial condition
    p.add_argument("--x0_mode", type=str, default="stationary",
                   choices=["zero", "uniform", "stationary"],
                   help="How to sample x0")
    p.add_argument("--x0_range", type=float, nargs=2, default=(-1.0, 1.0),
                   help="Used only when x0_mode=uniform")

    # Simulation options
    p.add_argument("--burn_in", type=int, default=0, help="Burn-in steps to discard before recording")
    p.add_argument("--meas_noise_std", type=float, default=0.0,
                   help="Additive measurement noise on observations (NOT cumulative)")

    # Optional per-sample normalization
    p.add_argument("--normalize", action="store_true", help="Normalize each sequence to zero-mean/unit-std")

    p.add_argument("--seed", type=int, default=37)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = make_split(
        n=args.n_train, L=args.L, delta=args.delta,
        theta_min=args.theta_min, theta_max=args.theta_max,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        x0_mode=args.x0_mode, x0_range=args.x0_range,
        burn_in=args.burn_in, meas_noise_std=args.meas_noise_std,
        normalize=args.normalize,
        seed=args.seed + 0,
    )
    x_val, y_val = make_split(
        n=args.n_val, L=args.L, delta=args.delta,
        theta_min=args.theta_min, theta_max=args.theta_max,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        x0_mode=args.x0_mode, x0_range=args.x0_range,
        burn_in=args.burn_in, meas_noise_std=args.meas_noise_std,
        normalize=args.normalize,
        seed=args.seed + 1,
    )
    x_test, y_test = make_split(
        n=args.n_test, L=args.L, delta=args.delta,
        theta_min=args.theta_min, theta_max=args.theta_max,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        x0_mode=args.x0_mode, x0_range=args.x0_range,
        burn_in=args.burn_in, meas_noise_std=args.meas_noise_std,
        normalize=args.normalize,
        seed=args.seed + 2,
    )

    meta = {
        "system": "ornstein_uhlenbeck",
        "obs": "x(t) only",
        "label": "(theta, sigma)",
        "discretization": "euler_maruyama",
        "L": int(args.L),
        "delta": float(args.delta),
        "theta_min": float(args.theta_min),
        "theta_max": float(args.theta_max),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "x0_mode": str(args.x0_mode),
        "x0_range": list(map(float, args.x0_range)),
        "burn_in": int(args.burn_in),
        "meas_noise_std": float(args.meas_noise_std),
        "normalize": bool(args.normalize),
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
