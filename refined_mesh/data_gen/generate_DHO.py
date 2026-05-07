#!/usr/bin/env python3
"""Generate train/val/test datasets for parameter inference on a damped harmonic oscillator.

Dynamics (2nd order):
    x¨ + 2*zeta*omega * x˙ + omega^2 * x = 0

Convert to first order:
    x' = v
    v' = -2*zeta*omega*v - omega^2*x

We integrate with RK4 at step size delta and return a length-L sequence of x values.

Outputs:
    Saves an .npz with keys:
        x_train, y_train, x_val, y_val, x_test, y_test, meta_json
    where x_* has shape (N, L, 1) and y_* has shape (N, 2) (omega, zeta).

Notes / pitfalls:
- Because the system decays to 0, very long sequences can become nearly flat and
  make omega harder to infer. Keep L*delta within a few dozen periods.
- Optionally you can set --normalize to standardize each sample (mean/std).
"""

import argparse
import json
from pathlib import Path
import numpy as np


def rk4_step(x, v, omega, zeta, dt):
    """One RK4 step for the damped harmonic oscillator first-order system."""
    # x' = v
    # v' = -2*zeta*omega*v - omega^2*x
    def f(x_, v_):
        dx = v_
        dv = -2.0 * zeta * omega * v_ - (omega * omega) * x_
        return dx, dv

    k1x, k1v = f(x, v)
    k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
    k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
    k4x, k4v = f(x + dt * k3x, v + dt * k3v)

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_next, v_next


def simulate_dho(omega, zeta, L, delta, x0, v0, noise_std=0.0, rng=None):
    """Simulate and return an x-sequence of length L."""
    rng = np.random.default_rng() if rng is None else rng

    x = float(x0)
    v = float(v0)
    omega = float(omega)
    zeta = float(zeta)
    delta = float(delta)

    xs = np.empty((L,), dtype=np.float32)
    for t in range(L):
        x, v = rk4_step(x, v, omega, zeta, delta)
        xs[t] = x

    if noise_std and noise_std > 0:
        xs = xs + rng.normal(0.0, float(noise_std), size=xs.shape).astype(np.float32)

    return xs


def make_split(n, L, delta, omega_min, omega_max, zeta_min, zeta_max,
               x0_range, v0_range, noise_std, normalize, seed):
    rng = np.random.default_rng(seed)

    omegas = rng.uniform(omega_min, omega_max, size=(n,)).astype(np.float32)
    zetas = rng.uniform(zeta_min, zeta_max, size=(n,)).astype(np.float32)

    # Nuisance: initial conditions
    x0s = rng.uniform(x0_range[0], x0_range[1], size=(n,)).astype(np.float32)
    v0s = rng.uniform(v0_range[0], v0_range[1], size=(n,)).astype(np.float32)

    X = np.empty((n, L, 1), dtype=np.float32)
    Y = np.stack([omegas, zetas], axis=1).astype(np.float32)  # (N, 2)

    for i in range(n):
        xs = simulate_dho(
            omega=float(omegas[i]),
            zeta=float(zetas[i]),
            L=L,
            delta=delta,
            x0=float(x0s[i]),
            v0=float(v0s[i]),
            noise_std=noise_std,
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
    p.add_argument("--delta", type=float, required=True, help="Integration step size")

    # Parameter distribution
    p.add_argument("--omega_min", type=float, default=0.5)
    p.add_argument("--omega_max", type=float, default=3.0)
    p.add_argument("--zeta_min", type=float, default=0.02)
    p.add_argument("--zeta_max", type=float, default=0.4)

    # Dataset sizes
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)

    # Initial condition ranges
    p.add_argument("--x0_range", type=float, nargs=2, default=(-2.0, 2.0))
    p.add_argument("--v0_range", type=float, nargs=2, default=(-2.0, 2.0))

    # Observation noise (measurement noise)
    p.add_argument("--noise_std", type=float, default=0.0, help="Additive Gaussian noise on x(t)")

    # Optional per-sample normalization
    p.add_argument("--normalize", action="store_true", help="Normalize each sequence to zero-mean/unit-std")

    p.add_argument("--seed", type=int, default=37)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = make_split(
        n=args.n_train, L=args.L, delta=args.delta,
        omega_min=args.omega_min, omega_max=args.omega_max,
        zeta_min=args.zeta_min, zeta_max=args.zeta_max,
        x0_range=args.x0_range, v0_range=args.v0_range,
        noise_std=args.noise_std, normalize=args.normalize,
        seed=args.seed + 0,
    )
    x_val, y_val = make_split(
        n=args.n_val, L=args.L, delta=args.delta,
        omega_min=args.omega_min, omega_max=args.omega_max,
        zeta_min=args.zeta_min, zeta_max=args.zeta_max,
        x0_range=args.x0_range, v0_range=args.v0_range,
        noise_std=args.noise_std, normalize=args.normalize,
        seed=args.seed + 1,
    )
    x_test, y_test = make_split(
        n=args.n_test, L=args.L, delta=args.delta,
        omega_min=args.omega_min, omega_max=args.omega_max,
        zeta_min=args.zeta_min, zeta_max=args.zeta_max,
        x0_range=args.x0_range, v0_range=args.v0_range,
        noise_std=args.noise_std, normalize=args.normalize,
        seed=args.seed + 2,
    )

    meta = {
        "system": "damped_harmonic_oscillator",
        "obs": "x(t) only",
        "label": "(omega, zeta)",
        "L": int(args.L),
        "delta": float(args.delta),
        "omega_min": float(args.omega_min),
        "omega_max": float(args.omega_max),
        "zeta_min": float(args.zeta_min),
        "zeta_max": float(args.zeta_max),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "x0_range": list(map(float, args.x0_range)),
        "v0_range": list(map(float, args.v0_range)),
        "noise_std": float(args.noise_std),
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
