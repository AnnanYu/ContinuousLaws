#!/usr/bin/env python3
"""Generate train/val/test datasets for parameter inference on the *forced Duffing oscillator*.

Forced Duffing (non-autonomous):
    x¨ + delta * x˙ + alpha * x + beta * x^3 = gamma * cos(omega * t)

Convert to first order:
    x' = v
    v' = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t)

Integration:
- Uses fixed-step RK4 with step size dt = delta_t (flag: --delta).
- Time enters explicitly through the forcing term.

About "before Lyapunov time":
- Chaotic divergence depends on the parameter regime.
- This script uses a conservative *short observation horizon* by default (T = L*dt ≈ 10),
  which is typically well before severe divergence in the common chaotic Duffing settings.

Dataset format (same as your other scripts):
    x_train, y_train, x_val, y_val, x_test, y_test, meta_json
where x_* has shape (N, L, 1) and y_* has shape (N, 1) (gamma by default).

Default (classic) parameter regime:
    alpha = -1, beta = 1, damp(delta) = 0.2, omega = 1.2,
    gamma in roughly [0.2, 0.4] (often chaotic/mixed).

"""


import argparse
import json
from pathlib import Path
import numpy as np


def rk4_step(x, v, t, gamma, damp, alpha, beta, omega, dt):
    """One RK4 step for forced Duffing."""
    def f(x_, v_, t_):
        dx = v_
        dv = -damp * v_ - alpha * x_ - beta * (x_ ** 3) + gamma * np.cos(omega * t_)
        return dx, dv

    k1x, k1v = f(x, v, t)
    k2x, k2v = f(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t + 0.5 * dt)
    k3x, k3v = f(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t + 0.5 * dt)
    k4x, k4v = f(x + dt * k3x, v + dt * k3v, t + dt)

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_next, v_next


def simulate_duffing(gamma, L, dt, x0, v0, damp, alpha, beta, omega,
                     burn_in=0, extra=0, noise_std=0.0, rng=None):
    """Simulate forced Duffing and return an x-sequence of length L.

    If extra>0, simulate L+extra steps and return a random contiguous window of length L.
    burn_in steps are discarded before recording (to reach attractor).
    """
    rng = np.random.default_rng() if rng is None else rng

    x = float(x0)
    v = float(v0)
    t = 0.0

    gamma = float(gamma)
    damp = float(damp)
    alpha = float(alpha)
    beta = float(beta)
    omega = float(omega)
    dt = float(dt)

    # Burn-in
    for _ in range(int(burn_in)):
        x, v = rk4_step(x, v, t, gamma, damp, alpha, beta, omega, dt)
        t += dt

    total = int(L + max(0, int(extra)))
    xs = np.empty((total,), dtype=np.float32)

    for i in range(total):
        x, v = rk4_step(x, v, t, gamma, damp, alpha, beta, omega, dt)
        t += dt
        xs[i] = x

    if extra and extra > 0:
        start = int(rng.integers(0, int(extra) + 1))
        xs = xs[start:start + L]
    else:
        xs = xs[:L]

    if noise_std and noise_std > 0:
        xs = xs + rng.normal(0.0, float(noise_std), size=xs.shape).astype(np.float32)

    return xs


def make_split(n, L, dt, gamma_min, gamma_max, damp, alpha, beta, omega,
               x0_range, v0_range, burn_in, extra, noise_std, normalize, seed):
    rng = np.random.default_rng(seed)

    gammas = rng.uniform(gamma_min, gamma_max, size=(n,)).astype(np.float32)
    x0s = rng.uniform(x0_range[0], x0_range[1], size=(n,)).astype(np.float32)
    v0s = rng.uniform(v0_range[0], v0_range[1], size=(n,)).astype(np.float32)

    X = np.empty((n, L, 1), dtype=np.float32)
    Y = gammas.reshape(n, 1).astype(np.float32)

    for i in range(n):
        xs = simulate_duffing(
            gamma=float(gammas[i]),
            L=L,
            dt=dt,
            x0=float(x0s[i]),
            v0=float(v0s[i]),
            damp=damp,
            alpha=alpha,
            beta=beta,
            omega=omega,
            burn_in=burn_in,
            extra=extra,
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
    p.add_argument("--delta", type=float, required=True, help="Time step size dt")

    # Predict ONLY gamma
    p.add_argument("--gamma_min", type=float, default=0.20)
    p.add_argument("--gamma_max", type=float, default=0.40)

    # Fixed Duffing parameters
    p.add_argument("--damp", type=float, default=0.2, help="delta in equation (damping coefficient)")
    p.add_argument("--alpha", type=float, default=-1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=1.2, help="forcing frequency")

    # Dataset sizes
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)

    # Initial conditions
    p.add_argument("--x0_range", type=float, nargs=2, default=(-1.0, 1.0))
    p.add_argument("--v0_range", type=float, nargs=2, default=(-1.0, 1.0))

    # Burn-in and random windowing
    p.add_argument("--burn_in", type=int, default=5000,
                   help="Burn-in steps (discard) to reach attractor / steady regime")
    p.add_argument("--extra", type=int, default=1000,
                   help="After burn-in, simulate L+extra and take a random length-L window")

    # Measurement noise
    p.add_argument("--noise_std", type=float, default=0.0, help="Additive Gaussian noise on x(t)")

    # Optional per-sample normalization
    p.add_argument("--normalize", action="store_true", help="Normalize each sequence to zero-mean/unit-std")

    p.add_argument("--seed", type=int, default=37)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = make_split(
        n=args.n_train, L=args.L, dt=args.delta,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        damp=args.damp, alpha=args.alpha, beta=args.beta, omega=args.omega,
        x0_range=args.x0_range, v0_range=args.v0_range,
        burn_in=args.burn_in, extra=args.extra,
        noise_std=args.noise_std, normalize=args.normalize,
        seed=args.seed + 0,
    )
    x_val, y_val = make_split(
        n=args.n_val, L=args.L, dt=args.delta,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        damp=args.damp, alpha=args.alpha, beta=args.beta, omega=args.omega,
        x0_range=args.x0_range, v0_range=args.v0_range,
        burn_in=args.burn_in, extra=args.extra,
        noise_std=args.noise_std, normalize=args.normalize,
        seed=args.seed + 1,
    )
    x_test, y_test = make_split(
        n=args.n_test, L=args.L, dt=args.delta,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        damp=args.damp, alpha=args.alpha, beta=args.beta, omega=args.omega,
        x0_range=args.x0_range, v0_range=args.v0_range,
        burn_in=args.burn_in, extra=args.extra,
        noise_std=args.noise_std, normalize=args.normalize,
        seed=args.seed + 2,
    )

    meta = {
        "system": "forced_duffing",
        "obs": "x(t) only",
        "label": "gamma (forcing amplitude)",
        "integrator": "rk4_fixed_step",
        "L": int(args.L),
        "dt": float(args.delta),
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "damp": float(args.damp),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "omega": float(args.omega),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "x0_range": list(map(float, args.x0_range)),
        "v0_range": list(map(float, args.v0_range)),
        "burn_in": int(args.burn_in),
        "extra": int(args.extra),
        "noise_std": float(args.noise_std),
        "normalize": bool(args.normalize),
        "seed": int(args.seed),
        "horizon_T": float(args.L * args.delta),
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
    print(f"Horizon T = L*dt = {args.L}*{args.delta} = {args.L*args.delta:.4f}")


if __name__ == "__main__":
    main()
