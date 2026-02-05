from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .config import make_config
from .dynamics import integrate_rk4
from .policy import PolicyTransformer
from .sampling import sample_theta_full


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trajs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=14)
    p.add_argument("--n-substeps", type=int, default=500)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--out", type=str, default="plot_trajectories_jax.png")
    p.add_argument("--params", type=str, default="")
    args = p.parse_args(argv)

    cfg = make_config(n_steps=args.n_steps, dt=args.dt, n_substeps=args.n_substeps)

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)

    model = PolicyTransformer()
    dummy = jnp.zeros((args.n_trajs, cfg.n_steps, 2), dtype=jnp.float32)
    params = model.init(init_key, dummy)["params"]

    if args.params:
        from flax import serialization

        params = serialization.from_bytes(params, Path(args.params).read_bytes())
        print("Loaded params from", args.params)

    # Sample one theta per trajectory.
    key, th_key = jax.random.split(key)
    theta_full = sample_theta_full(
        th_key, n_denom=1, batch=args.n_trajs, bounds=cfg.prior
    )
    theta_t = jnp.transpose(theta_full[0:2, 0, :], (1, 0))  # (B,2)

    u = jnp.broadcast_to(
        jnp.asarray([3.0, 0.25, 7.0], dtype=jnp.float32), (args.n_trajs, 3)
    )
    inp = jnp.zeros((args.n_trajs, cfg.n_steps, 2), dtype=jnp.float32)

    cs = []
    cx = []
    v = []
    q_hist = []

    for t in range(cfg.n_steps):
        q = model.apply({"params": params}, inp)  # (B,)
        u = integrate_rk4(u, theta_t, q, cfg.dt, cfg.n_substeps)
        obs = u[:, 0]

        cs.append(obs)
        cx.append(u[:, 1])
        v.append(u[:, 2])
        q_hist.append(q)

        inp = inp.at[:, t, 0].set(obs)
        inp = inp.at[:, t, 1].set(q)

    cs = jnp.stack(cs, axis=0)
    cx = jnp.stack(cx, axis=0)
    v = jnp.stack(v, axis=0)
    q_hist = jnp.stack(q_hist, axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150, sharex=True)
    axes = axes.ravel()

    axes[0].plot(cs)
    axes[0].set_title("C_s")
    axes[1].plot(cx)
    axes[1].set_title("C_x")
    axes[2].plot(v)
    axes[2].set_title("V")
    axes[3].plot(q_hist)
    axes[3].set_title("Q_in")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print("Saved", str(out))


if __name__ == "__main__":
    main()
