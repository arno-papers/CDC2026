from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import make_config
from .loss import LossConfig, targeted_spce_loss
from .policy import PolicyTransformer
from .sampling import sample_theta_full, sample_theta_nuisance


def _tree_l2_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves])))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ode-budget-traj", type=int, default=4144)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--n-substeps", type=int, default=500)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--assert-grad-nonzero", action="store_true")
    args = p.parse_args(argv)

    cfg = make_config(
        n_steps=args.n_steps,
        dt=args.dt,
        n_substeps=args.n_substeps,
        ode_budget_traj=args.ode_budget_traj,
        learning_rate=args.lr,
    )

    l = cfg.budget.l_contrastive
    b = cfg.budget.grad_batch
    m = cfg.budget.m_nuisance
    n_denom = l + 1

    print(
        "Debug config: "
        f"T={cfg.n_steps} dt={cfg.dt} substeps={cfg.n_substeps} "
        f"ODE_BUDGET_TRAJ={cfg.ode_budget_traj} -> L={l} B={b} M={m}"
    )
    print("JAX devices:", jax.devices())

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)

    model = PolicyTransformer()
    dummy = jnp.zeros((b, cfg.n_steps, 2), dtype=jnp.float32)
    params = model.init(init_key, dummy)["params"]

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(cfg.learning_rate)
    )
    loss_cfg = LossConfig(n_steps=cfg.n_steps, dt=cfg.dt, n_substeps=cfg.n_substeps)

    kt, kn, ke = jax.random.split(key, 3)
    theta_full = sample_theta_full(kt, n_denom=n_denom, batch=b, bounds=cfg.prior)
    theta_numer = sample_theta_nuisance(kn, m=m, batch=b, bounds=cfg.prior)
    eps = jax.random.normal(ke, (cfg.n_steps, b), dtype=jnp.float32)

    def loss_fn(p_):
        return targeted_spce_loss(
            state.apply_fn,
            p_,
            theta_full=theta_full,
            theta_numer=theta_numer,
            eps=eps,
            cfg=loss_cfg,
        )

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grad_norm = _tree_l2_norm(grads)

    print(f"loss={float(loss):.8f}")
    print(f"grad_norm={grad_norm:.6e}")

    if args.assert_grad_nonzero and grad_norm == 0.0:
        raise SystemExit("Gradient norm is exactly 0.0")


if __name__ == "__main__":
    main()
