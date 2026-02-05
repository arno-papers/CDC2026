from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .config import make_config
from .loss import LossConfig, targeted_spce_loss
from .policy import PolicyTransformer
from .sampling import sample_theta_full, sample_theta_nuisance


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iters", type=int, default=3)
    p.add_argument("--ode-budget-traj", type=int, default=4144)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--n-substeps", type=int, default=500)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--logdir", type=str, default="artifacts/jax_trace")
    p.add_argument(
        "--create-perfetto-link",
        action="store_true",
        help="Print a ui.perfetto.dev link after tracing (can be slow).",
    )
    args = p.parse_args(argv)

    cfg = make_config(
        n_steps=args.n_steps,
        dt=args.dt,
        n_substeps=args.n_substeps,
        ode_budget_traj=args.ode_budget_traj,
    )
    l = cfg.budget.l_contrastive
    b = cfg.budget.grad_batch
    m = cfg.budget.m_nuisance
    n_denom = l + 1
    loss_cfg = LossConfig(n_steps=cfg.n_steps, dt=cfg.dt, n_substeps=cfg.n_substeps)

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)

    model = PolicyTransformer()
    dummy = jnp.zeros((b, cfg.n_steps, 2), dtype=jnp.float32)
    params = model.init(init_key, dummy)["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(cfg.learning_rate)
    )

    @jax.jit
    def step(state, key):
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
        state = state.apply_gradients(grads=grads)
        return state, loss

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    print("Tracing to", str(logdir))

    try:
        from jax import profiler
    except Exception as e:
        raise SystemExit(f"jax.profiler unavailable: {e}")

    # Warmup compile.
    key, warm = jax.random.split(key)
    state, _ = step(state, warm)
    jax.block_until_ready(state.params)

    profiler.start_trace(
        str(logdir), create_perfetto_link=bool(args.create_perfetto_link)
    )
    try:
        for _ in range(args.n_iters):
            key, sk = jax.random.split(key)
            state, loss = step(state, sk)
            jax.block_until_ready(loss)
    finally:
        profiler.stop_trace()

    print("Done")


if __name__ == "__main__":
    main()
