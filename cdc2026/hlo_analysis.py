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
    p.add_argument("--ode-budget-traj", type=int, default=4144)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--n-substeps", type=int, default=500)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="artifacts/hlo")
    p.add_argument("--dialect", type=str, default="stablehlo")
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

    lowered = jax.jit(jax.value_and_grad(loss_fn)).lower(state.params)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Try the requested dialect first, then fall back.
    dialects = [args.dialect, "stablehlo", "hlo"]
    dumped = False
    for d in dialects:
        try:
            ir = lowered.compiler_ir(dialect=d)
        except Exception:
            continue
        (outdir / f"loss_and_grad.{d}.mlir").write_text(str(ir))
        print("Wrote", str(outdir / f"loss_and_grad.{d}.mlir"))
        dumped = True
        break

    if not dumped:
        raise SystemExit(
            "Could not dump compiler IR (dialect stablehlo/hlo unsupported in this JAX version)"
        )


if __name__ == "__main__":
    main()
