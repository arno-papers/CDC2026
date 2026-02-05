from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax.training import train_state

from .config import make_config
from .loss import LossConfig, targeted_spce_loss_and_metrics
from .policy import PolicyTransformer
from .sampling import sample_theta_full, sample_theta_nuisance


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves]))


def _tree_all_finite(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--ode-budget-traj", type=int, default=530_432)
    p.add_argument("--n-steps", type=int, default=14)
    p.add_argument("--n-substeps", type=int, default=500)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--l-contrastive", type=int, default=None)
    p.add_argument("--grad-batch", type=int, default=None)
    p.add_argument("--m-nuisance", type=int, default=None)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--plot-every", type=int, default=100)
    p.add_argument("--out-loss-plot", type=str, default="plot_loss_live_jax.png")
    p.add_argument("--out-params", type=str, default="checkpoints/policy_jax.msgpack")
    p.add_argument(
        "--microbatch",
        type=int,
        default=None,
        help="Split B episodes into microbatches to fit GPU memory.",
    )
    args = p.parse_args(argv)

    cfg = make_config(
        n_steps=args.n_steps,
        dt=args.dt,
        n_substeps=args.n_substeps,
        ode_budget_traj=args.ode_budget_traj,
        l_contrastive=args.l_contrastive,
        grad_batch=args.grad_batch,
        m_nuisance=args.m_nuisance,
        learning_rate=args.lr,
    )

    l = cfg.budget.l_contrastive
    b_total = cfg.budget.grad_batch
    m = cfg.budget.m_nuisance
    n_denom = l + 1

    b_micro = int(args.microbatch) if args.microbatch is not None else int(b_total)
    if b_micro <= 0:
        raise ValueError(f"microbatch must be > 0, got {b_micro}")
    if b_total % b_micro != 0:
        raise ValueError(
            f"microbatch must divide B. Got B={b_total} microbatch={b_micro}. "
            "Try e.g. 255, 300, 340, 425, 510, 850, 1020."
        )
    n_micro = b_total // b_micro

    print(
        "Config: "
        f"T={cfg.n_steps} dt={cfg.dt} substeps={cfg.n_substeps} "
        f"ODE_BUDGET_TRAJ={cfg.ode_budget_traj} -> L={l} B={b_total} M={m} "
        f"microbatch={b_micro} (n_micro={n_micro})"
    )
    print("JAX devices:", jax.devices())

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)

    model = PolicyTransformer()
    dummy = jnp.zeros((b_micro, cfg.n_steps, 2), dtype=jnp.float32)
    variables = model.init(init_key, dummy)
    params = variables["params"]

    tx = optax.adam(cfg.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    loss_cfg = LossConfig(n_steps=cfg.n_steps, dt=cfg.dt, n_substeps=cfg.n_substeps)

    @jax.jit
    def loss_and_grad(params_, theta_full_mb, theta_numer_mb, eps_mb):
        def loss_fn(p_):
            loss, metrics = targeted_spce_loss_and_metrics(
                state.apply_fn,
                p_,
                theta_full=theta_full_mb,
                theta_numer=theta_numer_mb,
                eps=eps_mb,
                cfg=loss_cfg,
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_)
        return loss, metrics, grads

    losses: list[float] = []

    for it in range(1, args.n_iters + 1):
        key, step_key = jax.random.split(key)

        kt, kn, ke = jax.random.split(step_key, 3)
        theta_full = sample_theta_full(
            kt, n_denom=n_denom, batch=b_total, bounds=cfg.prior
        )
        theta_numer = sample_theta_nuisance(kn, m=m, batch=b_total, bounds=cfg.prior)
        eps = jax.random.normal(ke, (cfg.n_steps, b_total), dtype=jnp.float32)

        weight = float(b_micro) / float(b_total)
        grads_acc = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        loss_acc = jnp.asarray(0.0, dtype=jnp.float32)
        metrics_acc = None

        for mi in range(n_micro):
            sl = slice(mi * b_micro, (mi + 1) * b_micro)
            loss_mb, metrics_mb, grads_mb = loss_and_grad(
                state.params,
                theta_full[:, :, sl],
                theta_numer[:, sl],
                eps[:, sl],
            )
            loss_acc = loss_acc + loss_mb * weight
            grads_acc = jax.tree_util.tree_map(
                lambda a, g: a + g * weight, grads_acc, grads_mb
            )

            if metrics_acc is None:
                metrics_acc = dict(metrics_mb)
                for k in [
                    "q_mean",
                    "q_frac_low",
                    "q_frac_high",
                    "log_num_mean",
                    "log_den_mean",
                    "den_w_true_mean",
                    "den_argmax0_frac",
                    "den_gap_mean",
                ]:
                    metrics_acc[k] = metrics_acc[k] * weight
            else:
                for k in [
                    "q_mean",
                    "q_frac_low",
                    "q_frac_high",
                    "log_num_mean",
                    "log_den_mean",
                    "den_w_true_mean",
                    "den_argmax0_frac",
                    "den_gap_mean",
                ]:
                    metrics_acc[k] = metrics_acc[k] + metrics_mb[k] * weight
                metrics_acc["q_min"] = jnp.minimum(
                    metrics_acc["q_min"], metrics_mb["q_min"]
                )
                metrics_acc["q_max"] = jnp.maximum(
                    metrics_acc["q_max"], metrics_mb["q_max"]
                )
                metrics_acc["den_gap_min"] = jnp.minimum(
                    metrics_acc["den_gap_min"], metrics_mb["den_gap_min"]
                )

        state = state.apply_gradients(grads=grads_acc)
        grad_norm = _tree_l2_norm(grads_acc)
        param_norm = _tree_l2_norm(state.params)
        grads_finite = _tree_all_finite(grads_acc)
        params_finite = _tree_all_finite(state.params)

        loss_f = float(loss_acc)
        losses.append(loss_f)

        if it == 1 or it % args.log_every == 0:
            m_ = metrics_acc if metrics_acc is not None else {}
            print(
                f"Iter [{it:4d}/{args.n_iters:4d}] "
                f"loss={loss_f:.8f} grad_norm={float(grad_norm):.6e} param_norm={float(param_norm):.6e} "
                f"q(mean/min/max)={float(m_.get('q_mean', jnp.nan)):.3f}/{float(m_.get('q_min', jnp.nan)):.3f}/{float(m_.get('q_max', jnp.nan)):.3f} "
                f"q(frac_low/high)={float(m_.get('q_frac_low', jnp.nan)):.3f}/{float(m_.get('q_frac_high', jnp.nan)):.3f} "
                f"den(w_true/argmax0)={float(m_.get('den_w_true_mean', jnp.nan)):.3f}/{float(m_.get('den_argmax0_frac', jnp.nan)):.3f} "
                f"den(gap_mean/min)={float(m_.get('den_gap_mean', jnp.nan)):.3f}/{float(m_.get('den_gap_min', jnp.nan)):.3f} "
                f"finite(params/grads)={bool(params_finite)}/{bool(grads_finite)}"
            )

        if args.plot_every > 0 and (it % args.plot_every == 0 or it == args.n_iters):
            out_plot = Path(args.out_loss_plot)
            out_plot.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 4), dpi=150)
            plt.plot(losses)
            plt.xlabel("Iteration")
            plt.ylabel("Targeted sPCE loss")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()

    if args.out_params:
        from flax import serialization

        out_path = Path(args.out_params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(serialization.to_bytes(state.params))
        print("Saved params to", str(out_path))


if __name__ == "__main__":
    main()
