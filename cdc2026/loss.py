from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .dynamics import integrate_rk4


@dataclass(frozen=True)
class LossConfig:
    n_steps: int
    dt: float
    n_substeps: int


def _logmeanexp(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    return jsp.special.logsumexp(x, axis=axis) - jnp.log(
        jnp.asarray(x.shape[axis], dtype=x.dtype)
    )


def targeted_spce_loss_and_metrics(
    apply_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    *,
    theta_full: jnp.ndarray,
    theta_numer: jnp.ndarray,
    eps: jnp.ndarray,
    cfg: LossConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Targeted sPCE loss (JAX/Flax port of `common.jl`).

    Shapes:
      theta_full: (3, n_denom, B)
      theta_numer: (M, B)
      eps: (T, B)
    """
    n_steps = int(cfg.n_steps)
    dt = float(cfg.dt)
    n_substeps = int(cfg.n_substeps)

    if theta_full.ndim != 3 or theta_full.shape[0] != 3:
        raise ValueError(
            f"theta_full must have shape (3,n_denom,B), got {theta_full.shape}"
        )
    if eps.shape[0] != n_steps:
        raise ValueError(f"eps must have shape (T,B) with T={n_steps}, got {eps.shape}")

    n_denom = int(theta_full.shape[1])
    b = int(theta_full.shape[2])
    m = int(theta_numer.shape[0])

    # True parameters per episode (first contrastive sample).
    theta_t_true = jnp.transpose(theta_full[0:2, 0, :], (1, 0))  # (B,2)
    sigma_true = theta_full[2, 0, :]  # (B,)

    u0 = jnp.asarray([3.0, 0.25, 7.0], dtype=theta_full.dtype)
    u_true0 = jnp.broadcast_to(u0, (b, 3))
    input0 = jnp.zeros((b, n_steps, 2), dtype=theta_full.dtype)

    def rollout_step(carry, t_idx):
        u, inp = carry
        q = apply_fn({"params": params}, inp)  # (B,)
        u = integrate_rk4(u, theta_t_true, q, dt, n_substeps)
        obs = u[:, 0]
        y = obs + sigma_true * eps[t_idx]
        inp = inp.at[:, t_idx, 0].set(y)
        inp = inp.at[:, t_idx, 1].set(q)
        return (u, inp), (q, y)

    (_, _), (designs, observations) = jax.lax.scan(
        rollout_step,
        (u_true0, input0),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    # designs: (T,B), observations: (T,B)

    # ------------------------------
    # Denominator
    # ------------------------------
    theta_t_denom = jnp.transpose(theta_full[0:2, :, :], (1, 2, 0))  # (n_denom,B,2)
    sigma2_denom = theta_full[2, :, :] ** 2  # (n_denom,B)
    u_denom0 = jnp.broadcast_to(u0, (n_denom, b, 3))
    ll_denom0 = jnp.zeros((n_denom, b), dtype=theta_full.dtype)

    def denom_step(carry, t_idx):
        u_d, ll = carry
        # Broadcast q across contrastive parameter samples.
        q = jnp.broadcast_to(designs[t_idx], (n_denom, b))
        u_d = integrate_rk4(u_d, theta_t_denom, q, dt, n_substeps)
        pred = u_d[..., 0]  # (n_denom,B)
        actual = observations[t_idx][None, :]  # (1,B)
        resid = actual - pred
        ll = ll - 0.5 * (resid**2 / sigma2_denom + jnp.log(sigma2_denom))
        return (u_d, ll), None

    (_, ll_denom), _ = jax.lax.scan(
        denom_step,
        (u_denom0, ll_denom0),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    logsum_den = jsp.special.logsumexp(ll_denom, axis=0)
    log_den = logsum_den - jnp.log(jnp.asarray(n_denom, dtype=ll_denom.dtype))

    # ------------------------------
    # Numerator
    # ------------------------------
    sigma2_numer = theta_numer**2  # (M,B)
    u_num0 = jnp.broadcast_to(u0, (b, 3))
    ll_num0 = jnp.zeros((m, b), dtype=theta_full.dtype)

    def numer_step(carry, t_idx):
        u_n, ll = carry
        q = designs[t_idx]
        u_n = integrate_rk4(u_n, theta_t_true, q, dt, n_substeps)
        pred = u_n[:, 0]  # (B,)
        actual = observations[t_idx]  # (B,)
        resid2 = (actual - pred) ** 2  # (B,)
        ll = ll - 0.5 * (resid2[None, :] / sigma2_numer + jnp.log(sigma2_numer))
        return (u_n, ll), None

    (_, ll_numer), _ = jax.lax.scan(
        numer_step,
        (u_num0, ll_num0),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    logsum_num = jsp.special.logsumexp(ll_numer, axis=0)
    log_num = logsum_num - jnp.log(jnp.asarray(m, dtype=ll_numer.dtype))

    loss_per_episode = -(log_num - log_den)
    loss = jnp.mean(loss_per_episode)

    # ------------------------------
    # Debug metrics (device-side reductions)
    # ------------------------------
    q_mean = jnp.mean(designs)
    q_min = jnp.min(designs)
    q_max = jnp.max(designs)
    q_frac_low = jnp.mean(designs < jnp.asarray(1e-3, dtype=designs.dtype))
    q_frac_high = jnp.mean(designs > jnp.asarray(10.0 - 1e-3, dtype=designs.dtype))

    # Denominator collapse diagnostics.
    den_argmax0_frac = jnp.mean(jnp.argmax(ll_denom, axis=0) == 0)
    den_w_true = jnp.exp(ll_denom[0, :] - logsum_den)
    den_w_true_mean = jnp.mean(den_w_true)

    top2 = jax.lax.top_k(jnp.transpose(ll_denom, (1, 0)), k=2)[0]  # (B,2)
    den_gap = top2[:, 0] - top2[:, 1]
    den_gap_mean = jnp.mean(den_gap)
    den_gap_min = jnp.min(den_gap)

    metrics = {
        "q_mean": q_mean,
        "q_min": q_min,
        "q_max": q_max,
        "q_frac_low": q_frac_low,
        "q_frac_high": q_frac_high,
        "log_num_mean": jnp.mean(log_num),
        "log_den_mean": jnp.mean(log_den),
        "den_w_true_mean": den_w_true_mean,
        "den_argmax0_frac": den_argmax0_frac,
        "den_gap_mean": den_gap_mean,
        "den_gap_min": den_gap_min,
    }
    return loss, metrics


def targeted_spce_loss(
    apply_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    *,
    theta_full: jnp.ndarray,
    theta_numer: jnp.ndarray,
    eps: jnp.ndarray,
    cfg: LossConfig,
) -> jnp.ndarray:
    loss, _ = targeted_spce_loss_and_metrics(
        apply_fn,
        params,
        theta_full=theta_full,
        theta_numer=theta_numer,
        eps=eps,
        cfg=cfg,
    )
    return loss
