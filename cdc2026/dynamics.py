from __future__ import annotations

import jax
import jax.numpy as jnp


def bioreactor_dynamics(
    u: jnp.ndarray, theta_t: jnp.ndarray, q_in: jnp.ndarray
) -> jnp.ndarray:
    """Bioreactor dynamics matching `common.jl`.

    u: (..., 3) for (C_s, C_x, V)
    theta_t: (..., 2) for (mu_max, K_s)
    q_in: (...) broadcastable to u[..., 2]
    """
    c_s = u[..., 0]
    c_x = u[..., 1]
    v = u[..., 2]

    mu_max = theta_t[..., 0]
    k_s = theta_t[..., 1]

    mu = mu_max * c_s / (k_s + c_s)
    sigma = mu / jnp.asarray(0.777, dtype=u.dtype)

    du1 = -sigma * c_x + (q_in / v) * (jnp.asarray(50.0, dtype=u.dtype) - c_s)
    du2 = mu * c_x - (q_in / v) * c_x
    du3 = jnp.asarray(0.0, dtype=u.dtype) * v + q_in

    return jnp.stack([du1, du2, du3], axis=-1)


def bioreactor_jac_u_q(
    u: jnp.ndarray, theta_t: jnp.ndarray, q_in: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Jacobian of dynamics w.r.t. state u and scalar input q.

    Returns:
      A: (..., 3, 3) where A[i,j] = d f_i / d u_j
      b: (..., 3) where b[i] = d f_i / d q
    """
    c_s = u[..., 0]
    c_x = u[..., 1]
    v = u[..., 2]

    mu_max = theta_t[..., 0]
    k_s = theta_t[..., 1]

    denom = k_s + c_s
    mu = mu_max * c_s / denom
    dmu_dcs = mu_max * k_s / (denom * denom)

    sigma = mu / jnp.asarray(0.777, dtype=u.dtype)
    dsigma_dcs = dmu_dcs / jnp.asarray(0.777, dtype=u.dtype)

    inv_v = jnp.asarray(1.0, dtype=u.dtype) / v
    inv_v2 = inv_v * inv_v
    q_over_v = q_in * inv_v

    a11 = -(dsigma_dcs * c_x) - q_over_v
    a12 = -sigma
    a13 = -(q_in * (jnp.asarray(50.0, dtype=u.dtype) - c_s)) * inv_v2

    a21 = c_x * dmu_dcs
    a22 = mu - q_over_v
    a23 = (q_in * c_x) * inv_v2

    z = jnp.zeros_like(a11)
    a31 = z
    a32 = z
    a33 = z

    a = jnp.stack(
        [
            jnp.stack([a11, a12, a13], axis=-1),
            jnp.stack([a21, a22, a23], axis=-1),
            jnp.stack([a31, a32, a33], axis=-1),
        ],
        axis=-2,
    )

    b1 = inv_v * (jnp.asarray(50.0, dtype=u.dtype) - c_s)
    b2 = -inv_v * c_x
    b3 = jnp.ones_like(b1)
    b = jnp.stack([b1, b2, b3], axis=-1)
    return a, b


def rk4_step(
    u: jnp.ndarray, theta_t: jnp.ndarray, q_in: jnp.ndarray, dt: float
) -> jnp.ndarray:
    dt = jnp.asarray(dt, dtype=u.dtype)
    k1 = bioreactor_dynamics(u, theta_t, q_in)
    k2 = bioreactor_dynamics(u + 0.5 * dt * k1, theta_t, q_in)
    k3 = bioreactor_dynamics(u + 0.5 * dt * k2, theta_t, q_in)
    k4 = bioreactor_dynamics(u + dt * k3, theta_t, q_in)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_step_with_sens(
    u: jnp.ndarray,
    s: jnp.ndarray,
    theta_t: jnp.ndarray,
    q_in: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One RK4 step for primal state and sensitivities.

    s has shape (..., 3, 4) representing d u / d p where
      p = [u0_0, u0_1, u0_2, q]
    """
    # Stage 1
    f1 = bioreactor_dynamics(u, theta_t, q_in)
    a1, b1 = bioreactor_jac_u_q(u, theta_t, q_in)
    fp1 = jnp.concatenate(
        [jnp.zeros((*b1.shape[:-1], 3, 3), dtype=u.dtype), b1[..., :, None]], axis=-1
    )
    g1 = a1 @ s + fp1

    # Stage 2
    u2 = u + 0.5 * dt * f1
    s2 = s + 0.5 * dt * g1
    f2 = bioreactor_dynamics(u2, theta_t, q_in)
    a2, b2 = bioreactor_jac_u_q(u2, theta_t, q_in)
    fp2 = jnp.concatenate(
        [jnp.zeros((*b2.shape[:-1], 3, 3), dtype=u.dtype), b2[..., :, None]], axis=-1
    )
    g2 = a2 @ s2 + fp2

    # Stage 3
    u3 = u + 0.5 * dt * f2
    s3 = s + 0.5 * dt * g2
    f3 = bioreactor_dynamics(u3, theta_t, q_in)
    a3, b3 = bioreactor_jac_u_q(u3, theta_t, q_in)
    fp3 = jnp.concatenate(
        [jnp.zeros((*b3.shape[:-1], 3, 3), dtype=u.dtype), b3[..., :, None]], axis=-1
    )
    g3 = a3 @ s3 + fp3

    # Stage 4
    u4 = u + dt * f3
    s4 = s + dt * g3
    f4 = bioreactor_dynamics(u4, theta_t, q_in)
    a4, b4 = bioreactor_jac_u_q(u4, theta_t, q_in)
    fp4 = jnp.concatenate(
        [jnp.zeros((*b4.shape[:-1], 3, 3), dtype=u.dtype), b4[..., :, None]], axis=-1
    )
    g4 = a4 @ s4 + fp4

    u_next = u + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
    s_next = s + (dt / 6.0) * (g1 + 2.0 * g2 + 2.0 * g3 + g4)
    return u_next, s_next


def _integrate_rk4_primal(
    u: jnp.ndarray, theta_t: jnp.ndarray, q_in: jnp.ndarray, dt: float, n_substeps: int
) -> jnp.ndarray:
    n_substeps = int(n_substeps)
    dt_sub = jnp.asarray(dt, dtype=u.dtype) / jnp.asarray(n_substeps, dtype=u.dtype)

    def body(_i, u_):
        return rk4_step(u_, theta_t, q_in, dt_sub)

    return jax.lax.fori_loop(0, n_substeps, body, u)


def _integrate_rk4_with_sens(
    u: jnp.ndarray, theta_t: jnp.ndarray, q_in: jnp.ndarray, dt: float, n_substeps: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate u and sensitivities S = d u / d [u0,q]."""
    n_substeps = int(n_substeps)
    dt_sub = jnp.asarray(dt, dtype=u.dtype) / jnp.asarray(n_substeps, dtype=u.dtype)

    # S init: identity for u0, zeros for q column.
    i3 = jnp.eye(3, dtype=u.dtype)
    z3 = jnp.zeros((3, 1), dtype=u.dtype)
    s0 = jnp.concatenate([i3, z3], axis=-1)  # (3,4)
    s = jnp.broadcast_to(s0, (*u.shape[:-1], 3, 4))

    def body(_i, carry):
        u_, s_ = carry
        u1, s1 = _rk4_step_with_sens(u_, s_, theta_t, q_in, dt_sub)
        return (u1, s1)

    u_out, s_out = jax.lax.fori_loop(0, n_substeps, body, (u, s))
    return u_out, s_out


def integrate_rk4(
    u: jnp.ndarray, theta_t: jnp.ndarray, q_in: jnp.ndarray, dt: float, n_substeps: int
) -> jnp.ndarray:
    return _integrate_rk4_primal(u, theta_t, q_in, dt, n_substeps)


# Treat dt and n_substeps as static (non-differentiable) args.
integrate_rk4 = jax.custom_vjp(integrate_rk4, nondiff_argnums=(3, 4))


def _integrate_rk4_fwd(u, theta_t, q_in, dt, n_substeps):
    u_out, s_out = _integrate_rk4_with_sens(u, theta_t, q_in, dt, n_substeps)
    # Save Jacobians wrt u and q; theta_t treated as constant for our use.
    j_u = s_out[..., :, 0:3]  # (...,3,3)
    j_q = s_out[..., :, 3]  # (...,3)
    return u_out, (j_u, j_q)


def _integrate_rk4_bwd(dt, n_substeps, res, g_u_out):
    j_u, j_q = res
    g_u_out = jnp.asarray(g_u_out)
    # VJP: g_u = J_u^T @ g_u_out
    g_u = jnp.einsum("...ij,...i->...j", j_u, g_u_out)
    # g_q = J_q^T @ g_u_out (q scalar)
    g_q = jnp.einsum("...i,...i->...", j_q, g_u_out)
    g_theta = jnp.zeros((*g_u_out.shape[:-1], 2), dtype=g_u_out.dtype)
    return (g_u, g_theta, g_q)


integrate_rk4.defvjp(_integrate_rk4_fwd, _integrate_rk4_bwd)
