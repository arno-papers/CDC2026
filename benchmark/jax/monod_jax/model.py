"""Monod bioreactor: dynamics, RK4 integration, prior sampling, budget allocation."""

import jax
import jax.numpy as jnp

# ============================================================================
#  Constants (match Julia model.jl exactly)
# ============================================================================

N_STEPS = 14
DT = 1.0
N_SUBSTEPS = 50
ACTION_HI = 10.0

# Prior bounds (uniform)
MU_MAX_LO, MU_MAX_HI = 0.3, 0.5
K_S_LO, K_S_HI = 0.3, 0.6
SIGMA_LO, SIGMA_HI = 0.05, 0.15
CX0_LO, CX0_HI = 0.10, 0.50

# Initial conditions
CS0 = 3.0
V0 = 7.0

# ============================================================================
#  Monod bioreactor dynamics
# ============================================================================


def dynamics(u, theta_dyn, Q_in):
    """Monod bioreactor ODE RHS.

    Supports both scalar states `(3,)` and batched states `(..., 3)`.
    """
    C_s, C_x, V = u[..., 0], u[..., 1], u[..., 2]
    mu_max, K_s = theta_dyn[..., 0], theta_dyn[..., 1]
    mu = mu_max * C_s / (K_s + C_s)
    sigma = mu / 0.777
    dC_s = -sigma * C_x + (Q_in / V) * (50.0 - C_s)
    dC_x = mu * C_x - (Q_in / V) * C_x
    dV = Q_in + jnp.zeros_like(V)
    return jnp.stack([dC_s, dC_x, dV], axis=-1)


# ============================================================================
#  RK4 integrator
# ============================================================================


def rk4_step(u, theta_dyn, Q_in, dt):
    """Single RK4 step."""
    k1 = dynamics(u, theta_dyn, Q_in)
    k2 = dynamics(u + 0.5 * dt * k1, theta_dyn, Q_in)
    k3 = dynamics(u + 0.5 * dt * k2, theta_dyn, Q_in)
    k4 = dynamics(u + dt * k3, theta_dyn, Q_in)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@jax.checkpoint
def integrate_unrolled(u, theta_dyn, Q_in):
    """Integrate one macro time step (DT) with unrolled RK4 substeps."""
    dt_sub = DT / N_SUBSTEPS
    for _ in range(N_SUBSTEPS):
        u = rk4_step(u, theta_dyn, Q_in, dt_sub)
    return u


@jax.checkpoint
def integrate_scan(u, theta_dyn, Q_in):
    """Integrate one macro time step with RK4 substeps in a `lax.scan`."""
    dt_sub = DT / N_SUBSTEPS

    def body_fn(state, _):
        return rk4_step(state, theta_dyn, Q_in, dt_sub), None

    u, _ = jax.lax.scan(body_fn, u, None, length=N_SUBSTEPS)
    return u


@jax.checkpoint
def integrate_fori(u, theta_dyn, Q_in):
    """Integrate one macro time step with RK4 substeps in a `lax.fori_loop`."""
    dt_sub = DT / N_SUBSTEPS

    def body_fn(_, state):
        return rk4_step(state, theta_dyn, Q_in, dt_sub)

    return jax.lax.fori_loop(0, N_SUBSTEPS, body_fn, u)


def get_integrator(substep_loop):
    """Select the RK4 substep loop implementation."""
    if substep_loop == "unrolled":
        return integrate_unrolled
    if substep_loop == "scan":
        return integrate_scan
    if substep_loop == "fori":
        return integrate_fori
    raise ValueError(f"Unsupported substep_loop={substep_loop!r}")


def integrate(u, theta_dyn, Q_in):
    """Default integrator used by the baseline path."""
    return integrate_unrolled(u, theta_dyn, Q_in)


# ============================================================================
#  Prior sampling
# ============================================================================


def sample_theta_full(key, n):
    """Sample n full parameter sets from uniform prior.

    Returns: (n, 4) with columns [mu_max, K_s, sigma, Cx0].
    """
    u = jax.random.uniform(key, (n, 4))
    lo = jnp.array([MU_MAX_LO, K_S_LO, SIGMA_LO, CX0_LO])
    hi = jnp.array([MU_MAX_HI, K_S_HI, SIGMA_HI, CX0_HI])
    return lo + (hi - lo) * u


def sample_theta_obs(key, n):
    """Sample n observation parameter sets [sigma, Cx0] from uniform prior."""
    u = jax.random.uniform(key, (n, 2))
    lo = jnp.array([SIGMA_LO, CX0_LO])
    hi = jnp.array([SIGMA_HI, CX0_HI])
    return lo + (hi - lo) * u


# ============================================================================
#  Budget allocation (port of src/utils.jl)
# ============================================================================


def allocate_budget(C, lambda_L=1.0, lambda_M=1.0, B_multiplier=1):
    """Jointly optimize L (contrastive), M (nuisance), B (batch) for budget C.

    Minimizes 1/(B*B_multiplier) + lambda_L/(L+1)^2 + lambda_M/M^2
    subject to B = C // (L + 2 + M).
    """
    L_seed = max(1, round((2 * lambda_L * C * B_multiplier) ** (1 / 3)) - 1)
    M_seed = max(1, round((2 * lambda_M * C * B_multiplier) ** (1 / 3)))
    w = max(20, L_seed // 3)
    best_L, best_M, best_B, best_obj = (
        L_seed, M_seed, C // (L_seed + 2 + M_seed), float("inf"),
    )
    for L in range(max(1, L_seed - w), L_seed + w + 1):
        for M in range(max(1, M_seed - w), M_seed + w + 1):
            B = C // (L + 2 + M)
            if B < 1:
                break
            obj = (
                1.0 / (B * B_multiplier)
                + lambda_L / (L + 1) ** 2
                + lambda_M / M ** 2
            )
            if obj < best_obj:
                best_obj, best_L, best_M, best_B = obj, L, M, B
    return best_L, best_M, best_B
