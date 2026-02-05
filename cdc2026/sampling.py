from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import PriorBounds


def sample_theta_full(
    key: jax.Array,
    *,
    n_denom: int,
    batch: int,
    bounds: PriorBounds,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Uniform prior over (mu_max, K_s, sigma).

    Returns shape (3, n_denom, batch).
    """
    u = jax.random.uniform(key, (3, int(n_denom), int(batch)), dtype=dtype)
    mu_max = bounds.mu_max_lo + (bounds.mu_max_hi - bounds.mu_max_lo) * u[0]
    k_s = bounds.k_s_lo + (bounds.k_s_hi - bounds.k_s_lo) * u[1]
    sigma = bounds.sigma_lo + (bounds.sigma_hi - bounds.sigma_lo) * u[2]
    return jnp.stack([mu_max, k_s, sigma], axis=0)


def sample_theta_nuisance(
    key: jax.Array,
    *,
    m: int,
    batch: int,
    bounds: PriorBounds,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Uniform prior over nuisance sigma.

    Returns shape (m, batch).
    """
    u = jax.random.uniform(key, (int(m), int(batch)), dtype=dtype)
    return bounds.sigma_lo + (bounds.sigma_hi - bounds.sigma_lo) * u
