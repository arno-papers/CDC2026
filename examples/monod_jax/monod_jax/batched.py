"""Batch-first targeted sPCE loss.

Keeps the outer Monte Carlo batch `B` fully batched, while chunking the
hypothesis axes `(L+1)` and `M` to control memory. This matches the paper's
estimator structure without materializing the full `(B, H, ...)` recurrence
state that OOMs on a 3080.
"""

import functools
import math

import jax
import jax.numpy as jnp

from .model import (
    N_STEPS,
    CS0,
    V0,
    get_integrator,
    sample_theta_full,
    sample_theta_obs,
)
from .policy import PE, ACTION_HI, D_MODEL, N_HEADS, HEAD_DIM


# ============================================================================
#  Batched policy forward: (B, N_STEPS, 2) -> (B,)
# ============================================================================


def _rms_norm_batched(x, scale):
    """RMSNorm over last dim. x: (B, seq, D)."""
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6) * scale


def _mha_batched(x, p):
    """Multi-head self-attention. x: (B, S, D)."""
    B, S, _ = x.shape
    q = (x @ p["q"].T).reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = (x @ p["k"].T).reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = (x @ p["v"].T).reshape(B, S, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    w = jax.nn.softmax(
        q @ jnp.swapaxes(k, -2, -1) / jnp.sqrt(jnp.float32(HEAD_DIM)), axis=-1
    )
    o = (w @ v).transpose(0, 2, 1, 3).reshape(B, S, D_MODEL)
    return o @ p["out_w"].T + p["out_b"]


def policy_forward_batched(params, buf):
    """Batched transformer policy. buf: (B, N_STEPS, 2) -> (B,) actions."""
    x = buf @ params["input_proj"]["weight"].T + params["input_proj"]["bias"] + PE
    x = x + _mha_batched(_rms_norm_batched(x, params["rms1"]["scale"]), params["mha"])
    h = _rms_norm_batched(x, params["rms2"]["scale"])
    ff = jax.nn.gelu(h @ params["ff"]["w1"].T + params["ff"]["b1"])
    x = x + ff @ params["ff"]["w2"].T + params["ff"]["b2"]
    logit = x[:, -1, :] @ params["out"]["w"].T + params["out"]["b"]
    return ACTION_HI * jax.nn.sigmoid(logit[:, 0])


# ============================================================================
#  Batched rollout over the true episode batch
# ============================================================================


def _rollout_batched(
    params, theta_dyn_true, sigma_true, Cx0_true, epsilon, substep_loop="unrolled"
):
    """Generate one adaptive trajectory per episode.

    Returns step-major arrays `(N_STEPS, B)` to match `lax.scan` directly and
    avoid transposing observations/designs before likelihood evaluation.
    """
    integrate_fn = get_integrator(substep_loop)
    B = theta_dyn_true.shape[0]
    dtype = theta_dyn_true.dtype
    u0 = jnp.stack([
        jnp.full((B,), CS0, dtype=dtype),
        Cx0_true,
        jnp.full((B,), V0, dtype=dtype),
    ], axis=-1)
    input_buffer0 = jnp.zeros((B, N_STEPS, 2), dtype=dtype)

    def adaptive_step(carry, step_idx):
        u, input_buffer = carry
        action = policy_forward_batched(params, input_buffer)
        u = integrate_fn(u, theta_dyn_true, action)
        y_obs = u[..., 0] + sigma_true * epsilon[:, step_idx]
        input_buffer = input_buffer.at[:, step_idx, 0].set(y_obs)
        input_buffer = input_buffer.at[:, step_idx, 1].set(action)
        return (u, input_buffer), (y_obs, action)

    (_, _), (observations_t, designs_t) = jax.lax.scan(
        adaptive_step, (u0, input_buffer0), jnp.arange(N_STEPS)
    )
    return observations_t, designs_t


# ============================================================================
#  Chunked log-likelihood reductions over hypothesis axes
# ============================================================================


def _scan_loglik_chunk(
    observations_t,
    designs_t,
    theta_dyn_chunk,
    theta_obs_chunk,
    substep_loop="unrolled",
):
    """Log-likelihood for one hypothesis chunk across all B episodes.

    Args:
        observations_t: (N_STEPS, B)
        designs_t: (N_STEPS, B)
        theta_dyn_chunk: (B, Hc, 2)
        theta_obs_chunk: (B, Hc, 2)
    Returns:
        (B, Hc) array of per-hypothesis log-likelihoods.
    """
    integrate_fn = get_integrator(substep_loop)
    B, Hc = theta_obs_chunk.shape[:2]
    dtype = theta_obs_chunk.dtype
    u0 = jnp.stack([
        jnp.full((B, Hc), CS0, dtype=dtype),
        theta_obs_chunk[..., 1],
        jnp.full((B, Hc), V0, dtype=dtype),
    ], axis=-1).reshape(B * Hc, 3)
    theta_dyn_flat = theta_dyn_chunk.reshape(B * Hc, 2)
    sigma_sq = theta_obs_chunk[..., 0] ** 2
    ll0 = jnp.zeros((B, Hc), dtype=dtype)

    def step_fn(carry, step_inputs):
        u_flat, ll = carry
        obs_t, design_t = step_inputs
        design = jnp.broadcast_to(design_t[:, None], sigma_sq.shape).reshape(B * Hc)
        u_flat = integrate_fn(u_flat, theta_dyn_flat, design)
        residual = obs_t[:, None] - u_flat[:, 0].reshape(B, Hc)
        ll = ll - 0.5 * (residual**2 / sigma_sq + jnp.log(sigma_sq))
        return (u_flat, ll), None

    (_, ll_total), _ = jax.lax.scan(
        step_fn, (u0, ll0), (observations_t, designs_t)
    )
    return ll_total


def _chunked_logmeanexp(
    observations_t,
    designs_t,
    theta_dyn,
    theta_obs,
    n_valid,
    chunk_size,
    substep_loop="unrolled",
):
    """Compute `log(mean(exp(ll)))` over a hypothesis axis without materializing it."""
    H = theta_dyn.shape[1]
    dtype = theta_dyn.dtype
    if chunk_size <= 0 or chunk_size >= H:
        ll = _scan_loglik_chunk(
            observations_t, designs_t, theta_dyn, theta_obs, substep_loop
        )
        return jax.scipy.special.logsumexp(ll, axis=1) - jnp.log(
            jnp.asarray(n_valid, dtype=dtype)
        )

    n_chunks = math.ceil(H / chunk_size)
    padded_H = n_chunks * chunk_size
    if padded_H != H:
        pad = padded_H - H
        pad_dyn = jnp.repeat(theta_dyn[:, :1, :], pad, axis=1)
        pad_obs = jnp.repeat(theta_obs[:, :1, :], pad, axis=1)
        theta_dyn = jnp.concatenate([theta_dyn, pad_dyn], axis=1)
        theta_obs = jnp.concatenate([theta_obs, pad_obs], axis=1)

    B = theta_dyn.shape[0]
    theta_dyn_chunks = jnp.swapaxes(
        theta_dyn.reshape(B, n_chunks, chunk_size, 2), 0, 1
    )
    theta_obs_chunks = jnp.swapaxes(
        theta_obs.reshape(B, n_chunks, chunk_size, 2), 0, 1
    )
    valid_mask = (jnp.arange(padded_H) < n_valid).reshape(n_chunks, chunk_size)

    def scan_chunk(logsumexp_acc, inputs):
        theta_dyn_chunk, theta_obs_chunk, mask_chunk = inputs
        ll_chunk = _scan_loglik_chunk(
            observations_t,
            designs_t,
            theta_dyn_chunk,
            theta_obs_chunk,
            substep_loop,
        )
        ll_chunk = jnp.where(mask_chunk[None, :], ll_chunk, -jnp.inf)
        chunk_lse = jax.scipy.special.logsumexp(ll_chunk, axis=1)
        return jnp.logaddexp(logsumexp_acc, chunk_lse), None

    init = jnp.full((B,), -jnp.inf, dtype=dtype)
    logsumexp_total, _ = jax.lax.scan(
        scan_chunk, init, (theta_dyn_chunks, theta_obs_chunks, valid_mask)
    )
    return logsumexp_total - jnp.log(jnp.asarray(n_valid, dtype=dtype))


# ============================================================================
#  Batched sPCE loss over B episodes
# ============================================================================


def _batched_spce_loss_per_episode(
    params, keys, L, M, hypothesis_chunk_size=32, substep_loop="unrolled"
):
    """Per-episode targeted sPCE loss with full outer batch semantics.

    Args:
        params: policy parameter pytree
        keys: (B,) PRNG keys
        L, M: contrastive / nuisance sample counts
        hypothesis_chunk_size: chunk size for `(L+1)` and `M` axes. Set to 0 to
            materialize the full hypothesis axis at once.
    Returns:
        (B,) loss values
    """
    B = keys.shape[0]
    n_denom = L + 1

    k_all = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
    k_theta = k_all[:, 0]
    k_obs_numer = k_all[:, 1]
    k_noise = k_all[:, 2]

    theta_full = jax.vmap(sample_theta_full, in_axes=(0, None))(k_theta, n_denom)

    theta_dyn_true = theta_full[:, 0, :2]
    sigma_true = theta_full[:, 0, 2]
    Cx0_true = theta_full[:, 0, 3]

    epsilon = jax.vmap(lambda k: jax.random.normal(k, (N_STEPS,)))(k_noise)
    observations_t, designs_t = _rollout_batched(
        params, theta_dyn_true, sigma_true, Cx0_true, epsilon, substep_loop
    )

    theta_dyn_denom = theta_full[:, :, :2]
    theta_obs_denom = theta_full[:, :, 2:]
    log_den = _chunked_logmeanexp(
        observations_t,
        designs_t,
        theta_dyn_denom,
        theta_obs_denom,
        n_denom,
        hypothesis_chunk_size,
        substep_loop,
    )

    theta_obs_numer = jax.vmap(sample_theta_obs, in_axes=(0, None))(k_obs_numer, M)
    theta_dyn_numer = jnp.broadcast_to(theta_dyn_true[:, None, :], (B, M, 2))
    log_num = _chunked_logmeanexp(
        observations_t,
        designs_t,
        theta_dyn_numer,
        theta_obs_numer,
        M,
        hypothesis_chunk_size,
        substep_loop,
    )

    return -(log_num - log_den)


def batched_spce_loss(params, keys, L, M, hypothesis_chunk_size=32, substep_loop="unrolled"):
    """Mean targeted sPCE loss over a fully batched key array."""
    return jnp.mean(
        _batched_spce_loss_per_episode(
            params, keys, L, M, hypothesis_chunk_size, substep_loop
        )
    )


def blocked_batched_spce_loss(
    params,
    keys,
    L,
    M,
    hypothesis_chunk_size=32,
    batch_block_size=0,
    substep_loop="unrolled",
):
    """Batch-first loss with optional blocking over the outer Monte Carlo axis.

    Blocking keeps the semantics of a batched outer expectation while avoiding
    the pathological compile/runtime behavior of both tiny per-episode chunks
    and a single monolithic `B=6706` block.
    """
    B = keys.shape[0]
    if batch_block_size <= 0 or batch_block_size >= B:
        return batched_spce_loss(
            params, keys, L, M, hypothesis_chunk_size, substep_loop
        )

    n_blocks = math.ceil(B / batch_block_size)
    padded_B = n_blocks * batch_block_size
    if padded_B != B:
        pad = padded_B - B
        keys = jnp.concatenate([keys, keys[:pad]], axis=0)
    key_blocks = keys.reshape(n_blocks, batch_block_size, *keys.shape[1:])
    valid_mask = (jnp.arange(padded_B) < B).reshape(n_blocks, batch_block_size)

    def scan_block(total_loss, block_inputs):
        block_keys, block_mask = block_inputs
        loss_per_episode = _batched_spce_loss_per_episode(
            params, block_keys, L, M, hypothesis_chunk_size, substep_loop
        )
        total_loss = total_loss + jnp.sum(
            jnp.where(block_mask, loss_per_episode, 0.0)
        )
        return total_loss, None

    total_loss, _ = jax.lax.scan(
        scan_block,
        jnp.asarray(0.0, dtype=jnp.float32),
        (key_blocks, valid_mask),
    )
    return total_loss / jnp.asarray(B, dtype=jnp.float32)
