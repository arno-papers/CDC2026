"""Targeted sPCE loss and training loop for Monod bioreactor."""

import functools
import math
import time

import jax
import jax.numpy as jnp
import optax

from .batched import blocked_batched_spce_loss
from .model import (
    N_STEPS,
    CS0,
    V0,
    get_integrator,
    sample_theta_full,
    sample_theta_obs,
)
from .policy import policy_forward

# ============================================================================
#  Log-likelihood for a single hypothesis
# ============================================================================


def _compute_ll(observations, designs, theta_dyn, theta_obs, substep_loop="unrolled"):
    """Total log-likelihood of observations under one parameter hypothesis."""
    integrate_fn = get_integrator(substep_loop)
    u0 = jnp.array([CS0, theta_obs[1], V0])
    sigma_sq = theta_obs[0] ** 2

    def step_fn(carry, x):
        u, ll = carry
        obs, design = x
        u = integrate_fn(u, theta_dyn, design)
        residual = obs - u[0]
        ll = ll - 0.5 * (residual**2 / sigma_sq + jnp.log(sigma_sq))
        return (u, ll), None

    (_, ll_total), _ = jax.lax.scan(step_fn, (u0, 0.0), (observations, designs))
    return ll_total


def _compute_ll_batch(
    observations, designs, theta_dyn_batch, theta_obs_batch, substep_loop="unrolled"
):
    """Vectorized `_compute_ll` over hypotheses."""
    return jax.vmap(
        lambda theta_dyn, theta_obs: _compute_ll(
            observations, designs, theta_dyn, theta_obs, substep_loop
        )
    )(theta_dyn_batch, theta_obs_batch)

# ============================================================================
#  Single-episode sPCE loss (vmapped over B outside)
# ============================================================================


def single_episode_loss(params, key, L, M, substep_loop="unrolled"):
    """Targeted sPCE loss for ONE episode.

    The batch (B) dimension is handled by an outer jax.vmap.

    Args:
        params: policy parameter pytree
        key: PRNG key for this episode
        L: number of contrastive samples (denominator has L+1 including true)
        M: number of nuisance samples for numerator
    Returns:
        scalar loss
    """
    integrate_fn = get_integrator(substep_loop)
    k_theta, k_obs_numer, k_noise = jax.random.split(key, 3)

    # --- Sample parameters ---
    n_denom = L + 1
    theta_full = sample_theta_full(k_theta, n_denom)  # (n_denom, 4)

    # True parameters (first sample)
    theta_dyn_true = theta_full[0, :2]  # (2,) [mu_max, K_s]
    sigma_true = theta_full[0, 2]  # scalar noise std
    Cx0_true = theta_full[0, 3]  # scalar initial biomass

    # Pre-sample observation noise
    epsilon = jax.random.normal(k_noise, (N_STEPS,))

    # --- Adaptive policy loop ---
    u0 = jnp.array([CS0, Cx0_true, V0])
    input_buffer0 = jnp.zeros((N_STEPS, 2))

    def adaptive_step(carry, step_idx):
        u, input_buffer = carry
        action = policy_forward(params, input_buffer)
        u = integrate_fn(u, theta_dyn_true, action)
        y_obs = u[0] + sigma_true * epsilon[step_idx]
        input_buffer = input_buffer.at[step_idx, 0].set(y_obs)
        input_buffer = input_buffer.at[step_idx, 1].set(action)
        return (u, input_buffer), (y_obs, action)

    (_, _), (observations, designs) = jax.lax.scan(
        adaptive_step, (u0, input_buffer0), jnp.arange(N_STEPS)
    )
    # observations: (N_STEPS,), designs: (N_STEPS,)

    # --- Denominator: L+1 hypotheses ---
    theta_dyn_denom = theta_full[:, :2]  # (n_denom, 2)
    theta_obs_denom = theta_full[:, 2:]  # (n_denom, 2)
    ll_denom = _compute_ll_batch(
        observations, designs, theta_dyn_denom, theta_obs_denom, substep_loop
    )

    # --- Numerator: M nuisance samples with true dynamics ---
    theta_obs_numer = sample_theta_obs(k_obs_numer, M)  # (M, 2)
    theta_dyn_numer = jnp.broadcast_to(theta_dyn_true, (M, 2))  # (M, 2)
    ll_numer = _compute_ll_batch(
        observations, designs, theta_dyn_numer, theta_obs_numer, substep_loop
    )

    # --- Targeted sPCE ---
    log_num = jax.scipy.special.logsumexp(ll_numer) - jnp.log(jnp.float32(M))
    log_den = jax.scipy.special.logsumexp(ll_denom) - jnp.log(jnp.float32(n_denom))
    return -(log_num - log_den)


# ============================================================================
#  Batch loss (vmapped over B)
# ============================================================================


@functools.partial(jax.checkpoint, static_argnums=(2, 3, 4))
def _episode_loss_remat(params, key, L, M, substep_loop="unrolled"):
    """Checkpointed single_episode_loss — stores only (key, loss) for backward,
    recomputes all intermediates. Allows full-vmap over large B."""
    return single_episode_loss(params, key, L, M, substep_loop)


def batch_loss(params, keys, L, M, chunk_size=0, substep_loop="unrolled"):
    """Mean sPCE loss over B episodes.

    When chunk_size > 0, processes episodes in small vmapped chunks via
    lax.scan. When chunk_size <= 0, uses full vmap with per-episode
    checkpointing for memory efficiency.

    Args:
        params: policy parameter pytree
        keys: (B,) array of PRNG keys
        L, M: contrastive / nuisance sample counts (static)
        chunk_size: vmap width per chunk (0 = full vmap with remat)
    Returns:
        scalar mean loss
    """
    if chunk_size <= 0 or chunk_size >= keys.shape[0]:
        # Full vmap with per-episode remat
        losses = jax.vmap(
            lambda key: _episode_loss_remat(params, key, L, M, substep_loop)
        )(keys)
        return jnp.mean(losses)

    # Chunked: scan over vmapped chunks. No chunk-level checkpoint —
    # the adaptive_step checkpoint already controls memory per episode.
    def _chunk_body(params, chunk_keys, L, M):
        return jax.vmap(
            lambda key: single_episode_loss(params, key, L, M, substep_loop)
        )(chunk_keys)

    n = keys.shape[0]
    n_chunks = n // chunk_size
    chunked_keys = keys.reshape(n_chunks, chunk_size, *keys.shape[1:])

    def _scan_step(acc, ck):
        return acc + jnp.sum(_chunk_body(params, ck, L, M)), None

    total, _ = jax.lax.scan(_scan_step, jnp.float32(0.0), chunked_keys)
    return total / n


# ============================================================================
#  Training loop
# ============================================================================


def train(
    params,
    key,
    *,
    L,
    M,
    B,
    n_iters=1000,
    lr_max=0.003,
    lr_min=1e-5,
    warmup=50,
    grad_accum=1,
    mode="full_batch",
    chunk_size=128,
    hypothesis_chunk_size=32,
    batch_block_size=1024,
    substep_loop="unrolled",
    print_every=10,
):
    """Train the adaptive policy.

    Args:
        params: initial parameter pytree
        key: PRNG key
        L, M: contrastive / nuisance sample counts
        B: total batch size (split into grad_accum micro-batches)
        n_iters: training iterations
        lr_max, lr_min, warmup: cosine schedule parameters
        grad_accum: gradient accumulation steps
        mode: "full_batch" (default) or "legacy_chunked"
        chunk_size: episode vmap chunk size for legacy mode
        hypothesis_chunk_size: hypothesis chunk size for full-batch mode
        batch_block_size: outer batch block size for full-batch mode
        substep_loop: RK4 substep loop implementation
        print_every: logging interval
    Returns:
        (params, loss_history)
    """
    B_micro = B // grad_accum

    # Cosine LR with warmup (matches common_core.jl:53-59)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_min,
        peak_value=lr_max,
        warmup_steps=warmup,
        decay_steps=n_iters,
        end_value=lr_min,
    )
    opt = optax.adam(learning_rate=schedule)
    opt_state = opt.init(params)

    if mode == "full_batch":
        compiled_chunks = (hypothesis_chunk_size, batch_block_size, substep_loop)

        @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
        def grad_fn(
            params, key, L, M, hypothesis_chunk_size, batch_block_size, substep_loop
        ):
            keys = jax.random.split(key, B_micro)
            return jax.value_and_grad(blocked_batched_spce_loss)(
                params,
                keys,
                L,
                M,
                hypothesis_chunk_size,
                batch_block_size,
                substep_loop,
            )
    elif mode == "legacy_chunked":
        compiled_chunks = (chunk_size, substep_loop)

        @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
        def grad_fn(params, key, L, M, chunk_size, substep_loop):
            keys = jax.random.split(key, B_micro)
            return jax.value_and_grad(batch_loss)(
                params, keys, L, M, chunk_size, substep_loop
            )
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    loss_history = []
    t_start = time.time()

    for step in range(n_iters):
        total_loss = 0.0
        acc_grads = None

        for _ in range(grad_accum):
            key, sk = jax.random.split(key)
            if mode == "full_batch":
                loss, grads = grad_fn(
                    params,
                    sk,
                    L,
                    M,
                    compiled_chunks[0],
                    compiled_chunks[1],
                    compiled_chunks[2],
                )
            else:
                loss, grads = grad_fn(
                    params, sk, L, M, compiled_chunks[0], compiled_chunks[1]
                )
            total_loss += float(loss)
            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = jax.tree.map(jnp.add, acc_grads, grads)

        avg_grads = jax.tree.map(lambda g: g / grad_accum, acc_grads)
        updates, opt_state = opt.update(avg_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        avg_loss = total_loss / grad_accum
        loss_history.append(avg_loss)

        if (step + 1) % print_every == 0 or step == 0:
            elapsed = time.time() - t_start
            lr_t = float(schedule(step))
            print(
                f"Iter: [{step + 1:4d}/{n_iters}]\tLoss: {avg_loss:.8f}\t"
                f"lr: {lr_t:.6f}\ttime: {elapsed:.1f}s"
            )

    return params, loss_history
