from __future__ import annotations

import jax.numpy as jnp


def sinusoidal_pe(
    seq_len: int,
    embed_dim: int = 32,
    *,
    base: float = 1000.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Sinusoidal positional encoding matching `common.jl`.

    Returns shape (seq_len, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    position = jnp.arange(seq_len, dtype=dtype)[None, :]  # (1, T)
    div_term = jnp.exp(
        jnp.arange(0, embed_dim, 2, dtype=dtype)
        * (-(jnp.log(jnp.asarray(base, dtype=dtype)) / embed_dim))
    )  # (D/2,)
    angles = div_term[:, None] * position  # (D/2, T)

    pe = jnp.zeros((seq_len, embed_dim), dtype=dtype)
    pe = pe.at[:, 0::2].set(jnp.sin(angles).T)
    pe = pe.at[:, 1::2].set(jnp.cos(angles).T)
    return pe
