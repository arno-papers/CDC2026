from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from .positional_encoding import sinusoidal_pe


class PolicyTransformer(nn.Module):
    embed_dim: int = 32
    num_heads: int = 4
    ff_mult: int = 2
    action_scale: float = 10.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        x: (B, T, 2) where x[..., 0] is observation and x[..., 1] is action.
        Returns: (B,) action in (0, action_scale).
        """
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected x shape (B,T,2), got {x.shape}")

        t = x.shape[1]
        h = nn.Dense(self.embed_dim, name="input_proj")(x)
        h = h + sinusoidal_pe(t, self.embed_dim, base=1000.0, dtype=h.dtype)[None, :, :]

        h_norm = nn.RMSNorm(name="rms1")(h)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            name="mha",
        )(h_norm, h_norm)
        h = h + attn

        ff_in = nn.RMSNorm(name="rms2")(h)
        ff = nn.Dense(self.embed_dim * self.ff_mult, name="ff1")(ff_in)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.embed_dim, name="ff2")(ff)
        h = h + ff

        last = h[:, -1, :]
        a = nn.Dense(1, name="output_head")(last)
        a = self.action_scale * nn.sigmoid(a)
        return a[:, 0]
