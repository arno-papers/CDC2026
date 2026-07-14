"""Transformer policy for Monod bioreactor (pure JAX, no Flax/Equinox)."""

import jax
import jax.numpy as jnp

from .model import N_STEPS, ACTION_HI

# ============================================================================
#  Architecture constants
# ============================================================================

D_MODEL = 32
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 8
D_FF = 64

# ============================================================================
#  Sinusoidal positional encoding (precomputed)
# ============================================================================


def _sinusoidal_pe():
    """Positional encoding matching common_core.jl:39-47.

    Returns (N_STEPS, D_MODEL) array with interleaved sin/cos.
    """
    pos = jnp.arange(N_STEPS, dtype=jnp.float32)
    div = jnp.exp(
        jnp.arange(0, D_MODEL, 2, dtype=jnp.float32) * -(jnp.log(1000.0) / D_MODEL)
    )
    angles = jnp.outer(pos, div)  # (N_STEPS, D_MODEL//2)
    pe = jnp.zeros((N_STEPS, D_MODEL), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(angles))
    pe = pe.at[:, 1::2].set(jnp.cos(angles))
    return pe


PE = _sinusoidal_pe()

# ============================================================================
#  Parameter initialization
# ============================================================================


def _glorot(key, shape):
    """Glorot uniform initialization."""
    fan_in, fan_out = shape[-1], shape[-2] if len(shape) > 1 else shape[-1]
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_params(key):
    """Initialize transformer policy parameters.

    Returns a nested dict (pytree) matching the Lux architecture in model.jl:166-182.
    """
    k = jax.random.split(key, 8)
    return {
        "input_proj": {
            "weight": _glorot(k[0], (D_MODEL, 2)),
            "bias": jnp.zeros(D_MODEL),
        },
        "rms1": {"scale": jnp.ones(D_MODEL)},
        "mha": {
            "q": _glorot(k[1], (D_MODEL, D_MODEL)),
            "k": _glorot(k[2], (D_MODEL, D_MODEL)),
            "v": _glorot(k[3], (D_MODEL, D_MODEL)),
            "out_w": _glorot(k[4], (D_MODEL, D_MODEL)),
            "out_b": jnp.zeros(D_MODEL),
        },
        "rms2": {"scale": jnp.ones(D_MODEL)},
        "ff": {
            "w1": _glorot(k[5], (D_FF, D_MODEL)),
            "b1": jnp.zeros(D_FF),
            "w2": _glorot(k[6], (D_MODEL, D_FF)),
            "b2": jnp.zeros(D_MODEL),
        },
        "out": {
            "w": _glorot(k[7], (1, D_MODEL)),
            "b": jnp.full((1,), -4.0),
        },
    }


# ============================================================================
#  Building blocks
# ============================================================================


def _rms_norm(x, scale):
    """RMSNorm over last dimension, matching Lux RMSNorm((32,))."""
    return x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6) * scale


def _mha(x, p):
    """Multi-head self-attention (4 heads, no causal mask).

    x: (seq_len, D_MODEL), p: mha param dict.
    """
    S = x.shape[0]
    q = (x @ p["q"].T).reshape(S, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = (x @ p["k"].T).reshape(S, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = (x @ p["v"].T).reshape(S, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    w = jax.nn.softmax(
        q @ jnp.swapaxes(k, -2, -1) / jnp.sqrt(jnp.float32(HEAD_DIM)), axis=-1
    )
    o = (w @ v).transpose(1, 0, 2).reshape(S, D_MODEL)
    return o @ p["out_w"].T + p["out_b"]


# ============================================================================
#  Forward pass
# ============================================================================


def policy_forward(params, buf):
    """Transformer policy forward pass.

    Args:
        params: parameter pytree from init_params()
        buf: (N_STEPS, 2) input buffer [observations, actions]
    Returns:
        scalar action in [0, ACTION_HI]
    """
    # Input projection + positional encoding
    x = buf @ params["input_proj"]["weight"].T + params["input_proj"]["bias"] + PE
    # Pre-norm self-attention block
    x = x + _mha(_rms_norm(x, params["rms1"]["scale"]), params["mha"])
    # Pre-norm feed-forward block
    h = _rms_norm(x, params["rms2"]["scale"])
    ff = jax.nn.gelu(h @ params["ff"]["w1"].T + params["ff"]["b1"])
    x = x + ff @ params["ff"]["w2"].T + params["ff"]["b2"]
    # Output: last position -> sigmoid -> scale
    logit = x[-1] @ params["out"]["w"].T + params["out"]["b"]
    return ACTION_HI * jax.nn.sigmoid(logit[0])
