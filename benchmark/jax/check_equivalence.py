"""Compare the legacy and full-batch JAX losses on fixed inputs."""

import argparse

import jax
import jax.numpy as jnp

from monod_jax.batched import batched_spce_loss
from monod_jax.policy import init_params
from monod_jax.training import batch_loss


def _tree_max_abs(tree):
    leaves = jax.tree.leaves(tree)
    return max(float(jnp.nanmax(jnp.abs(x))) for x in leaves)


def _tree_max_abs_diff(tree_a, tree_b):
    diffs = jax.tree.map(lambda a, b: jnp.nanmax(jnp.abs(a - b)), tree_a, tree_b)
    return max(float(x) for x in jax.tree.leaves(diffs))


def _tree_max_rel_diff(tree_a, tree_b, eps):
    rel = jax.tree.map(
        lambda a, b: jnp.max(
            jnp.nan_to_num(jnp.abs(a - b), nan=jnp.inf)
            / jnp.maximum(jnp.maximum(jnp.abs(a), jnp.abs(b)), eps)
        ),
        tree_a,
        tree_b,
    )
    return max(float(x) for x in jax.tree.leaves(rel))


def _tree_has_nan(tree):
    return any(bool(jnp.isnan(x).any()) for x in jax.tree.leaves(tree))


def main():
    parser = argparse.ArgumentParser(description="Check JAX loss/gradient equivalence")
    parser.add_argument("--B", type=int, default=64, help="Episode batch size")
    parser.add_argument("--L", type=int, default=8, help="Contrastive samples")
    parser.add_argument("--M", type=int, default=8, help="Nuisance samples")
    parser.add_argument(
        "--episode-chunk",
        type=int,
        default=16,
        help="Legacy episode chunk size",
    )
    parser.add_argument(
        "--hyp-chunk",
        type=int,
        default=4,
        help="Full-batch hypothesis chunk size",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    key = jax.random.key(args.seed)
    key, init_key, batch_key = jax.random.split(key, 3)
    params = init_params(init_key)
    keys = jax.random.split(batch_key, args.B)

    legacy_fn = jax.jit(jax.value_and_grad(batch_loss), static_argnums=(2, 3, 4))
    full_fn = jax.jit(
        jax.value_and_grad(batched_spce_loss), static_argnums=(2, 3, 4)
    )

    legacy_loss, legacy_grads = legacy_fn(
        params, keys, args.L, args.M, args.episode_chunk
    )
    full_loss, full_grads = full_fn(params, keys, args.L, args.M, args.hyp_chunk)

    legacy_loss, full_loss, legacy_grads, full_grads = jax.tree.map(
        jax.block_until_ready,
        (legacy_loss, full_loss, legacy_grads, full_grads),
    )

    loss_abs = float(jnp.abs(legacy_loss - full_loss))
    loss_rel = float(
        loss_abs / max(abs(float(legacy_loss)), abs(float(full_loss)), 1e-8)
    )
    grad_abs = _tree_max_abs_diff(legacy_grads, full_grads)
    grad_rel = _tree_max_rel_diff(legacy_grads, full_grads, 1e-8)
    grad_scale = max(_tree_max_abs(legacy_grads), _tree_max_abs(full_grads), 1e-8)
    legacy_has_nan = _tree_has_nan(legacy_grads)
    full_has_nan = _tree_has_nan(full_grads)

    print("=== JAX Equivalence Check ===")
    print(f"B={args.B}, L={args.L}, M={args.M}")
    print(
        f"legacy episode_chunk={args.episode_chunk}, "
        f"full-batch hyp_chunk={args.hyp_chunk}"
    )
    print(f"loss legacy={float(legacy_loss):.8f}, full_batch={float(full_loss):.8f}")
    print(f"loss abs diff={loss_abs:.3e}, rel diff={loss_rel:.3e}")
    print(f"legacy grad has_nan={legacy_has_nan}, full grad has_nan={full_has_nan}")
    print(f"grad max abs diff={grad_abs:.3e}")
    print(f"grad max rel diff={grad_rel:.3e}")
    print(f"grad scale={grad_scale:.3e}")

    if loss_abs > args.atol and loss_rel > args.rtol:
        raise SystemExit("Loss equivalence check failed.")
    if legacy_has_nan or full_has_nan:
        raise SystemExit("Gradient equivalence check failed: NaNs detected.")
    if grad_abs > args.atol and grad_rel > args.rtol:
        raise SystemExit("Gradient equivalence check failed.")

    print("Equivalence check passed.")


if __name__ == "__main__":
    main()
