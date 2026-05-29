"""Benchmark JAX Monod sPCE gradients on GPU.

Defaults are chosen to match the Julia/Reactant micro-batch benchmark:
`B_multiplier=8`, `grad_accum=1`, and `mode=full-batch`.
"""

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from monod_jax.batched import blocked_batched_spce_loss
from monod_jax.model import allocate_budget
from monod_jax.policy import init_params
from monod_jax.training import batch_loss


def main():
    parser = argparse.ArgumentParser(description="JAX Monod sPCE GPU benchmark")
    parser.add_argument("--B", type=int, default=0, help="Micro-batch size (0=auto)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("full-batch", "legacy-chunked"),
        default="full-batch",
        help="Execution mode",
    )
    parser.add_argument(
        "--chunk",
        "--episode-chunk",
        dest="episode_chunk",
        type=int,
        default=128,
        help="Episode vmap chunk size for legacy-chunked mode (0=full vmap)",
    )
    parser.add_argument(
        "--hyp-chunk",
        type=int,
        default=32,
        help="Hypothesis chunk size for full-batch mode (0=materialize full H)",
    )
    parser.add_argument(
        "--batch-block",
        type=int,
        default=1024,
        help="Outer batch block size for full-batch mode (0=single full-B block)",
    )
    parser.add_argument(
        "--substep-loop",
        type=str,
        choices=("unrolled", "scan", "fori"),
        default="unrolled",
        help="RK4 substep loop implementation",
    )
    parser.add_argument("--budget", type=int, default=6_250_000, help="ODE trajectory budget")
    parser.add_argument(
        "--B-multiplier",
        type=int,
        default=8,
        help="B_multiplier for budget allocation (Julia/Reactant micro-batch uses 8)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (each over B_micro)",
    )
    parser.add_argument("--n-runs", type=int, default=20, help="Timed gradient evaluations")
    parser.add_argument("--n-warmup", type=int, default=2, help="Warmup runs (incl. compile)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    L, M, B_auto = allocate_budget(args.budget, B_multiplier=args.B_multiplier)
    B_micro = args.B if args.B > 0 else B_auto
    if args.mode == "legacy-chunked" and args.episode_chunk > 0:
        B_micro = (B_micro // args.episode_chunk) * args.episode_chunk
    grad_accum = args.grad_accum
    B_total = B_micro * grad_accum
    traj_per_step = B_total * (L + 2 + M)

    print(f"Device: {jax.devices()[0]}")
    print(f"B_micro={B_micro}, B_total={B_total}, grad_accum={grad_accum}")
    if args.mode == "full-batch":
        print(
            f"hyp_chunk_size={args.hyp_chunk}, "
            f"batch_block_size={args.batch_block}, "
            f"substep_loop={args.substep_loop}"
        )
    else:
        effective_chunk = args.episode_chunk if args.episode_chunk > 0 else B_micro
        print(
            f"episode_chunk_size={effective_chunk}, "
            f"n_chunks={B_micro // effective_chunk}, "
            f"substep_loop={args.substep_loop}"
        )
    print(f"L={L}, M={M}")
    print(f"Trajectories per grad step: {traj_per_step:,}")
    print()

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Parameters: {n_params:,}")

    @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def grad_step_full_batch(
        params, keys, L, M, hyp_chunk_size, batch_block_size, substep_loop
    ):
        return jax.value_and_grad(blocked_batched_spce_loss)(
            params, keys, L, M, hyp_chunk_size, batch_block_size, substep_loop
        )

    @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def grad_step_legacy(params, keys, L, M, episode_chunk_size, substep_loop):
        return jax.value_and_grad(batch_loss)(
            params, keys, L, M, episode_chunk_size, substep_loop
        )

    def run_one_step(params, key):
        total_loss = 0.0
        acc_grads = None
        for _ in range(grad_accum):
            key, sk = jax.random.split(key)
            keys = jax.random.split(sk, B_micro)
            if args.mode == "full-batch":
                loss, grads = grad_step_full_batch(
                    params,
                    keys,
                    L,
                    M,
                    args.hyp_chunk,
                    args.batch_block,
                    args.substep_loop,
                )
            else:
                loss, grads = grad_step_legacy(
                    params, keys, L, M, args.episode_chunk, args.substep_loop
                )
            total_loss += float(loss)
            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
        jax.block_until_ready(acc_grads)
        return key, total_loss / grad_accum

    print("Compiling...", flush=True)
    t_compile_start = time.perf_counter()
    for i in range(args.n_warmup):
        key, avg_loss = run_one_step(params, key)
        if i == 0:
            t_compile = time.perf_counter() - t_compile_start
            print(f"First call (compile): {t_compile:.1f}s, loss={avg_loss:.6f}")

    print(f"\nBenchmarking {args.n_runs} gradient evaluations...")
    times = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        key, avg_loss = run_one_step(params, key)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    median_ms = np.median(times) * 1000
    mean_ms = np.mean(times) * 1000
    traj_per_sec = traj_per_step / np.median(times)

    print(f"\n{'=' * 60}")
    print("JAX GPU Benchmark — Monod Bioreactor sPCE")
    print(f"{'=' * 60}")
    if args.mode == "full-batch":
        mode = (
            f"full-batch (hyp_chunk={args.hyp_chunk}, "
            f"batch_block={args.batch_block}, "
            f"substep_loop={args.substep_loop})"
        )
    else:
        mode = (
            f"legacy-chunked (episode_chunk={args.episode_chunk}, "
            f"substep_loop={args.substep_loop})"
        )
    print(f"  B_micro={B_micro}, B_total={B_total}, grad_accum={grad_accum}")
    print(f"  mode={mode}, L={L}, M={M}")
    print(f"  Trajectories/step:  {traj_per_step:>12,}")
    print(f"  Parameters:         {n_params:>12,}")
    print(f"  Grad step:          {median_ms:>10.2f} ms (median), {mean_ms:.2f} ms (mean)")
    print(f"  Throughput:         {traj_per_sec:>12,.0f} trajectories/s")
    print(f"  Compile:            {t_compile:>10.1f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
