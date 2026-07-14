"""Profile JAX Monod sPCE gradients for comparison with Reactant."""

import argparse
import functools
import os
import time

import jax

from monod_jax.batched import blocked_batched_spce_loss
from monod_jax.model import allocate_budget
from monod_jax.policy import init_params
from monod_jax.training import batch_loss


def main():
    parser = argparse.ArgumentParser(description="JAX Monod sPCE profiling")
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
        default=64,
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
    parser.add_argument("--B-multiplier", type=int, default=8)
    parser.add_argument("--n-runs", type=int, default=3, help="Profiled runs")
    parser.add_argument("--n-warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perfetto", type=str, default=None, help="Directory for Perfetto trace")
    parser.add_argument("--hlo", action="store_true", help="Dump compiled HLO text")
    args = parser.parse_args()

    L, M, B_auto = allocate_budget(args.budget, B_multiplier=args.B_multiplier)
    B = args.B if args.B > 0 else B_auto
    if args.mode == "legacy-chunked" and args.episode_chunk > 0:
        B = (B // args.episode_chunk) * args.episode_chunk

    print(f"Device: {jax.devices()[0]}")
    if args.mode == "full-batch":
        print(
            f"B={B}, hyp_chunk={args.hyp_chunk}, "
            f"batch_block={args.batch_block}, "
            f"substep_loop={args.substep_loop}, L={L}, M={M}"
        )
    else:
        print(
            f"B={B}, episode_chunk={args.episode_chunk}, "
            f"substep_loop={args.substep_loop}, L={L}, M={M}"
        )
    print(f"Trajectories per grad step: {B * (L + 2 + M):,}")
    print()

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)

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

    grad_step = (
        grad_step_full_batch if args.mode == "full-batch" else grad_step_legacy
    )
    mode_args = (
        (args.hyp_chunk, args.batch_block, args.substep_loop)
        if args.mode == "full-batch"
        else (args.episode_chunk, args.substep_loop)
    )

    if args.hlo:
        print("Lowering and compiling for HLO inspection...")
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, B)
        lowered = grad_step.lower(params, keys, L, M, *mode_args)
        compiled = lowered.compile()
        hlo = compiled.as_text()

        n_fusion = hlo.count("fusion(")
        n_while = hlo.count("while(")
        n_command_buffer = hlo.count("command_buffer")
        print(
            f"HLO stats: {n_fusion} fusions, {n_while} while-loops, "
            f"{n_command_buffer} command_buffers"
        )
        print(f"HLO size: {len(hlo):,} chars")

        hlo_path = "/tmp/monod_jax_grad.hlo.txt"
        with open(hlo_path, "w", encoding="utf-8") as f:
            f.write(hlo)
        print(f"Full HLO written to {hlo_path}")
        return

    print("Compiling...", flush=True)
    t0 = time.perf_counter()
    for i in range(args.n_warmup):
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, B)
        loss, grads = grad_step(params, keys, L, M, *mode_args)
        jax.block_until_ready(loss)
        if i == 0:
            print(
                f"First call (compile): {time.perf_counter() - t0:.1f}s, "
                f"loss={float(loss):.6f}"
            )

    perfetto_dir = args.perfetto or "/tmp/jax_monod_trace"
    os.makedirs(perfetto_dir, exist_ok=True)

    print(f"\nProfiling {args.n_runs} grad steps -> {perfetto_dir}")
    jax.profiler.start_trace(perfetto_dir)
    for i in range(args.n_runs):
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, B)
        t0 = time.perf_counter()
        loss, grads = grad_step(params, keys, L, M, *mode_args)
        jax.block_until_ready(loss)
        dt = time.perf_counter() - t0
        print(f"  Run {i+1}: {dt*1000:.1f} ms, loss={float(loss):.6f}")
    jax.profiler.stop_trace()

    print(f"\nPerfetto trace saved to {perfetto_dir}")
    print("View at https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
