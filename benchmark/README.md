# JAX vs Julia/Reactant — Monod sPCE gradient-step benchmark

A fair, matched-config head-to-head of the **targeted sPCE gradient step** (the
inner loop of adaptive experimental design training) implemented two ways:

- **JAX** — the `monod_jax` package from commit `3188aaa` ("Add JAX monod
  implementation"), vendored under [`jax/`](jax/).
- **Julia / Reactant** — the production `targeted_spce_loss` + transformer policy
  (`common.jl`, copied from `src/`) with two interchangeable ODE integrators, plus
  the standalone Monod model from `simplediffeq_test/`, vendored under
  [`julia/`](julia/).

Both compute the same estimator on the same model at the same `(L, M, B)`, in
float32, and time one **loss + reverse-mode gradient** evaluation on the GPU.

## Why a separate folder / fresh environment

`simplediffeq_test/` pins a **dev checkout** of Reactant (`0.2.263`). This folder
pins the **latest registered release**, `Reactant = "=0.2.273"`, so the benchmark
doubles as a check that the production loss still compiles, differentiates, and
trains on a released Reactant — not just a local branch.

## What is being compared

| | integrator / substep loop | source |
|---|---|---|
| Julia `hand` | hand-written RK4 inside a Reactant `@trace for` | `julia/common.jl` (production) |
| Julia `sd` | SimpleDiffEq `SimpleRK4` `step!` inside a `@trace for` | `julia/bench.jl` |
| JAX `scan` | RK4 substeps in `lax.scan` | `jax/monod_jax/model.py` |
| JAX `fori` | RK4 substeps in `lax.fori_loop` | `jax/monod_jax/model.py` |

The JAX package also ships a fully-`unrolled` substep loop, but it is **not**
benchmarked here — see the asymmetry note below.

Model/config identical across all rows: Monod bioreactor (state `[C_s, C_x, V]`),
`N_STEPS=14` adaptive windows, `N_SUBSTEPS=50` RK4 substeps/window, a ~8.5k-param
transformer policy, and the targeted sPCE estimator with `L+1` contrastive and `M`
nuisance hypotheses.

## Fairness ground rules

1. **Matched `(L, M, B)`.** Default `L=255, M=256, B=512`. The JAX harness normally
   derives `L, M` from an ODE budget; `benchmark.py` gained `--L/--M` overrides here
   so both sides run the *exact* same shape.
2. **Same timed region: loss + gradient only.** JAX times `value_and_grad`. Julia
   times `single_train_step!`, which additionally applies one Adam update — but on
   ~8.5k parameters that is a handful of elementwise ops (µs-scale, <<1% of a
   ~200 ms step), so the two measure effectively the same thing.
3. **Full materialization on both sides.** JAX runs with `--hyp-chunk 0 --batch-block 0`
   (no hypothesis/batch chunking), matching Reactant materializing the full
   `(3, L+1, B)` / `(M, B)` tensors. This is the clean apples-to-apples point; at
   much larger `B` the two diverge by *memory strategy* (JAX chunks, Reactant
   `@trace mincut` rematerializes), which is a different comparison.
4. **Data prep excluded, device sync forced.** Batch sampling happens outside the
   timer; the loss scalar is materialized inside (`Float32(l)` / `block_until_ready`)
   so we time the whole compiled program, not an async dispatch.
5. **Median of `NTIME`/`--n-runs` steps after warmup** (warmup absorbs compilation).
6. **Sequential runs.** `run_all.sh` never runs two variants at once, so neither is
   starved on the shared card. Each process is capped to `MEM_FRAC` of GPU memory.

### Known minor asymmetries (documented, not corrected)

- The vendored `common.jl` sets `--xla_gpu_deterministic_ops=true` (a production
  setting for reproducible gradients); the JAX path does not force determinism.
  Deterministic reductions can be marginally slower, so if anything this slightly
  *disadvantages* the Julia numbers.
- Policy weight initialization and RNG differ between the two implementations, so the
  reported `loss=` values are sanity signals (finite, right magnitude), **not** a
  numerical-equivalence check. Compute cost is dominated by ODE integration, which is
  identical in structure.

### Why fully-unrolled JAX is deliberately excluded

JAX can fully **unroll** the substep loop; Reactant cannot. Deep unrolling blows up
Reactant's `call_with_reactant` tracing recursion (StackOverflow / minutes-long
compiles), so `@trace for` is its practical ceiling — there is **no Julia/Reactant
counterpart to benchmark against**. Unrolling also does not scale on its own terms:
its compile time explodes super-linearly with `N_SUBSTEPS` and is impractical for
everyone by `N_SUBSTEPS≈500` (the deep-Monod regime), where all roads lead back to
`scan`/`@trace for`. Since it is neither comparable across frameworks nor viable at
production depth, it is left out. The honest apples-to-apples axis is **JAX
`scan`/`fori` vs Reactant `@trace for`**, which is what this benchmark reports.

## Environment

- Julia 1.12.6 (`release` juliaup channel), `Reactant = "=0.2.273"` (latest release).
- JAX `jax[cuda12]` **0.10.2** (uv-resolved; pinned in `jax/uv.lock`).
- GPU: NVIDIA RTX 4080 SUPER (16 GB), CUDA 12. float32 throughout.

## Running

```bash
# One shot: build both envs first (see below), then
./benchmark/run_all.sh                     # matched L=255 M=256 B=512
L=255 M=256 B=512 MEM_FRAC=0.45 ./benchmark/run_all.sh   # explicit

# Environments (first time)
julia +release --project=benchmark/julia -e 'using Pkg; Pkg.instantiate()'
cd benchmark/jax && uv sync    # installs jax[cuda12] from the committed uv.lock
```

Individual runs:

```bash
# Julia
INTEG=hand L=255 M=256 BMICRO=512 BACKEND=gpu \
  julia +release --project=benchmark/julia benchmark/julia/bench.jl
INTEG=sd   L=255 M=256 BMICRO=512 BACKEND=gpu \
  julia +release --project=benchmark/julia benchmark/julia/bench.jl

# JAX
uv run --project=benchmark/jax python benchmark/jax/benchmark.py \
  --L 255 --M 256 --B 512 --mode full-batch --hyp-chunk 0 --batch-block 0 \
  --substep-loop scan     # scan | fori
```

## Results

Matched config: **L=255, M=256, B=512**, `N_STEPS=14`, `N_SUBSTEPS=50`, float32,
262,656 trajectories/step. RTX 4080 SUPER (shared, `MEM_FRAC=0.45`), Reactant
`0.2.273`, jax `0.10.2`. Per-step = median of 20 loss+gradient evaluations after
warmup.

| implementation | substep loop | per-step (median) | compile |
|---|---|---|---|
| **Julia / Reactant** — SimpleDiffEq | `@trace for` + `step!` | **105.7 ms** | 205 s |
| **Julia / Reactant** — hand-RK4 | `@trace for` | **113.5 ms** | 200 s |
| **JAX** | `lax.scan` | 289.0 ms | **2.8 s** |
| **JAX** | `lax.fori_loop` | 290.1 ms | **2.7 s** |

Per-step distributions were tight (Julia hand 112.8–115.5 ms; sd 104.9–106.4 ms).

### Findings

1. **On the apples-to-apples loop axis, Reactant is ~2.7× faster per step than JAX**
   (≈106–114 ms vs ≈289–290 ms) at this matched full-materialization config. Both
   `lax.scan` and `lax.fori_loop` lower to an XLA `while` loop and land within 0.4%
   of each other; Reactant's `@trace for` lowers to a tighter `while`.
2. **hand-RK4 ≈ SimpleDiffEq** (113.5 vs 105.7 ms, ~7%; SimpleDiffEq marginally
   faster here). XLA fuses both to equivalent HLO — the SimpleDiffEq `step!`-in-
   `@trace for` integrator is a genuine drop-in with no runtime penalty. Both
   produce the *identical* loss to 5 significant figures (−0.71462).
3. **JAX compiles ~70–75× faster** (≈2.7 s vs ≈200 s). This matters for
   dev-iteration latency; amortized over a multi-hour training run it is negligible.
4. Absolute Reactant per-step times here (~110 ms) are ~2× faster than the earlier
   `simplediffeq_test` measurement (~240 ms on a dev checkout) — consistent with the
   `0.2.263`→`0.2.273` release improvements. The JAX loop numbers are comparable to
   before (~290 vs ~310 ms), so the relative gap widened in Reactant's favour.

The two loss values across frameworks (Reactant −0.71462, JAX −0.62585) differ
because weight init and RNG differ; they are sanity signals of the same magnitude,
not an equivalence check (compute is dominated by the identically-structured ODE
integration).
