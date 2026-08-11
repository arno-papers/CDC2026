# JAX versus Reactant benchmark handoff

The deck's benchmark slide (frame 13) is a placeholder pending Sebastian's
validation. This is intentional: useful work exists, but the current evidence
does not yet support a clean head-to-head claim.

## Existing work

- Remote branch: `origin/smc/jax-reactant-benchmark`
- Commit: `9c0d03946f5d2447614684b19c3e85e84d6ab739`
- Author/date: Sebastian Micluța-Câmpeanu, 2026-07-14
- Main documentation: `benchmark/README.md`
- Orchestration: `benchmark/run_all.sh`
- Julia harness: `benchmark/julia/bench.jl`
- JAX harness: `benchmark/jax/benchmark.py`

Read it without switching branches:

```bash
git show 9c0d03946f5d2447614684b19c3e85e84d6ab739:benchmark/README.md
git diff --name-status 08c3d08cb4b3685936425af42c43b3e6f61b41df...9c0d03946f5d2447614684b19c3e85e84d6ab739
```

The branch pins Julia 1.12.6, Reactant 0.2.273, and JAX 0.10.2. Its matched
shape is `L=255`, `M=256`, `B=512`, float32, 14 experimental steps, and 50 RK4
substeps per experimental step: 262,656 trajectories per timed step. The draft
run used a shared NVIDIA RTX 4080 SUPER and configured the XLA client memory
fraction to 45% (`MEM_FRAC=0.45`); that is not a verified hard cap. This is
about 24 times smaller than the 6.25M-trajectory production microbatch quoted on
frame 7 of the deck.

The predeclared primary comparison is:

- Julia + Reactant, production hand-RK4 with `@trace for`;
- Python + JAX, hand-RK4 with `lax.scan`.

SimpleDiffEq and `lax.fori_loop` are secondary sensitivity variants, not rows
from which to choose the most favorable result after running.

## Preliminary values — do not present yet

| Variant | Reported median timed step | Reported “compile” |
|---|---:|---:|
| Reactant + SimpleDiffEq | 105.7 ms | 205 s |
| Reactant + hand RK4 | 113.5 ms | 200 s |
| JAX `lax.scan` | 289.0 ms | 2.8 s |
| JAX `lax.fori_loop` | 290.1 ms | 2.7 s |

If ultimately validated, the defensible title is: “At one matched workload,
Reactant has faster steady-state steps; JAX has much lower first-call latency.”
It is not a general “Julia is faster than JAX” result.

## Why these remain placeholders

- Raw logs and `results.txt` are gitignored and absent from the commit.
- JAX samples parameters and noise inside the timed differentiated function;
  Julia prepares and transfers its batch before the timer.
- Julia times `single_train_step!`, including the Adam update; JAX times only
  loss plus gradient.
- Julia's reported compile number includes three warmup calls; JAX reports the
  first call. They are not the same metric.
- The implementations use different initialization and random-number streams;
  cross-framework numerical equivalence has not been demonstrated.
- Deterministic-reduction settings differ.
- Fully unrolled JAX is excluded; document why if it remains excluded.
- Results cover one shape, one GPU, and one within-process timing series.

`benchmark/jax/check_equivalence.py` only compares two JAX implementations. It
is not evidence of cross-stack equivalence.

## Final metric contract

Agree on this contract before rerunning; do not revise it after seeing results.

1. Generate one deterministic Float32 fixture and map the same parameter,
   hypothesis, observation-noise, and policy tensors into both layouts.
2. Transfer that fixture to the device before any timed call.
3. Time loss plus reverse-mode gradient only. Exclude data generation, transfer,
   and optimizer updates from both implementations.
4. Explicitly synchronize the returned loss and gradient device values before
   stopping the timer. Compute gradient norms and equivalence diagnostics only
   after the timer; the diagnostic reduction is not part of either benchmark.
5. Define first-call latency as the first synchronized invocation, including
   compilation and execution.
6. After that call, run exactly five unmeasured warmups followed by 30 recorded
   steady-state calls.
7. Predeclare Float32 agreement thresholds: loss `rtol=1e-4, atol=1e-4`;
   gradient relative L2 error at most `1e-3`; gradient maximum absolute error at
   most `1e-3`. If these fail, investigate before timing. If a common fixture
   cannot be established, use “matched workload,” never “identical computation.”
8. Run at least five fresh processes per primary variant. Balance or randomize
   variant order. Keep every raw step time. Plot all process-level steady-state
   medians and first-call values, with an across-process median and range; do not
   imply a stable quartile estimate from the minimum five processes.
9. Prefer an exclusive, uncontended GPU. Otherwise record contention, memory
   cap, clocks/power state, CPU, OS, driver, CUDA, and allocator settings.

## Required harness changes

The committed harnesses cannot execute the final contract yet. Before rerunning:

1. Generate and archive one deterministic Float32 fixture containing policy
   parameters, target/nuisance hypotheses, initial states, observation noise,
   and every other stochastic input. Give both languages loaders for that exact
   artifact and record its SHA-256 hash.
2. Refactor the JAX timed function to accept the fixture instead of sampling or
   splitting random keys internally.
3. Add a Julia loss-and-gradient entry point that accepts the same fixture and
   does not update `TrainState` or Adam. Time that entry point, not
   `single_train_step!`.
4. Add framework-native synchronization of the returned loss and gradient to
   both timers. Compute loss/gradient comparison diagnostics after the timer.
5. Record the first synchronized invocation separately in a fresh process, then
   run exactly five warmups and 30 measured steady-state calls.
6. Make both harnesses emit every raw time, loss diagnostics, gradient
   diagnostics, memory measurement, and environment metadata as machine-readable
   rows.
7. Update `run_all.sh` to accept `OUTDIR`, `RUN_ID`, and `ORDER_SEED`, preserve
   rather than truncate artifacts, and randomize/balance variant order.

## Reproduction procedure

Create a commit-pinned worktree so branch movement cannot silently change the
benchmark:

```bash
git fetch --no-tags origin smc/jax-reactant-benchmark
BENCH_DIR=$(mktemp -d /tmp/cdc2026-benchmark.XXXXXX)
git worktree add "$BENCH_DIR" 9c0d03946f5d2447614684b19c3e85e84d6ab739
cd "$BENCH_DIR"
julia +1.12.6 --startup=no --project=benchmark/julia -e 'using Pkg; Pkg.instantiate()'
BENCH_PYTHON=3.12.8
uv python install "$BENCH_PYTHON"
(cd benchmark/jax && uv sync --frozen --python "$BENCH_PYTHON")
julia +1.12.6 --version
(cd benchmark/jax && uv run --frozen python --version)
```

Before measuring, diff the vendored implementations against the presentation's
final production code and the current JAX branch. Record and resolve any drift.

After implementing the harness changes above, the final five-process invocation
should be:

```bash
for rep in 01 02 03 04 05; do
  OUTDIR="benchmark/artifacts/final" \
  RUN_ID="process-${rep}" ORDER_SEED="$rep" \
  JULIA_CHANNEL=1.12.6 NWARM=5 NTIME=30 N_WARMUP=5 N_RUNS=30 \
  ./benchmark/run_all.sh
done
```

The current `benchmark/run_all.sh` does **not** support this command: it truncates
`benchmark/results.txt` and reuses `benchmark/logs/`. The required `OUTDIR`,
`RUN_ID`, and `ORDER_SEED` changes are a gate, not optional cleanup. A final
archive should look like:

```text
presentation/benchmark_artifacts/<date>-<commit>/
  metadata.txt
  fixture.sha256
  results.csv
  plot_benchmarks.jl
  figures/runtime.pdf
  figures/first_call.pdf
  logs/<variant>/<process>.log
```

`metadata.txt` must contain both repository commits, manifests/lockfile hashes,
language and package versions, OS/CPU/GPU, driver/CUDA, precision, `L/M/B`, loop
variants, warmup/timing counts, memory settings, and the exact commands.

## Sign-off gate

- [ ] Archive raw logs and a machine-readable result file with the deck release.
- [ ] Pin Julia, Reactant, JAX, CUDA, driver, GPU, precision, and commit hashes.
- [ ] Make the timed regions identical, including or excluding data generation,
      device transfer, optimizer update, and synchronization on both sides.
- [ ] Use one compile-time definition for both frameworks.
- [ ] Demonstrate matched forward loss and gradient on common deterministic
      inputs, or label the work only as a “matched workload.”
- [ ] Run at least five fresh processes per variant; show every process point and
      report the median/range, not only repeated steps within one process.
- [ ] Record steady-state step time, trajectories/s, first-call latency, and peak GPU
      memory.
- [ ] Explain the chosen loop variants and the exclusion of unrolled JAX.
- [ ] Have Sebastian confirm the final plots, numbers, wording, and commit.
- [ ] Replace the placeholder table only after every item above passes.
- [ ] Rename the slide title to an evidence-backed conclusion, update the speaker
      notes, rebuild the PDF twice, and visually inspect page 13 at projector
      resolution.

If the gate is still incomplete on presentation day, do not show empty numeric
slots to the room: comment out the results table on frame 13 and present that
frame as methodology only. There is no separate `live` build target.
