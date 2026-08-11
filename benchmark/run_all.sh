#!/usr/bin/env bash
# ============================================================================
# Fair head-to-head: JAX vs Julia/Reactant on the Monod targeted-sPCE gradient
# step. Runs every variant SEQUENTIALLY (never concurrently) so neither side is
# starved of the shared GPU, at a single MATCHED (L, M, B) config, float32,
# N_STEPS=14, N_SUBSTEPS=50, full materialization (no hypothesis/batch chunking).
#
# Both harnesses print a `RESULT ...` line; we grep those into results.txt.
#
# Config (override via env): L=255 M=256 B=512 NWARM/NTIME for Julia,
# N_WARMUP/N_RUNS for JAX.
#
# Memory caps assume a contended GPU (~8 GB free of 16). Adjust MEM_FRAC if you
# have the card to yourself.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

L=${L:-255}
M=${M:-256}
B=${B:-512}
MEM_FRAC=${MEM_FRAC:-0.45}     # fraction of TOTAL GPU memory each process may use
JULIA_CHANNEL=${JULIA_CHANNEL:-release}

LOGDIR=logs
mkdir -p "$LOGDIR"
RESULTS=results.txt
: > "$RESULTS"

echo "############################################################"
echo "# Matched config: L=$L  M=$M  B=$B  (float32, N_STEPS=14, N_SUBSTEPS=50)"
echo "# GPU mem cap per process: ${MEM_FRAC} of total"
echo "############################################################"

# ---------------------------------------------------------------------------
# Julia / Reactant : hand-RK4 and SimpleDiffEq, both in a @trace for loop.
# ---------------------------------------------------------------------------
for INTEG in hand sd; do
  echo ">>> Julia/Reactant  integ=$INTEG"
  log="$LOGDIR/julia_${INTEG}.log"
  INTEG=$INTEG L=$L M=$M BMICRO=$B BACKEND=gpu \
    XLA_REACTANT_GPU_PREALLOCATE=false \
    XLA_REACTANT_GPU_MEM_FRACTION=$MEM_FRAC \
    julia +"$JULIA_CHANNEL" --startup=no --project=julia julia/bench.jl 2>&1 | tee "$log"
  grep '^RESULT' "$log" >> "$RESULTS" || true
done

# ---------------------------------------------------------------------------
# JAX : scan / fori substep loops (the loop-based analogues of Reactant's
# @trace for). Full materialization to match Reactant: --hyp-chunk 0 (full
# hypothesis axis) --batch-block 0 (single block).
#
# The package also has a fully-`unrolled` loop, intentionally NOT run here: it
# has no Reactant counterpart (deep unroll blows up tracing) and does not scale
# with substep depth. See README.md.
# ---------------------------------------------------------------------------
for LOOP in scan fori; do
  echo ">>> JAX  substep=$LOOP"
  log="$LOGDIR/jax_${LOOP}.log"
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC \
    uv run --project=jax python jax/benchmark.py \
      --L "$L" --M "$M" --B "$B" \
      --mode full-batch --hyp-chunk 0 --batch-block 0 \
      --substep-loop "$LOOP" 2>&1 | tee "$log"
  grep '^RESULT' "$log" >> "$RESULTS" || true
done

echo "############################################################"
echo "# Collected results ($RESULTS):"
cat "$RESULTS"
echo "############################################################"
