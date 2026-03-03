# Project overview for agents

This file provides guidance to AI agents working with code in this repository.

## Project Overview

This project is about deep adaptive experimental designs for dynamic systems.
Applied to bioreactor examples (Monod kinetics, with Haldane planned).
The goal is a research paper for the CDC conference.

## Repository Structure

```
src/                     # Shared Julia infrastructure
  args.jl                # CLI argument parsing
  common_core.jl         # RK4, positional encoding, diagnostics, I/O (CPU-safe)
  common.jl              # Reactant: integrate(), targeted_spce_loss(), train_policy()
  plotting.jl            # Shared plot styles, histograms, t-tests

examples/
  monod/                 # Monod bioreactor example
    model.jl             # Dynamics, constants, sampling, policy, BIM functions
    train.jl             # Training (GPU)
    optimize_bim.jl      # Static BIM design optimization (CPU, supports cheating=true)
    eval_spce.jl         # sPCE evaluation (GPU)
    optimize_static.jl   # Static sPCE design optimization (GPU)
    eval_posterior.jl     # Posterior evaluation (GPU)
    plot_training.jl     # Training loss curve (CPU)
    plot_trajectories.jl # Policy rollout visualization (CPU)
    results/             # Flat results directory (git-tracked)
  haldane/               # Haldane bioreactor (substrate inhibition, spike-and-slab prior)
    model.jl             # Dynamics, constants, spike-and-slab sampling, policy
    train.jl             # Training (GPU)
    plot_comparison.jl   # Comparison plot: no-inhibition vs inhibition (CPU)
    results/             # Flat results directory (git-tracked)

paper/                   # Research paper (renamed from latex/)
  figures/               # Git-tracked, copied from example results by Snakefile

vsc/                     # Supercomputer scripts
  submit.sh              # Direct submission: EXAMPLE=monod TASK=train ./vsc/submit.sh
  train_remote.sh        # Idempotent wrapper called by Snakefile train rule
  job.slurm              # Parameterized SLURM template
  bootstrap_julia.sh     # One-time Julia install on cluster

reference/               # Reference material (gitignored)
```

## Include Path Strategy

All scripts use `joinpath(@__DIR__, ...)` for robust path resolution:
1. `model.jl` is included first (defines `dynamics()`, constants, sampling, policy)
2. `src/common.jl` (or `src/common_core.jl` for CPU-only) is included next
3. `src/args.jl` for CLI argument parsing

## Implementation

Julia 1.12 with Lux.jl, Reactant.jl, and Enzyme.jl for training.

## Running the pipeline

The project uses Snakemake. Training runs on the VSC cluster; everything else runs locally.

```bash
# Full pipeline (two-pass: training is remote)
snakemake --keep-going   # BIM runs locally, training submits to cluster, stops
# ... hours later ...
snakemake                # sees training done, syncs checkpoint, runs eval/plots locally

# Paper
snakemake paper          # copy figures + compile LaTeX

# Dry run (show what would run)
snakemake -n
```

### Individual scripts

```bash
julia --project=. examples/monod/train.jl n_iters=1000 grad_accum=16
julia --project=. examples/monod/optimize_bim.jl
julia --project=. examples/monod/optimize_bim.jl cheating=true
julia --project=. examples/monod/plot_training.jl
julia --project=. examples/monod/plot_trajectories.jl
```

### Direct cluster submission (bypassing Snakemake)

```bash
EXAMPLE=monod TASK=train ./vsc/submit.sh
DRY_RUN=1 EXAMPLE=monod TASK=train ./vsc/submit.sh  # cost estimate only
```

See VSC_README.md for full details about the Flemish supercomputer.
