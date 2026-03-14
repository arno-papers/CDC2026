# Project overview for agents

This file provides guidance to AI agents working with code in this repository.

## Project Overview

This project is about deep adaptive experimental designs for dynamic systems.
Applied to bioreactor examples (Monod kinetics, with Haldane planned).
The goal is a research paper for the CDC conference.

## Repository Structure

```
src/                     # Shared Julia infrastructure
  common_core.jl         # RK4, positional encoding, I/O, load_static_designs() (CPU-safe)
  common.jl              # Reactant: integrate(), targeted_spce_loss(), train_policy()
  plotting.jl            # Shared plot styles, convex hulls

examples/
  monod/                 # Monod bioreactor example
    model.jl             # Dynamics, constants, sampling, policy, BIM functions
    train.jl             # Training (GPU)
    optimize_bim.jl      # Static BIM design optimization (CPU) → design_bim.jls
    optimize_static.jl   # Static sPCE design optimization (GPU) → design_spce.jls
    eval_spce.jl         # sPCE evaluation (GPU) → spce_scores.jls
    eval_posterior.jl     # Posterior evaluation (GPU) → posterior_results.jls
    plot_dynamics.jl     # Design comparison plot (CPU)
    plot_trajectories.jl # CPU rollout helpers, plot_design_comparison() (included by plot_dynamics.jl)
    plot_posterior.jl    # Posterior scatter plot (CPU)
    results/             # Flat results directory (git-tracked)
  haldane/               # Haldane bioreactor (substrate inhibition, uniform α prior)
    model.jl             # Dynamics, constants, uniform prior sampling, policy
    train.jl             # Training (GPU)
    plot_comparison.jl   # Comparison plot: no-inhibition vs inhibition (CPU)
    results/             # Flat results directory (git-tracked)

paper/                   # Research paper
  figures/               # Git-tracked, copied from example results by Snakefile
  tables/                # Auto-generated LaTeX table fragments

scripts/
  generate_tables.jl         # Generate paper/tables/spce_table.tex from results
  profile_budget.jl          # Find max ODE budget for GPU (search + probe modes)
  submit_profile_budget.sh   # Submit budget profiling to VSC

vsc/                     # Supercomputer scripts
  submit.sh              # Direct submission: EXAMPLE=monod TASK=train ./vsc/submit.sh
  train_remote.sh        # Idempotent wrapper called by Snakefile train rule
  job.slurm              # Parameterized SLURM template
  bootstrap_julia.sh     # One-time Julia install on cluster

reference/               # Reference material (gitignored)
```

## Include Path Strategy

All scripts use `joinpath(@__DIR__, ...)` for robust path resolution:
1. `model.jl` is included first (defines `dynamics()`, constants, sampling, observation model callbacks, policy)
2. `src/common.jl` (or `src/common_core.jl` for CPU-only) is included next

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
snakemake paper          # copy figures + generate tables + compile LaTeX

# Dry run (show what would run)
snakemake -n
```

### Individual scripts

```bash
julia --project=. examples/monod/train.jl
julia --project=. examples/monod/optimize_bim.jl
julia --project=. examples/monod/plot_dynamics.jl
julia --project=. examples/monod/plot_posterior.jl
julia --project=. scripts/generate_tables.jl
```

### Direct cluster submission (bypassing Snakemake)

```bash
EXAMPLE=monod TASK=train ./vsc/submit.sh
DRY_RUN=1 EXAMPLE=monod TASK=train ./vsc/submit.sh  # cost estimate only
```

### Budget profiling (GPU)

```bash
# Local GPU
julia --project=. scripts/profile_budget.jl example=monod

# VSC (batch job with full GPU memory)
EXAMPLE=monod ./scripts/submit_profile_budget.sh
```

See VSC_README.md for full details about the Flemish supercomputer.
