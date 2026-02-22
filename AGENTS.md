# Project overview for agents

This file provides guidance to AI agents working with code in this repository.

## Project Overview

This project is about deep adaptive experimental designs for dynamic systems.
For now, applied to a bioreactor.
The goal is a research paper for CDC conference.

## latex folder

Contains the research paper.

## reference

Contains reference material regarding deep adaptive designs and experimental design for dynamic systems, in particular the bioreactor.

## implementation

Julia 1.12 with Lux.jl, Reactant.jl, and Enzyme.jl for training.

## Running local

```bash
julia --project training.jl [--n_iters 1000] [--grad_accum 16] [--results_dir results/my_run]
```
## Running supercomputer

See vsc_scripts for info about Flemish supercomputer.

## Post-training analysis
```bash
julia --project plot_diagnostics.jl   # Convergence metrics (reads diagnostics.jls)
julia --project plot_trajectories.jl  # Policy rollout visualization (reads checkpoint.jls)
julia --project compare_static_bim.jl # Bayesian information matrix evaluation
```
