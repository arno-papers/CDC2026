#!/usr/bin/env julia
# BIM (Bayesian Information Matrix) design optimization for DC motor.
# CPU-only, uses ForwardDiff for gradients through the ODE.
#
# Usage:
#   julia --project=. examples/dcmotor/optimize_bim.jl

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    rng = Random.MersenneTwister(0)
    n_prior = 200
    prior_samples = draw_prior_samples(rng, n_prior)

    println("\n=== DC Motor: BIM Design Optimization ===")
    println("Prior samples: $n_prior")
    println("N_STEPS=$N_STEPS, DT=$DT, N_SUBSTEPS=$N_SUBSTEPS")
    println()

    # Standard BIM
    println("--- Standard BIM (average over full prior) ---")
    flush(stdout)
    best_design, best_score = optimize_static_design_grad(prior_samples;
        n_iters=300, n_restarts=4, results_dir=results_dir, prefix="bim_std")

    serialize(joinpath(results_dir, "bim_std_design.jls"), Dict(
        "static_design" => best_design,
        "bim_score" => best_score,
        "n_prior" => n_prior,
    ))

    println("\nBest BIM design: [$(join(round.(best_design; digits=3), ", "))]")
    @printf("BIM score: %.5f\n", best_score)
    println("Saved to: $(results_dir)/bim_std_design.jls")
end
