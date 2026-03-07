#!/usr/bin/env julia
# Optimize static BIM design (ForwardDiff + gradient).
#
# Usage:
#   julia --project=. examples/monod/optimize_bim.jl

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Dates
using Plots
using Printf
using Random
using Serialization

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    seed           = 0
    n_prior_opt    = 512
    n_prior_report = 1024
    n_substeps     = N_SUBSTEPS
    grad_iters     = 300
    grad_restarts  = 1
    lr_max         = 0.1
    lr_min         = 0.001

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    println("\n=== Bayesian D-optimal Static Design (ForwardDiff + gradient) ===")
    println("results_dir       = $results_dir")
    println("seed              = $seed")
    println("n_prior_opt       = $n_prior_opt")
    println("n_prior_report    = $n_prior_report")
    println("grad              = $grad_iters iters x $grad_restarts restarts")
    println("lr                = [$lr_min, $lr_max] cosine")
    println("BIM: 3x3 FIM + prior, Schur complement -> 2x2")
    @printf("Prior precision: mu_max=%.1f  K_s=%.1f  Cx0=%.1f\n", PRIOR_PREC...)
    flush(stdout)

    # ---- Optimize static design ----
    println("\n--- Optimizing static design ---")
    flush(stdout)
    rng_opt = MersenneTwister(seed)

    prior_opt = draw_prior_samples(rng_opt, n_prior_opt)
    static_design, static_obj = optimize_static_design_grad(
        prior_opt; n_iters=grad_iters, lr_max=lr_max, lr_min=lr_min,
        n_restarts=grad_restarts, n_substeps=n_substeps,
        results_dir=results_dir, prefix="bim_std")

    rng_report = MersenneTwister(seed + 1)
    prior_report = draw_prior_samples(rng_report, n_prior_report)
    static_obj_report = bim_logdet(Float64.(static_design), prior_report; n_substeps=n_substeps)

    println("\nStatic design optimized.")
    @printf("  Train BIM logdet:  %.5f\n", static_obj)
    @printf("  Report BIM logdet: %.5f\n", static_obj_report)
    println("  Design: [", join(round.(static_design; digits=3), ", "), "]")
    flush(stdout)

    # ---- Save ----
    serialize(joinpath(results_dir, "bim_std_design.jls"), Dict(
        "static_design"     => static_design,
        "static_obj_opt"    => static_obj,
        "static_obj_report" => static_obj_report,
    ))

    println("\nDone. Outputs in: $results_dir")
end
