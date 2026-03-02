#!/usr/bin/env julia
# Optimize static BIM design (ForwardDiff + gradient).
#
# Usage:
#   julia --project=. examples/monod/optimize_bim.jl [checkpoint=...]
#   julia --project=. examples/monod/optimize_bim.jl cheating=true

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "args.jl"))

using Dates
using Plots
using Printf
using Random
using Serialization

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    checkpoint    = parse_kwarg(ARGS, "checkpoint"; default=joinpath(@__DIR__, "results"))
    output_dir_arg = parse_kwarg(ARGS, "output_dir"; default=nothing)
    seed          = parse_int(ARGS, "seed"; default=0)
    n_prior_opt   = parse_int(ARGS, "n_prior_opt"; default=512)
    n_prior_report = parse_int(ARGS, "n_prior_report"; default=1024)
    n_substeps    = parse_int(ARGS, "n_substeps"; default=N_SUBSTEPS)
    grad_iters    = parse_int(ARGS, "grad_iters"; default=300)
    grad_restarts = parse_int(ARGS, "grad_restarts"; default=1)
    lr_max        = parse_float64(ARGS, "lr_max"; default=0.1)
    lr_min        = parse_float64(ARGS, "lr_min"; default=0.001)
    cheating      = parse_bool(ARGS, "cheating"; default=false)
    true_mu_max   = parse_float64(ARGS, "true_mu_max"; default=Float64(μ_max_lo + μ_max_hi) / 2)
    true_K_s      = parse_float64(ARGS, "true_K_s"; default=Float64(K_s_lo + K_s_hi) / 2)
    true_sigma    = parse_float64(ARGS, "true_sigma"; default=Float64(σ_lo + σ_hi) / 2)
    n_cx0_opt     = parse_int(ARGS, "n_cx0_opt"; default=512)

    results_dir = joinpath(@__DIR__, "results")

    prefix = cheating ? "bim_cheat" : "bim_std"
    output_dir = output_dir_arg !== nothing ? output_dir_arg : results_dir
    mkpath(output_dir)

    mode_str = cheating ? "CHEATING (known kinetics)" : "standard"
    println("\n=== Bayesian D-optimal Static Design (ForwardDiff + gradient) [$mode_str] ===")
    println("output_dir        = $output_dir")
    println("seed              = $seed")
    if cheating
        @printf("true_mu_max       = %.4f\n", true_mu_max)
        @printf("true_K_s          = %.4f\n", true_K_s)
        @printf("true_sigma        = %.4f\n", true_sigma)
        println("n_cx0_opt         = $n_cx0_opt")
    else
        println("n_prior_opt       = $n_prior_opt")
        println("n_prior_report    = $n_prior_report")
    end
    println("grad              = $grad_iters iters x $grad_restarts restarts")
    println("lr                = [$lr_min, $lr_max] cosine")
    println("BIM: 3x3 FIM + prior, Schur complement -> 2x2")
    @printf("Prior precision: mu_max=%.1f  K_s=%.1f  Cx0=%.1f\n", PRIOR_PREC...)
    flush(stdout)

    # ---- Optimize static design ----
    println("\n--- Optimizing static design ---")
    flush(stdout)
    rng_opt = MersenneTwister(seed)

    local static_design, static_obj, static_obj_report

    if cheating
        theta_T_true = Float32[Float32(true_mu_max), Float32(true_K_s)]
        sigma_true   = Float32(true_sigma)
        cx0_opt = sample_cx0(rng_opt, n_cx0_opt)
        objective = ξ -> bim_logdet_cheating(ξ, theta_T_true, sigma_true, cx0_opt;
                                              n_substeps=n_substeps)
        static_design, static_obj = optimize_design_grad(objective;
            n_iters=grad_iters, lr_max=lr_max, lr_min=lr_min,
            n_restarts=grad_restarts, results_dir=output_dir, prefix=prefix)

        rng_report = MersenneTwister(seed + 1)
        cx0_report = sample_cx0(rng_report, n_prior_report)
        static_obj_report = bim_logdet_cheating(Float64.(static_design),
            theta_T_true, sigma_true, cx0_report; n_substeps=n_substeps)
    else
        prior_opt = draw_prior_samples(rng_opt, n_prior_opt)
        static_design, static_obj = optimize_static_design_grad(
            prior_opt; n_iters=grad_iters, lr_max=lr_max, lr_min=lr_min,
            n_restarts=grad_restarts, n_substeps=n_substeps,
            results_dir=output_dir, prefix=prefix)

        rng_report = MersenneTwister(seed + 1)
        prior_report = draw_prior_samples(rng_report, n_prior_report)
        static_obj_report = bim_logdet(Float64.(static_design), prior_report; n_substeps=n_substeps)
    end

    println("\nStatic design optimized.")
    @printf("  Train BIM logdet:  %.5f\n", static_obj)
    @printf("  Report BIM logdet: %.5f\n", static_obj_report)
    println("  Design: [", join(round.(static_design; digits=3), ", "), "]")
    flush(stdout)

    # ---- Save ----
    serialize(joinpath(output_dir, "$(prefix)_design.jls"), Dict(
        "static_design"     => static_design,
        "static_obj_opt"    => static_obj,
        "static_obj_report" => static_obj_report,
    ))

    println("\nDone. Outputs in: $output_dir")
end
