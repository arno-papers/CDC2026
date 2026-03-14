#!/usr/bin/env julia
# Plot posterior evaluation results (CPU-only, loads posterior_results.jls).
#
# Produces:
#   plot_posterior.png          — scatter with 95% confidence hulls (paper figure)
#   plot_posterior_std.png      — boxplots of posterior std per design
#   plot_posterior_contours.png — 5-trial contour panels
#
# Usage:
#   julia --project=. examples/monod/plot_posterior.jl

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

using Plots
using StatsPlots
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU observation generation
# ============================================================================

function generate_observations(rng, theta_T, sigma, Cx0, design;
                                n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    return Float32[cs[k] + Float64(sigma) * randn(rng) for k in 1:N_STEPS]
end

# ============================================================================
#  CPU log-likelihood (for posterior contour plots)
# ============================================================================

function log_likelihood(observations::Vector{Float64}, theta_T, sigma, Cx0,
                         design::AbstractVector; n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    σ² = Float64(sigma)^2
    ll = 0.0
    for k in 1:N_STEPS
        residual = observations[k] - cs[k]
        ll -= 0.5 * (residual^2 / σ² + log(σ²))
    end
    return ll
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    n_substeps = N_SUBSTEPS
    seed       = 0
    N_post     = 5000

    results_dir = joinpath(@__DIR__, "results")

    results_file = joinpath(results_dir, "posterior_results.jls")
    @assert isfile(results_file) "posterior_results.jls not found. Run eval_posterior.jl first."
    results = deserialize(results_file)

    all_post_means = results["posterior_means"]
    all_post_vars  = results["posterior_vars"]
    all_ess        = results["ess"]
    true_μ   = results["true_mu_max"]
    true_K   = results["true_K_s"]
    true_σ   = results["true_sigma"]
    true_Cx0 = results["true_Cx0"]
    n_trials = results["n_trials"]
    N_post   = results["N_post"]
    θT       = Float32[true_μ, true_K]

    ps_cpu, st_cpu, _ = load_checkpoint_cpu(results_dir)
    static_designs = load_static_designs(results_dir)

    # ---- Plot 1: Posterior mean scatter ----
    p = plot(; xlabel = "Posterior mean μ_max", ylabel = "Posterior mean K_s",
               title = @sprintf("Posterior means (%d trials, N=%d)", n_trials, N_post),
               legend = :outertopright, size = (750, 600))

    for name in DESIGN_ORDER
        haskey(all_post_means, name) || continue
        style = get(DESIGN_STYLES, name, (label = name, color = :black))
        scatter!(p, [NaN], [NaN]; color = style.color, ms = 3, msw = 0,
                 label = style.label)
    end
    for name in reverse(DESIGN_ORDER)
        haskey(all_post_means, name) || continue
        style = get(DESIGN_STYLES, name, (label = name, color = :black))
        pm = all_post_means[name]
        scatter!(p, pm[1, :], pm[2, :];
                 color = style.color, alpha = 0.4, ms = 3, msw = 0,
                 label = "")
    end
    for name in reverse(DESIGN_ORDER)
        haskey(all_post_means, name) || continue
        style = get(DESIGN_STYLES, name, (label = name, color = :black))
        pm = all_post_means[name]
        hx, hy = confidence_hull(pm[1, :], pm[2, :])
        plot!(p, Shape(hx, hy); fillcolor = style.color, fillalpha = 0.1,
              linecolor = style.color, lw = 2, linealpha = 1.0, label = "")
    end

    scatter!(p, [true_μ], [true_K];
             color = :black, shape = :xcross, ms = 10, msw = 3, label = "True value")

    save_plot(p, joinpath(results_dir, "plot_posterior.png"))

    # ---- Plot 2: Posterior std boxplots ----
    p_std = plot(layout = (1, 2), size = (800, 400),
                 title = ["Posterior std μ_max" "Posterior std K_s"])

    for (param_idx, param_name) in enumerate(["μ_max", "K_s"])
        labels = String[]
        data_vecs = Vector{Float64}[]
        colors = []
        for name in DESIGN_ORDER
            haskey(all_post_vars, name) || continue
            style = get(DESIGN_STYLES, name, (label = name, color = :black))
            stds = sqrt.(Float64.(all_post_vars[name][param_idx, :]))
            push!(labels, style.label)
            push!(data_vecs, stds)
            push!(colors, style.color)
        end
        for (i, (lab, stds, col)) in enumerate(zip(labels, data_vecs, colors))
            boxplot!(p_std[param_idx], [lab], stds;
                     color = col, fillalpha = 0.4, lw = 1.5,
                     outliers = true, label = "")
        end
    end
    save_plot(p_std, joinpath(results_dir, "plot_posterior_std.png"))

    # ---- Plot 3: Posterior contours for 5 sample trials (CPU) ----
    println("Computing posterior contours for 5 sample trials (CPU)...")
    flush(stdout)

    n_contour_trials = 5
    rng_contour = MersenneTwister(seed + 99)

    θ_prior = sample_θ_full(rng_contour, N_post)

    p_contour = plot(layout = (1, n_contour_trials),
                     size = (250 * n_contour_trials, 250),
                     xlabel = "μ_max", ylabel = "K_s")

    for trial in 1:n_contour_trials
        rng_trial = MersenneTwister(seed + trial)

        for name in DESIGN_ORDER
            haskey(all_post_means, name) || continue
            style = get(DESIGN_STYLES, name, (label = name, color = :black))

            if name == "adaptive"
                design = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng_trial,
                            θT, true_σ; Cx0=true_Cx0, n_substeps=n_substeps)
            else
                design = Dict(static_designs)[name]
            end

            obs_rng = MersenneTwister(seed + (name == "adaptive" ? 0 : findfirst(p -> p.first == name, static_designs)) * 1000 + trial)
            obs = generate_observations(obs_rng, θT, true_σ, true_Cx0, design; n_substeps=n_substeps)

            ll = zeros(Float64, N_post)
            for j in 1:N_post
                θT_j = Float64[θ_prior[1, j], θ_prior[2, j]]
                σ_j = Float64(θ_prior[3, j])
                Cx0_j = Float64(θ_prior[4, j])
                ll[j] = log_likelihood(Float64.(obs), θT_j, σ_j, Cx0_j,
                                        Float64.(design); n_substeps=n_substeps)
            end
            ll_max = maximum(ll)
            w = exp.(ll .- ll_max)
            w ./= sum(w)

            n_resample = 500
            cum_w = cumsum(w)
            idx = [searchsortedfirst(cum_w, rand(rng_contour)) for _ in 1:n_resample]
            idx = clamp.(idx, 1, N_post)

            lab = trial == 1 ? style.label : ""
            scatter!(p_contour[trial], Float64.(θ_prior[1, idx]), Float64.(θ_prior[2, idx]);
                     color = style.color, alpha = 0.25, ms = 2, msw = 0,
                     label = lab)
        end

        scatter!(p_contour[trial], [Float64(true_μ)], [Float64(true_K)];
                 color = :black, shape = :xcross, ms = 8, msw = 2,
                 label = (trial == 1 ? "True" : ""))

        plot!(p_contour[trial]; title = "Trial $trial")
    end
    save_plot(p_contour, joinpath(results_dir, "plot_posterior_contours.png"))

    println("Done. Outputs in: $results_dir")
end
