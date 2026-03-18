#!/usr/bin/env julia
# Plot posterior evaluation results (CPU-only, loads posterior_results.jls).
#
# Produces:
#   plot_posterior.png — scatter with 95% confidence hulls (paper figure)
#
# Usage:
#   julia --project=. examples/monod/plot_posterior.jl

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

using Plots
using Printf
using Serialization
using Statistics

# ============================================================================
#  Main
# ============================================================================

seed       = 0
Random.seed!(seed)
results_dir = joinpath(@__DIR__, "results")

results_file = joinpath(results_dir, "posterior_results.jls")
@assert isfile(results_file) "posterior_results.jls not found. Run eval_posterior.jl first."
results = deserialize(results_file)

all_post_means = results["posterior_means"]
true_μ   = results["true_mu_max"]
true_K   = results["true_K_s"]
n_trials = results["n_trials"]
N_post   = results["N_post"]

# ---- Plot 1: Posterior mean scatter ----
p = plot(; xlabel = "Posterior mean μ_max", ylabel = "Posterior mean K_s",
           title = "",
           legend = :outertopright, size = (750, 600))

for name in reverse(DESIGN_ORDER)
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

println("Done. Outputs in: $results_dir")
