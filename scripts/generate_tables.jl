#!/usr/bin/env julia
# Generate LaTeX table data from sPCE and posterior evaluation results.
#
# Reads:  examples/monod/results/spce_scores.jls
#         examples/monod/results/posterior_results.jls
# Writes: paper/tables/spce_table.tex  (table rows only, no \begin{table} wrapper)
#
# Usage:
#   julia --project=. scripts/generate_tables.jl

using Serialization
using Statistics
using Printf

# Design labels matching the paper
const DESIGN_LABELS = Dict(
    "adaptive"    => "Adaptive policy",
    "static_spce" => "Static (sPCE-opt)",
    "static_std"  => "Static (BIM)",
)
const DESIGN_ORDER = ["adaptive", "static_spce", "static_std"]

function main()
    root = dirname(@__DIR__)
    results_dir = joinpath(root, "examples", "monod", "results")

    scores_file = joinpath(results_dir, "spce_scores.jls")
    posterior_file = joinpath(results_dir, "posterior_results.jls")
    @assert isfile(scores_file) "spce_scores.jls not found at $scores_file"
    @assert isfile(posterior_file) "posterior_results.jls not found at $posterior_file"

    scores_dict = deserialize(scores_file)
    posterior_dict = deserialize(posterior_file)
    true_μ = posterior_dict["true_mu_max"]
    true_K = posterior_dict["true_K_s"]

    # Collect sPCE scores and RMSE per design
    rows = NamedTuple{(:name, :spce_mean, :spce_sem, :rmse_μ, :rmse_K), Tuple{String, Float64, Float64, Float64, Float64}}[]
    for name in DESIGN_ORDER
        # sPCE scores
        key = name == "adaptive" ? "adaptive_scores" : "$(name)_scores"
        haskey(scores_dict, key) || continue
        scores = Float64.(scores_dict[key])
        m = mean(scores)
        sem = std(scores) / sqrt(length(scores))

        # Posterior RMSE
        pm = posterior_dict["posterior_means"][name]
        rmse_μ = sqrt(mean((pm[1, :] .- true_μ) .^ 2))
        rmse_K = sqrt(mean((pm[2, :] .- true_K) .^ 2))

        push!(rows, (; name, spce_mean=m, spce_sem=sem, rmse_μ, rmse_K))
    end

    # Find best (highest sPCE, lowest RMSE)
    best_spce = argmax([r.spce_mean for r in rows])
    best_rmse_μ = argmin([r.rmse_μ for r in rows])
    best_rmse_K = argmin([r.rmse_K for r in rows])

    # Write table rows
    outfile = joinpath(root, "paper", "tables", "spce_table.tex")
    mkpath(dirname(outfile))
    open(outfile, "w") do io
        for (i, r) in enumerate(rows)
            label = get(DESIGN_LABELS, r.name, r.name)

            # sPCE column
            if i == best_spce
                spce_str = "\$\\mathbf{" * @sprintf("%.2f \\pm %.2f", r.spce_mean, r.spce_sem) * "}\$"
            else
                spce_str = @sprintf("\$%.2f \\pm %.2f\$", r.spce_mean, r.spce_sem)
            end

            # RMSE columns (×10³ for readability)
            rmse_μ_scaled = r.rmse_μ * 1000
            rmse_K_scaled = r.rmse_K * 1000
            if i == best_rmse_μ
                rmse_μ_str = "\$\\mathbf{" * @sprintf("%.1f", rmse_μ_scaled) * "}\$"
            else
                rmse_μ_str = @sprintf("\$%.1f\$", rmse_μ_scaled)
            end
            if i == best_rmse_K
                rmse_K_str = "\$\\mathbf{" * @sprintf("%.1f", rmse_K_scaled) * "}\$"
            else
                rmse_K_str = @sprintf("\$%.1f\$", rmse_K_scaled)
            end

            println(io, "$label & $spce_str & $rmse_μ_str & $rmse_K_str \\\\")
        end
        println(io, "\\hline")
    end

    println("Generated: $outfile")
    for r in rows
        label = get(DESIGN_LABELS, r.name, r.name)
        @printf("  %-20s  sPCE = %5.2f ± %.2f   RMSE(μ) = %.4f   RMSE(K) = %.4f\n",
                label, r.spce_mean, r.spce_sem, r.rmse_μ, r.rmse_K)
    end
end

main()
