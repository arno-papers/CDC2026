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

function generate_monod_table(root)
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

function generate_dcmotor_table(root)
    results_dir = joinpath(root, "examples", "dcmotor", "results")
    comparison_file = joinpath(results_dir, "comparison_results.jls")
    @assert isfile(comparison_file) "comparison_results.jls not found at $comparison_file"

    d = deserialize(comparison_file)
    true_k = d["true_k"]
    true_J = d["true_J"]

    # Build rows: (name, spce_mean, spce_sem, rmse_k, rmse_J, time_str)
    rows = []
    for (name, label, scores_key, pm_key, time_key) in [
        ("spce", "Adaptive (sPCE)", "spce_scores_spce", "post_means_spce", "spce_step_times"),
        ("bim",  "Adaptive (BIM)",  "spce_scores_bim",  "post_means_bim",  "bim_step_times"),
    ]
        scores = Float64.(d[scores_key])
        m = mean(scores)
        sem = std(scores) / sqrt(length(scores))

        pm = d[pm_key]
        rmse_k = sqrt(mean((pm[1, :] .- true_k) .^ 2))
        rmse_J = sqrt(mean((pm[2, :] .- true_J) .^ 2))

        med_time = median(vec(d[time_key]))
        if med_time < 0.001
            time_str = @sprintf("\$%.0f\\,\\mu\$s", med_time * 1e6)
        elseif med_time < 1.0
            time_str = @sprintf("\$%.1f\\,\$ms", med_time * 1e3)
        else
            time_str = @sprintf("\$%.1f\\,\$s", med_time)
        end

        push!(rows, (; name, label, spce_mean=m, spce_sem=sem, rmse_k, rmse_J, time_str))
    end

    best_spce = argmax([r.spce_mean for r in rows])
    best_rmse_k = argmin([r.rmse_k for r in rows])
    best_rmse_J = argmin([r.rmse_J for r in rows])

    outfile = joinpath(root, "paper", "tables", "dcmotor_table.tex")
    mkpath(dirname(outfile))
    open(outfile, "w") do io
        for (i, r) in enumerate(rows)
            # sPCE column
            if i == best_spce
                spce_str = "\$\\mathbf{" * @sprintf("%.2f \\pm %.2f", r.spce_mean, r.spce_sem) * "}\$"
            else
                spce_str = @sprintf("\$%.2f \\pm %.2f\$", r.spce_mean, r.spce_sem)
            end

            # RMSE columns (×10³)
            rmse_k_scaled = r.rmse_k * 1000
            rmse_J_scaled = r.rmse_J * 1000
            if i == best_rmse_k
                rmse_k_str = "\$\\mathbf{" * @sprintf("%.1f", rmse_k_scaled) * "}\$"
            else
                rmse_k_str = @sprintf("\$%.1f\$", rmse_k_scaled)
            end
            if i == best_rmse_J
                rmse_J_str = "\$\\mathbf{" * @sprintf("%.1f", rmse_J_scaled) * "}\$"
            else
                rmse_J_str = @sprintf("\$%.1f\$", rmse_J_scaled)
            end

            println(io, "$(r.label) & $spce_str & $rmse_k_str & $rmse_J_str & $(r.time_str) \\\\")
        end
        println(io, "\\hline")
    end

    println("Generated: $outfile")
    for r in rows
        @printf("  %-20s  sPCE = %5.2f ± %.2f   RMSE(k) = %.4f   RMSE(J) = %.5f   time = %s\n",
                r.label, r.spce_mean, r.spce_sem, r.rmse_k, r.rmse_J, r.time_str)
    end
end

function main()
    root = dirname(@__DIR__)

    generate_monod_table(root)

    # DC motor table (optional — only if results exist)
    dcmotor_file = joinpath(root, "examples", "dcmotor", "results", "comparison_results.jls")
    if isfile(dcmotor_file)
        println()
        generate_dcmotor_table(root)
    end
end

main()
