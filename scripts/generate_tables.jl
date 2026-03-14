#!/usr/bin/env julia
# Generate LaTeX table data from sPCE evaluation results.
#
# Reads:  examples/monod/results/spce_scores.jls
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
    scores_file = joinpath(root, "examples", "monod", "results", "spce_scores.jls")
    @assert isfile(scores_file) "spce_scores.jls not found at $scores_file"

    scores_dict = deserialize(scores_file)

    # Collect mean ± SEM per design
    rows = Tuple{String, Float64, Float64, Int}[]
    for name in DESIGN_ORDER
        key = name == "adaptive" ? "adaptive_scores" : "$(name)_scores"
        haskey(scores_dict, key) || continue
        scores = Float64.(scores_dict[key])
        n = length(scores)
        m = mean(scores)
        sem = std(scores) / sqrt(n)
        push!(rows, (name, m, sem, n))
    end

    # Find best (highest mean)
    best_idx = argmax([r[2] for r in rows])

    # Write table rows
    outfile = joinpath(root, "paper", "tables", "spce_table.tex")
    mkpath(dirname(outfile))
    open(outfile, "w") do io
        for (i, (name, m, sem, n)) in enumerate(rows)
            label = get(DESIGN_LABELS, name, name)
            score_str = @sprintf("\$%.2f \\pm %.2f\$", m, sem)
            if i == best_idx
                score_str = "\\mathbf{" * @sprintf("%.2f \\pm %.2f", m, sem) * "}"
                score_str = "\$" * score_str * "\$"
            end
            println(io, "$label & $score_str \\\\")
        end
    end

    println("Generated: $outfile")
    for (name, m, sem, n) in rows
        label = get(DESIGN_LABELS, name, name)
        @printf("  %-20s  mean = %8.4f ± %.4f  (n=%d)\n", label, m, sem, n)
    end
end

main()
