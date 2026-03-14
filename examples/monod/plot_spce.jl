#!/usr/bin/env julia
# Plot sPCE evaluation results (CPU-only, loads spce_scores.jls).
#
# Produces: plot_spce_histograms.png
#
# Usage:
#   julia --project=. examples/monod/plot_spce.jl

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

using Serialization

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")

    scores_file = joinpath(results_dir, "spce_scores.jls")
    @assert isfile(scores_file) "spce_scores.jls not found. Run eval_spce.jl first."

    scores_dict = deserialize(scores_file)
    designs = extract_designs(scores_dict)
    plot_spce_histograms(designs;
        output_path = joinpath(results_dir, "plot_spce_histograms.png"))

    println("Done. Outputs in: $results_dir")
end
