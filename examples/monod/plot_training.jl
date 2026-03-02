#!/usr/bin/env julia
# Plot training loss curve from checkpoint.
#
# Usage:
#   julia --project=. examples/monod/plot_training.jl [checkpoint=results]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "args.jl"))

using Plots
using Serialization

checkpoint = parse_kwarg(ARGS, "checkpoint"; default=joinpath(@__DIR__, "results"))
results_dir = joinpath(@__DIR__, "results")

r = load_results(isdir(checkpoint) ? checkpoint : dirname(checkpoint))
loss_history = r.loss_history

println("Loaded $(length(loss_history)) loss values")

p = plot(loss_history;
    xlabel = "Iteration",
    ylabel = "Targeted sPCE Loss",
    title = "Training Loss",
    label = "loss",
    linewidth = 2,
)

out_png = joinpath(results_dir, "plot_training_loss.png")
savefig(p, out_png)
println("Saved: $out_png")
