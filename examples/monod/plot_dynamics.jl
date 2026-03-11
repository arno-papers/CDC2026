using Plots, Serialization
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

include(joinpath(@__DIR__, "plot_trajectories.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")
    file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
    @assert isfile(file) "Checkpoint not found: $file. Run training first."

    ckpt = deserialize(file)
    rng = Random.MersenneTwister(42)
    plot_trajectories(policy, ckpt["parameters"], ckpt["states"]; rng,
        outfile=joinpath(results_dir, "plot_dynamics.png"))
end
