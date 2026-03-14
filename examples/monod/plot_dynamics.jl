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

    # Load static designs (ordered by performance: sPCE-opt, then BIM)
    static_designs = Pair{String, Vector{Float32}}[]

    spce_file = joinpath(results_dir, "spce_static_design.jls")
    if isfile(spce_file)
        spce_data = deserialize(spce_file)
        push!(static_designs, "static_spce" => Float32.(spce_data["design"]))
    end

    bim_file = joinpath(results_dir, "bim_std_design.jls")
    if isfile(bim_file)
        bim_data = deserialize(bim_file)
        push!(static_designs, "static_std" => Float32.(bim_data["static_design"]))
    end

    rng = Random.MersenneTwister(42)
    plot_design_comparison(policy, ckpt["parameters"], ckpt["states"],
        static_designs; rng,
        outfile=joinpath(results_dir, "plot_dynamics.png"))
end
