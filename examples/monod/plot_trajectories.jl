using Plots, Serialization

function rollout_policy_cpu(model, ps_cpu, st_cpu; theta, u0)
    u = reshape(u0, 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    traj = zeros(Float32, N_STEPS + 1, 3)
    designs = zeros(Float32, N_STEPS)
    traj[1, :] .= vec(u)

    st_local = st_cpu
    for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        Q_in = Float32(action[1])
        designs[step] = Q_in

        u = integrate_cpu(u, reshape(theta, 2, 1), Q_in, DT, N_SUBSTEPS)
        traj[step + 1, :] .= vec(u)

        y_obs = u[1, 1]
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = Q_in
    end

    return traj, designs
end

function plot_trajectories(model, train_state; rng, n_samples=20, outfile=joinpath(@__DIR__, "results", "plot_trajectories.png"))
    _to_cpu(x) = x
    _to_cpu(x::AbstractArray) = collect(x)
    _to_cpu(x::NamedTuple) = map(_to_cpu, x)
    _to_cpu(x::Tuple) = map(_to_cpu, x)
    plot_trajectories(model, _to_cpu(train_state.parameters), _to_cpu(train_state.states); rng, n_samples, outfile)
end

function plot_trajectories(model, ps_cpu::NamedTuple, st_cpu::NamedTuple; rng, n_samples=20, outfile=joinpath(@__DIR__, "results", "plot_trajectories.png"))

    t_states = 0:N_STEPS
    t_designs = 1:N_STEPS

    p1 = plot(title = "Substrate Cs", xlabel = "Step", ylabel = "Cs")
    p2 = plot(title = "Biomass Cx", xlabel = "Step", ylabel = "Cx")
    p3 = plot(title = "Volume V", xlabel = "Step", ylabel = "V")
    p4 = plot(title = "Design Qin", xlabel = "Step", ylabel = "Qin")

    for i in 1:n_samples
        theta = Float32[
            mu_max_lo + (mu_max_hi - mu_max_lo) * rand(rng),
            K_s_lo + (K_s_hi - K_s_lo) * rand(rng),
        ]
        Cx0 = Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng)
        u0 = Float32[3.0, Cx0, 7.0]
        traj, designs = rollout_policy_cpu(model, ps_cpu, st_cpu; theta=theta, u0=u0)

        label = i == 1 ? "samples" : ""
        plot!(p1, t_states, traj[:, 1]; alpha=0.5, color=:blue, label=label)
        plot!(p2, t_states, traj[:, 2]; alpha=0.5, color=:red, label=label)
        plot!(p3, t_states, traj[:, 3]; alpha=0.5, color=:green, label=label)
        plot!(p4, t_designs, designs; alpha=0.5, color=:black, label=label, marker=:circle, markersize=2)
    end

    plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (900, 700))
    savefig(plt, outfile)
    println("Saved trajectories plot: $outfile")

    return plt
end

# Standalone usage
if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "model.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

    results_dir = joinpath(@__DIR__, "results")
    file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
    @assert isfile(file) "Checkpoint not found: $file. Run training first."

    ckpt = deserialize(file)
    rng = Random.MersenneTwister(42)
    plot_trajectories(policy, ckpt["parameters"], ckpt["states"]; rng)
end
