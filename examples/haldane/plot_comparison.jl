include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

function rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng,
        theta_dyn::Vector{Float32}, sigma::Float32;
        Cx0::Float32=0.25f0, n_substeps::Int=N_SUBSTEPS)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    cs_traj = zeros(Float32, N_STEPS + 1)
    designs = zeros(Float32, N_STEPS)
    cs_traj[1] = u[1, 1]
    st_local = st_cpu
    theta_mat = reshape(theta_dyn, N_PARAMS_DYN, 1)
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = d
        u = integrate_cpu(u, theta_mat, d, DT, n_substeps)
        y_obs = u[1, 1] + sigma * randn(rng, Float32)
        cs_traj[step + 1] = u[1, 1]
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end
    return cs_traj, designs
end

function plot_comparison(model, ps_cpu, st_cpu;
        rng=Random.MersenneTwister(42),
        n_samples=20,
        outfile=joinpath(@__DIR__, "results", "plot_comparison.png"))

    mu_max_mid = 0.5f0 * (mu_max_lo + mu_max_hi)
    K_s_mid = 0.5f0 * (K_s_lo + K_s_hi)

    alpha_no_inhib = 0.001f0
    alpha_inhib = 0.10f0

    theta_no = Float32[mu_max_mid, K_s_mid, alpha_no_inhib]
    theta_yes = Float32[mu_max_mid, K_s_mid, alpha_inhib]

    t_states = 0:N_STEPS
    t_designs = 1:N_STEPS

    p1 = plot(title = "No inhibition (\u03b1 \u2248 0)", xlabel = "Time (h)", ylabel = "C_s (g/L)")
    p2 = plot(title = "Substrate inhibition (\u03b1 = 0.10)", xlabel = "Time (h)", ylabel = "C_s (g/L)")
    p3 = plot(title = "No inhibition (\u03b1 \u2248 0)", xlabel = "Time (h)", ylabel = "d (L/h)", ylims = (-0.5, 10.5))
    p4 = plot(title = "Substrate inhibition (\u03b1 = 0.10)", xlabel = "Time (h)", ylabel = "d (L/h)", ylims = (-0.5, 10.5))

    for i in 1:n_samples
        sigma = σ_lo + (σ_hi - σ_lo) * rand(rng, Float32)
        Cx0 = Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng, Float32)

        cs_no, d_no = rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng, theta_no, sigma; Cx0=Cx0)
        label_no = i == 1 ? "samples" : ""
        plot!(p1, t_states, cs_no; alpha=0.4, color=:dodgerblue, label=label_no)
        plot!(p3, t_designs, d_no; alpha=0.4, color=:dodgerblue, label=label_no, marker=:circle, markersize=2)

        cs_yes, d_yes = rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng, theta_yes, sigma; Cx0=Cx0)
        label_yes = i == 1 ? "samples" : ""
        plot!(p2, t_states, cs_yes; alpha=0.4, color=:crimson, label=label_yes)
        plot!(p4, t_designs, d_yes; alpha=0.4, color=:crimson, label=label_yes, marker=:circle, markersize=2)
    end

    plt = plot(p1, p2, p3, p4; layout=(2, 2), size=(900, 700))
    save_plot(plt, outfile)
    return plt
end

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")
    ckpt_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
    @assert isfile(ckpt_file) "Checkpoint not found: $ckpt_file. Run training first."

    ckpt = deserialize(ckpt_file)
    rng = Random.MersenneTwister(42)
    plot_comparison(policy, ckpt["parameters"], ckpt["states"]; rng=rng)
end
