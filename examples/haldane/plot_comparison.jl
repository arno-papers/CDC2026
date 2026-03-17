include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots

function rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng,
        theta_dyn::Vector{Float32}, sigma::Float32;
        Cx0::Float32=0.25f0, n_substeps::Int=N_SUBSTEPS)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    n_total = N_STEPS * n_substeps + 1
    cs_traj = zeros(Float32, n_total)
    cx_traj = zeros(Float32, n_total)
    designs = zeros(Float32, N_STEPS)
    cs_traj[1] = u[1, 1]
    cx_traj[1] = u[2, 1]
    st_local = st_cpu
    theta_mat = reshape(theta_dyn, N_PARAMS_DYN, 1)
    dt_sub = DT / n_substeps
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = d
        for s in 1:n_substeps
            u = rk4_step(u, theta_mat, d, dt_sub)
            idx = (step - 1) * n_substeps + s + 1
            cs_traj[idx] = u[1, 1]
            cx_traj[idx] = u[2, 1]
        end
        y_obs = u[1, 1] + sigma * randn(rng, Float32)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end
    return cs_traj, cx_traj, designs
end

function plot_comparison(model, ps_cpu, st_cpu;
        rng=Random.MersenneTwister(42),
        n_samples=20,
        outfile=joinpath(@__DIR__, "results", "plot_comparison.png"))

    mu_max_mid = 0.5f0 * (mu_max_lo + mu_max_hi)
    K_s_mid = 0.5f0 * (K_s_lo + K_s_hi)

    alpha_no_inhib = 0.0f0    # no inhibition (α = 0)
    alpha_inhib = 0.15f0      # strong inhibition (α = 0.15)

    theta_no = Float32[mu_max_mid, K_s_mid, alpha_no_inhib]
    theta_yes = Float32[mu_max_mid, K_s_mid, alpha_inhib]

    t_states = range(0, N_STEPS * DT; length=N_STEPS * N_SUBSTEPS + 1)
    t_designs = (1:N_STEPS) .* DT

    p_cs = plot(title="Substrate concentration", xlabel="Time (h)", ylabel="Cs (g/L)",
                legend=:topleft, xticks=0:2:N_STEPS)
    p_cx = plot(title="Biomass concentration", xlabel="Time (h)", ylabel="Cx (g/L)",
                legend=false, xticks=0:2:N_STEPS)
    p_qin = plot(title="Feed rate (design)", xlabel="Time (h)", ylabel="Qin (L/h)",
                 ylims=(-0.5, ACTION_HI + 0.5), legend=false, xticks=0:2:N_STEPS)

    for i in 1:n_samples
        sigma = σ_lo + (σ_hi - σ_lo) * rand(rng, Float32)
        Cx0 = Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng, Float32)

        label_no = i == 1 ? "No inhibition (α = 0)" : ""
        label_yes = i == 1 ? "Inhibition (α = 0.15)" : ""

        cs_no, cx_no, d_no = rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng, theta_no, sigma; Cx0=Cx0)
        plot!(p_cs, t_states, cs_no; lw=0.8, alpha=0.3, color=:steelblue, label=label_no)
        plot!(p_cx, t_states, cx_no; lw=0.8, alpha=0.3, color=:steelblue, label=label_no)
        plot!(p_qin, t_designs, d_no; lw=0.8, alpha=0.3, color=:steelblue, label=label_no,
              marker=:circle, markersize=2)

        cs_yes, cx_yes, d_yes = rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng, theta_yes, sigma; Cx0=Cx0)
        plot!(p_cs, t_states, cs_yes; lw=0.8, alpha=0.3, color=:crimson, label=label_yes)
        plot!(p_cx, t_states, cx_yes; lw=0.8, alpha=0.3, color=:crimson, label=label_yes)
        plot!(p_qin, t_designs, d_yes; lw=0.8, alpha=0.3, color=:crimson, label=label_yes,
              marker=:circle, markersize=2)
    end

    plt = plot(p_cs, p_cx, p_qin; layout=(1, 3), size=(1200, 350),
               bottom_margin=5Plots.mm, left_margin=5Plots.mm)
    save_plot(plt, outfile)
    return plt
end

results_dir = joinpath(@__DIR__, "results")
ckpt_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
@assert isfile(ckpt_file) "Checkpoint not found: $ckpt_file. Run training first."

ckpt = deserialize(ckpt_file)
rng = Random.MersenneTwister(42)
plot_comparison(policy, ckpt["parameters"], ckpt["states"]; rng=rng)
