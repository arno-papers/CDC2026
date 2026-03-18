using Plots, Random, Serialization
include(joinpath(@__DIR__, "plotting.jl"))

# ============================================================================
#  CPU trajectory helpers
# ============================================================================

function rollout_policy_cpu(model, ps_cpu, st_cpu, rng; theta, sigma, u0)
    u = reshape(u0, 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    n_total = N_STEPS * N_SUBSTEPS + 1
    traj = zeros(Float32, n_total, 3)
    designs = zeros(Float32, N_STEPS)
    observations = zeros(Float32, N_STEPS)
    traj[1, :] .= vec(u)

    st_local = st_cpu
    theta_mat = reshape(theta, N_PARAMS_DYN, 1)
    dt_sub = DT / N_SUBSTEPS
    for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = Float32(action[1])
        designs[step] = d

        for s in 1:N_SUBSTEPS
            u = rk4_step(u, theta_mat, d, dt_sub)
            idx = (step - 1) * N_SUBSTEPS + s + 1
            traj[idx, :] .= vec(u)
        end

        y_obs = u[1, 1] + sigma * randn(rng, Float32)
        observations[step] = y_obs
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end

    return traj, designs, observations
end

function simulate_static_cpu(design, theta, sigma, rng, Cx0)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    n_total = N_STEPS * N_SUBSTEPS + 1
    traj = zeros(Float32, n_total, 3)
    observations = zeros(Float32, N_STEPS)
    traj[1, :] .= vec(u)
    theta_mat = reshape(theta, N_PARAMS_DYN, 1)
    dt_sub = DT / N_SUBSTEPS
    for step in 1:N_STEPS
        for s in 1:N_SUBSTEPS
            u = rk4_step(u, theta_mat, Float32(design[step]), dt_sub)
            idx = (step - 1) * N_SUBSTEPS + s + 1
            traj[idx, :] .= vec(u)
        end
        observations[step] = u[1, 1] + sigma * randn(rng, Float32)
    end
    return traj, observations
end

# ============================================================================
#  Overlaid 3-design comparison plot (Cs, Cx, Qin panels)
# ============================================================================

function plot_design_comparison(model, ps_cpu, st_cpu,
        static_designs::Vector{<:Pair};
        rng, n_samples=20,
        outfile=joinpath(@__DIR__, "results", "plot_design_comparison.png"))

    t_states = range(0, N_STEPS * DT; length=N_STEPS * N_SUBSTEPS + 1)
    t_obs = Float32.(1:N_STEPS) .* DT
    t_designs = (1:N_STEPS) .* DT

    # Pre-sample parameters (shared across all designs for fair comparison)
    samples = [(
        Float32[mu_max_lo + (mu_max_hi - mu_max_lo) * rand(rng),
                K_s_lo + (K_s_hi - K_s_lo) * rand(rng)],
        Float32(Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng)),
        Float32(σ_lo + (σ_hi - σ_lo) * rand(rng))
    ) for _ in 1:n_samples]

    p_cs  = plot(xlabel="Time (h)", ylabel="Cs (g/L)", title="Substrate concentration",
                 legend=:topleft, xticks=0:2:N_STEPS)
    p_cx  = plot(xlabel="Time (h)", ylabel="Cx (g/L)", title="Biomass concentration",
                 legend=false, xticks=0:2:N_STEPS)
    p_qin = plot(xlabel="Time (h)", ylabel="Qin (L/h)", title="Feed rate (design)",
                 legend=false, xticks=0:2:N_STEPS)

    # --- Adaptive rollouts ---
    astyle = DESIGN_STYLES["adaptive"]
    for (i, (theta, Cx0, sigma)) in enumerate(samples)
        u0 = Float32[3.0f0, Cx0, 7.0f0]
        traj, designs, obs = rollout_policy_cpu(model, ps_cpu, st_cpu, rng;
                                                theta=theta, sigma=sigma, u0=u0)
        lab = i == 1 ? astyle.label : ""
        plot!(p_cs, t_states, traj[:, 1]; alpha=0.3, color=astyle.color, lw=0.8, label=lab)
        scatter!(p_cs, t_obs, obs; markersize=2, alpha=0.3, color=astyle.color, label="")
        plot!(p_cx, t_states, traj[:, 2]; alpha=0.3, color=astyle.color, lw=0.8, label=lab)
        plot!(p_qin, t_designs, designs; alpha=0.3, color=astyle.color, lw=0.8, label=lab,
              seriestype=:steppost)
    end

    # --- Static designs: bold design line + thin trajectory rollouts ---
    for (name, design) in static_designs
        style = get(DESIGN_STYLES, name, (label=name, color=:black))
        plot!(p_qin, t_designs, Float64.(design); color=style.color, lw=2.5, label=style.label,
              seriestype=:steppost)
        for (i, (theta, Cx0, sigma)) in enumerate(samples)
            traj, obs = simulate_static_cpu(design, theta, sigma, rng, Cx0)
            lab = i == 1 ? style.label : ""
            plot!(p_cs, t_states, traj[:, 1]; alpha=0.3, color=style.color, lw=0.8, label=lab)
            scatter!(p_cs, t_obs, obs; markersize=2, alpha=0.3, color=style.color, label="")
            plot!(p_cx, t_states, traj[:, 2]; alpha=0.3, color=style.color, lw=0.8, label=lab)
        end
    end

    plt = plot(p_cs, p_cx, p_qin; layout=(1, 3), size=(1200, 350),
               bottom_margin=5Plots.mm, left_margin=5Plots.mm)
    save_plot(plt, outfile)
    return plt
end

