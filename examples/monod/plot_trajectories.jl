using Plots, Random, Serialization
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

# ============================================================================
#  CPU trajectory helpers
# ============================================================================

function rollout_policy_cpu(model, ps_cpu, st_cpu; theta, u0)
    u = reshape(u0, 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    traj = zeros(Float32, N_STEPS + 1, 3)
    designs = zeros(Float32, N_STEPS)
    traj[1, :] .= vec(u)

    st_local = st_cpu
    for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = Float32(action[1])
        designs[step] = d

        u = integrate_cpu(u, reshape(theta, 2, 1), d, DT, N_SUBSTEPS)
        traj[step + 1, :] .= vec(u)

        y_obs = u[1, 1]
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end

    return traj, designs
end

function simulate_static_cpu(design, theta, Cx0)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    traj = zeros(Float32, N_STEPS + 1, 3)
    traj[1, :] .= vec(u)
    theta_mat = reshape(theta, 2, 1)
    for step in 1:N_STEPS
        u = integrate_cpu(u, theta_mat, Float32(design[step]), DT, N_SUBSTEPS)
        traj[step + 1, :] .= vec(u)
    end
    return traj
end

# ============================================================================
#  Overlaid 3-design comparison plot (Cs, Cx, Qin panels)
# ============================================================================

function plot_design_comparison(model, ps_cpu, st_cpu,
        static_designs::Vector{<:Pair};
        rng, n_samples=20,
        outfile=joinpath(@__DIR__, "results", "plot_design_comparison.png"))

    t_states = 0:N_STEPS
    t_designs = 1:N_STEPS

    # Pre-sample parameters (shared across all designs for fair comparison)
    samples = [(
        Float32[mu_max_lo + (mu_max_hi - mu_max_lo) * rand(rng),
                K_s_lo + (K_s_hi - K_s_lo) * rand(rng)],
        Float32(Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng))
    ) for _ in 1:n_samples]

    p_cs  = plot(xlabel="Time (h)", ylabel="Cs (g/L)", title="Substrate concentration")
    p_cx  = plot(xlabel="Time (h)", ylabel="Cx (g/L)", title="Biomass concentration")
    p_qin = plot(xlabel="Time (h)", ylabel="Qin (L/h)", title="Feed rate (design)")

    # --- Adaptive rollouts ---
    astyle = DESIGN_STYLES["adaptive"]
    for (i, (theta, Cx0)) in enumerate(samples)
        u0 = Float32[3.0f0, Cx0, 7.0f0]
        traj, designs = rollout_policy_cpu(model, ps_cpu, st_cpu; theta=theta, u0=u0)
        lab = i == 1 ? astyle.label : ""
        plot!(p_cs, t_states, traj[:, 1]; alpha=0.3, color=astyle.color, lw=0.8, label=lab)
        plot!(p_cx, t_states, traj[:, 2]; alpha=0.3, color=astyle.color, lw=0.8, label=lab)
        plot!(p_qin, t_designs, designs; alpha=0.3, color=astyle.color, lw=0.8, label=lab)
    end

    # --- Static designs: bold design line + thin trajectory rollouts ---
    for (name, design) in static_designs
        style = get(DESIGN_STYLES, name, (label=name, color=:black))
        plot!(p_qin, t_designs, Float64.(design); color=style.color, lw=2.5, label=style.label)
        for (i, (theta, Cx0)) in enumerate(samples)
            traj = simulate_static_cpu(design, theta, Cx0)
            lab = i == 1 ? style.label : ""
            plot!(p_cs, t_states, traj[:, 1]; alpha=0.3, color=style.color, lw=0.8, label=lab)
            plot!(p_cx, t_states, traj[:, 2]; alpha=0.3, color=style.color, lw=0.8, label=lab)
        end
    end

    plt = plot(p_cs, p_cx, p_qin; layout=(1, 3), size=(1200, 350),
               legend=:topleft, bottom_margin=5Plots.mm)
    save_plot(plt, outfile)
    return plt
end

