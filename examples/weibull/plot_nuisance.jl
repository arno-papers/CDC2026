include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots, Serialization
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

function rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng,
        theta_dyn::Vector{Float32}, sigma_prop::Float32, sigma_add::Float32;
        n_substeps::Int=N_SUBSTEPS)
    u = zeros(Float32, 5, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    cc_traj = zeros(Float32, N_STEPS + 1)
    cp_traj = zeros(Float32, N_STEPS + 1)
    designs = zeros(Float32, N_STEPS)
    st_local = st_cpu
    theta_mat = reshape(theta_dyn, N_PARAMS_DYN, 1)
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = d
        u = integrate_cpu(u, theta_mat, d, DT, N_SUBSTEPS)
        cc_traj[step + 1] = u[4, 1] / V_C
        cp_traj[step + 1] = u[5, 1] / V_P
        cc = cc_traj[step + 1]
        y_obs = cc + sigma_prop * cc * randn(rng, Float32) + sigma_add * randn(rng, Float32)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end
    return cc_traj, cp_traj, designs
end

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")
    ckpt_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
    @assert isfile(ckpt_file) "Checkpoint not found: $ckpt_file. Run training first."

    ckpt = deserialize(ckpt_file)
    ps_cpu = ckpt["parameters"]
    st_cpu = ckpt["states"]

    rng = Random.MersenneTwister(42)
    n_samples = 20

    t_states = Float32.(0:N_STEPS) .* DT
    t_designs = Float32.(1:N_STEPS) .* DT

    # Fix absorption parameters at midrange
    k_a_mid  = 0.5f0 * (K_A_LO + K_A_HI)
    k_tr_mid = 0.5f0 * (K_TR_LO + K_TR_HI)

    # Vary CL and Q_d jointly: low elimination vs high elimination
    sp_mid  = 0.5f0 * (SIGMA_PROP_LO + SIGMA_PROP_HI)
    sa_mid  = 0.5f0 * (SIGMA_ADD_LO + SIGMA_ADD_HI)

    theta_low  = Float32[k_a_mid, k_tr_mid, CL_LO, Q_D_LO]   # slow elimination + low redistribution
    theta_high = Float32[k_a_mid, k_tr_mid, CL_HI, Q_D_HI]   # fast elimination + high redistribution

    # Row 1: Central concentration C_c
    p_cc_lo = plot(title="Slow elimination (CL=$(CL_LO), Q_d=$(Q_D_LO))",
                   ylabel="C_c (mg/L)")
    p_cc_hi = plot(title="Fast elimination (CL=$(CL_HI), Q_d=$(Q_D_HI))")

    # Row 2: Peripheral concentration C_p
    p_cp_lo = plot(ylabel="C_p (mg/L)")
    p_cp_hi = plot()

    # Row 3: Dosing (drug input design)
    p_d_lo = plot(xlabel="Time (hr)", ylabel="Dose rate (mg/hr)",
                  ylims=(-0.5, ACTION_HI + 0.5))
    p_d_hi = plot(xlabel="Time (hr)",
                  ylims=(-0.5, ACTION_HI + 0.5))

    for i in 1:n_samples
        label = i == 1 ? "samples" : ""

        cc_l, cp_l, d_l = rollout_trajectory_cpu(policy, ps_cpu, st_cpu, rng, theta_low, sp_mid, sa_mid)
        plot!(p_cc_lo, t_states, cc_l; alpha=0.4, color=:steelblue, label=label)
        plot!(p_cp_lo, t_states, cp_l; alpha=0.4, color=:steelblue, label=label)
        plot!(p_d_lo, t_designs, d_l; alpha=0.4, color=:steelblue, label=label,
              marker=:circle, markersize=2)

        cc_h, cp_h, d_h = rollout_trajectory_cpu(policy, ps_cpu, st_cpu, rng, theta_high, sp_mid, sa_mid)
        plot!(p_cc_hi, t_states, cc_h; alpha=0.4, color=:crimson, label=label)
        plot!(p_cp_hi, t_states, cp_h; alpha=0.4, color=:crimson, label=label)
        plot!(p_d_hi, t_designs, d_h; alpha=0.4, color=:crimson, label=label,
              marker=:circle, markersize=2)
    end

    plt = plot(p_cc_lo, p_cc_hi, p_cp_lo, p_cp_hi, p_d_lo, p_d_hi;
               layout=(3, 2), size=(1000, 900))
    outfile = joinpath(results_dir, "plot_nuisance.png")
    save_plot(plt, outfile)
end
