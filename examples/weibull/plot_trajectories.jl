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
    designs = zeros(Float32, N_STEPS)
    st_local = st_cpu
    theta_mat = reshape(theta_dyn, N_PARAMS_DYN, 1)
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        q_in = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = q_in
        u = integrate_cpu(u, theta_mat, q_in, DT, n_substeps)
        cc = u[4, 1] / V_C
        cc_traj[step + 1] = cc
        y_obs = cc + sigma_prop * cc * randn(rng, Float32) + sigma_add * randn(rng, Float32)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = q_in
    end
    return cc_traj, designs
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

    # Slow absorption
    k_a_slow, k_tr_slow = 0.8f0, 0.8f0
    # Fast absorption
    k_a_fast, k_tr_fast = 2.5f0, 2.5f0

    p1 = plot(title="Slow absorption (k_a=0.8, k_tr=0.8)",
              xlabel="Time (hr)", ylabel="C_c (mg/L)")
    p2 = plot(title="Fast absorption (k_a=2.5, k_tr=2.5)",
              xlabel="Time (hr)", ylabel="C_c (mg/L)")
    p3 = plot(title="Dosing — slow absorption",
              xlabel="Time (hr)", ylabel="Q_in (mg/hr)", ylims=(-0.5, ACTION_HI + 0.5))
    p4 = plot(title="Dosing — fast absorption",
              xlabel="Time (hr)", ylabel="Q_in (mg/hr)", ylims=(-0.5, ACTION_HI + 0.5))

    for i in 1:n_samples
        cl = CL_LO + (CL_HI - CL_LO) * rand(rng, Float32)
        qd = Q_D_LO + (Q_D_HI - Q_D_LO) * rand(rng, Float32)
        sp = SIGMA_PROP_LO + (SIGMA_PROP_HI - SIGMA_PROP_LO) * rand(rng, Float32)
        sa = SIGMA_ADD_LO + (SIGMA_ADD_HI - SIGMA_ADD_LO) * rand(rng, Float32)
        label = i == 1 ? "samples" : ""

        theta_slow = Float32[k_a_slow, k_tr_slow, cl, qd]
        cc_s, d_s = rollout_trajectory_cpu(policy, ps_cpu, st_cpu, rng, theta_slow, sp, sa)
        plot!(p1, t_states, cc_s; alpha=0.4, color=:steelblue, label=label)
        plot!(p3, t_designs, d_s; alpha=0.4, color=:steelblue, label=label,
              marker=:circle, markersize=2)

        theta_fast = Float32[k_a_fast, k_tr_fast, cl, qd]
        cc_f, d_f = rollout_trajectory_cpu(policy, ps_cpu, st_cpu, rng, theta_fast, sp, sa)
        plot!(p2, t_states, cc_f; alpha=0.4, color=:crimson, label=label)
        plot!(p4, t_designs, d_f; alpha=0.4, color=:crimson, label=label,
              marker=:circle, markersize=2)
    end

    plt = plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 700))
    outfile = joinpath(results_dir, "plot_trajectories.png")
    save_plot(plt, outfile)
end
