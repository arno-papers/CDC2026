include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots, Serialization

function rollout_trajectory_cpu(model, ps_cpu, st_cpu, rng,
        theta_dyn::Vector{Float32}, sigma_prop::Float32, sigma_add::Float32;
        n_substeps::Int=N_SUBSTEPS)
    u = zeros(Float32, 5, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    n_total = N_STEPS * n_substeps + 1
    cc_traj = zeros(Float32, n_total)
    cp_traj = zeros(Float32, n_total)
    designs = zeros(Float32, N_STEPS)
    observations = zeros(Float32, N_STEPS)
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
            cc_traj[idx] = u[4, 1] / V_C
            cp_traj[idx] = u[5, 1] / V_P
        end
        cc = u[4, 1] / V_C
        y_obs = cc + sigma_prop * cc * randn(rng, Float32) + sigma_add * randn(rng, Float32)
        observations[step] = y_obs
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end
    return cc_traj, cp_traj, designs, observations
end

results_dir = joinpath(@__DIR__, "results")
ckpt_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
@assert isfile(ckpt_file) "Checkpoint not found: $ckpt_file. Run training first."

ckpt = deserialize(ckpt_file)
ps_cpu = ckpt["parameters"]
st_cpu = ckpt["states"]

rng = Random.MersenneTwister(42)
n_samples = 20

t_states = range(0, N_STEPS * DT; length=N_STEPS * N_SUBSTEPS + 1)
t_obs = Float32.(1:N_STEPS) .* DT
t_designs = Float32.(1:N_STEPS) .* DT

# Fix absorption parameters at midrange
k_a_mid  = 0.5f0 * (K_A_LO + K_A_HI)
k_tr_mid = 0.5f0 * (K_TR_LO + K_TR_HI)

# Vary CL and Q_d jointly: low elimination vs high elimination
sp_mid  = 0.5f0 * (SIGMA_PROP_LO + SIGMA_PROP_HI)
sa_mid  = 0.5f0 * (SIGMA_ADD_LO + SIGMA_ADD_HI)

theta_low  = Float32[k_a_mid, k_tr_mid, CL_LO, Q_D_LO]   # slow elimination + low redistribution
theta_high = Float32[k_a_mid, k_tr_mid, CL_HI, Q_D_HI]   # fast elimination + high redistribution

p_cc = plot(title="Central concentration", xlabel="Time (hr)", ylabel="Cc (mg/L)",
            legend=:topleft, xticks=0:4:N_STEPS)
p_cp = plot(title="Peripheral concentration", xlabel="Time (hr)", ylabel="Cp (mg/L)",
            legend=false, xticks=0:4:N_STEPS)
p_d  = plot(title="Infusion rate (design)", xlabel="Time (hr)", ylabel="Rinf (mg/hr)",
            ylims=(-0.5, ACTION_HI + 0.5), legend=false, xticks=0:4:N_STEPS)

for i in 1:n_samples
    label_lo = i == 1 ? "Slow elim. (CL=$(CL_LO), Qd=$(Q_D_LO))" : ""
    label_hi = i == 1 ? "Fast elim. (CL=$(CL_HI), Qd=$(Q_D_HI))" : ""

    cc_l, cp_l, d_l, obs_l = rollout_trajectory_cpu(policy, ps_cpu, st_cpu, rng, theta_low, sp_mid, sa_mid)
    plot!(p_cc, t_states, cc_l; lw=0.8, alpha=0.3, color=:steelblue, label=label_lo)
    scatter!(p_cc, t_obs, obs_l; markersize=2, alpha=0.3, color=:steelblue, label="")
    plot!(p_cp, t_states, cp_l; lw=0.8, alpha=0.3, color=:steelblue, label=label_lo)
    plot!(p_d, t_designs, d_l; lw=0.8, alpha=0.3, color=:steelblue, label=label_lo,
          seriestype=:steppost)

    cc_h, cp_h, d_h, obs_h = rollout_trajectory_cpu(policy, ps_cpu, st_cpu, rng, theta_high, sp_mid, sa_mid)
    plot!(p_cc, t_states, cc_h; lw=0.8, alpha=0.3, color=:crimson, label=label_hi)
    scatter!(p_cc, t_obs, obs_h; markersize=2, alpha=0.3, color=:crimson, label="")
    plot!(p_cp, t_states, cp_h; lw=0.8, alpha=0.3, color=:crimson, label=label_hi)
    plot!(p_d, t_designs, d_h; lw=0.8, alpha=0.3, color=:crimson, label=label_hi,
          seriestype=:steppost)
end

plt = plot(p_cc, p_cp, p_d;
           layout=(1, 3), size=(1200, 350), bottom_margin=5Plots.mm, left_margin=5Plots.mm)
outfile = joinpath(results_dir, "plot_nuisance.png")
save_plot(plt, outfile)
