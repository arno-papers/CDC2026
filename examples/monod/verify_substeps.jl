include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots, Serialization, Random, Printf

# Load checkpoint
results_dir = joinpath(@__DIR__, "results")
ckpt = deserialize(joinpath(results_dir, "checkpoint.jls"))
ps_cpu = ckpt["parameters"]
st_cpu = ckpt["states"]

rng = MersenneTwister(42)

const N_SUBSTEPS_REF = 500
const N_SUBSTEPS_TEST = N_SUBSTEPS  # 50, from model.jl

function rollout_static(Q_in_val::Float32, theta, u0; n_substeps)
    u = reshape(u0, 3, 1)
    traj = zeros(Float32, N_STEPS + 1, 3)
    traj[1, :] .= vec(u)
    theta_mat = reshape(theta, 2, 1)
    for step in 1:N_STEPS
        u = integrate_cpu(u, theta_mat, Q_in_val, DT, n_substeps)
        traj[step + 1, :] .= vec(u)
    end
    return traj
end

function rollout_adaptive(model, ps, st, theta, u0; n_substeps)
    u = reshape(u0, 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    traj = zeros(Float32, N_STEPS + 1, 3)
    designs = zeros(Float32, N_STEPS)
    traj[1, :] .= vec(u)
    st_local = st
    theta_mat = reshape(theta, 2, 1)
    for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps, st_local)
        Q_in = Float32(action[1])
        designs[step] = Q_in
        u = integrate_cpu(u, theta_mat, Q_in, DT, n_substeps)
        traj[step + 1, :] .= vec(u)
        input_buffer[1, step, 1] = u[1, 1]
        input_buffer[2, step, 1] = Q_in
    end
    return traj, designs
end

# Sample many parameter sets from the prior
N_SAMPLES = 200
thetas = [Float32[mu_max_lo + (mu_max_hi - mu_max_lo) * rand(rng),
                   K_s_lo + (K_s_hi - K_s_lo) * rand(rng)] for _ in 1:N_SAMPLES]
Cx0s = [Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng, Float32) for _ in 1:N_SAMPLES]

state_names = ["C_s", "C_x", "V"]

# =========================================================================
#  1. Accuracy comparison: N_SUBSTEPS=50 vs 500 (adaptive policy)
# =========================================================================
println("=== Accuracy: N_SUBSTEPS=$N_SUBSTEPS_TEST vs $N_SUBSTEPS_REF (adaptive) ===")

max_abs_err = zeros(Float32, 3)
max_rel_err = zeros(Float32, 3)
all_abs_errs = [Float32[] for _ in 1:3]

for i in 1:N_SAMPLES
    u0 = Float32[3.0, Cx0s[i], 7.0]
    traj_ref, _ = rollout_adaptive(policy, ps_cpu, st_cpu, thetas[i], u0; n_substeps=N_SUBSTEPS_REF)
    traj_test, _ = rollout_adaptive(policy, ps_cpu, st_cpu, thetas[i], u0; n_substeps=N_SUBSTEPS_TEST)
    for s in 1:3
        abs_err = maximum(abs.(traj_test[:, s] .- traj_ref[:, s]))
        ref_scale = maximum(abs.(traj_ref[:, s])) + 1f-10
        rel_err = abs_err / ref_scale
        max_abs_err[s] = max(max_abs_err[s], abs_err)
        max_rel_err[s] = max(max_rel_err[s], rel_err)
        push!(all_abs_errs[s], abs_err)
    end
end

for s in 1:3
    med = sort(all_abs_errs[s])[N_SAMPLES ÷ 2]
    p95 = sort(all_abs_errs[s])[Int(ceil(0.95 * N_SAMPLES))]
    @printf("  %s: max_abs=%.2e  max_rel=%.2e  median_abs=%.2e  p95_abs=%.2e\n",
            state_names[s], max_abs_err[s], max_rel_err[s], med, p95)
end
@printf("  (measurement noise σ ∈ [%.2f, %.2f])\n", sigma_lo, sigma_hi)

# =========================================================================
#  2. Static profiles: always-on and always-off
# =========================================================================
println("\n=== Accuracy: static profiles ===")
for (label, Q_val) in [("Always ON", ACTION_HI), ("Always OFF", ACTION_LO)]
    errs = [Float32[] for _ in 1:3]
    for i in 1:N_SAMPLES
        u0 = Float32[3.0, Cx0s[i], 7.0]
        traj_ref = rollout_static(Q_val, thetas[i], u0; n_substeps=N_SUBSTEPS_REF)
        traj_test = rollout_static(Q_val, thetas[i], u0; n_substeps=N_SUBSTEPS_TEST)
        for s in 1:3
            push!(errs[s], maximum(abs.(traj_test[:, s] .- traj_ref[:, s])))
        end
    end
    println("  $label (Q_in=$Q_val):")
    for s in 1:3
        sorted = sort(errs[s])
        @printf("    %s: max=%.2e  median=%.2e  p95=%.2e\n",
                state_names[s], sorted[end], sorted[N_SAMPLES ÷ 2], sorted[Int(ceil(0.95 * N_SAMPLES))])
    end
end

# =========================================================================
#  3. Trajectory fan plots (adaptive policy, 50 samples)
# =========================================================================
println("\n=== Generating trajectory fan plots ===")

N_PLOT = 50
t = 0:N_STEPS

p1 = plot(title="Substrate C_s", xlabel="Step", ylabel="g/L")
p2 = plot(title="Biomass C_x", xlabel="Step", ylabel="g/L")
p3 = plot(title="Volume V", xlabel="Step", ylabel="L")
p4 = plot(title="Design Q_in", xlabel="Step", ylabel="L/hr")

for i in 1:N_PLOT
    u0 = Float32[3.0, Cx0s[i], 7.0]
    traj, designs = rollout_adaptive(policy, ps_cpu, st_cpu, thetas[i], u0; n_substeps=N_SUBSTEPS_TEST)
    lab = i == 1 ? "Adaptive" : ""
    plot!(p1, t, traj[:, 1]; alpha=0.3, color=:blue, label=lab)
    plot!(p2, t, traj[:, 2]; alpha=0.3, color=:red, label=lab)
    plot!(p3, t, traj[:, 3]; alpha=0.3, color=:green, label=lab)
    plot!(p4, 1:N_STEPS, designs; alpha=0.3, color=:blue, label=lab)
end

# Overlay always-on and always-off for one mid-range sample
u0_mid = Float32[3.0, 0.3, 7.0]
theta_mid = Float32[0.4, 0.45]
traj_on = rollout_static(ACTION_HI, theta_mid, u0_mid; n_substeps=N_SUBSTEPS_TEST)
traj_off = rollout_static(ACTION_LO, theta_mid, u0_mid; n_substeps=N_SUBSTEPS_TEST)

plot!(p1, t, traj_on[:, 1]; lw=2, color=:red, ls=:dash, label="Always ON")
plot!(p1, t, traj_off[:, 1]; lw=2, color=:gray, ls=:dot, label="Always OFF")
plot!(p2, t, traj_on[:, 2]; lw=2, color=:red, ls=:dash, label="")
plot!(p2, t, traj_off[:, 2]; lw=2, color=:gray, ls=:dot, label="")
plot!(p3, t, traj_on[:, 3]; lw=2, color=:red, ls=:dash, label="")
plot!(p3, t, traj_off[:, 3]; lw=2, color=:gray, ls=:dot, label="")
hline!(p4, [ACTION_HI]; lw=2, color=:red, ls=:dash, label="Always ON")
hline!(p4, [ACTION_LO]; lw=2, color=:gray, ls=:dot, label="Always OFF")

plt = plot(p1, p2, p3, p4; layout=(2,2), size=(1000, 700),
           plot_title="Monod: $N_PLOT adaptive samples + static baselines (N_SUBSTEPS=$N_SUBSTEPS_TEST)")
outfile = joinpath(results_dir, "verify_substeps.png")
savefig(plt, outfile)
println("Saved: $outfile")
