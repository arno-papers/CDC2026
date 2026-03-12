#!/usr/bin/env julia
# Analyze whether outliers are a training failure.

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Printf
using Random
using Serialization
using Statistics

results_dir = joinpath(@__DIR__, "results")
ps_cpu, st_cpu, _ = load_checkpoint_cpu(results_dir)

true_μ   = 0.47f0
true_K   = 0.345f0
true_σ   = 0.1f0
true_Cx0 = 0.3f0
θT = Float32[true_μ, true_K]

# ---- 1. Check output head bias: does the policy default to low actions? ----
println("=== Output head analysis ===")
# The output is ACTION_HI * sigmoid(output_head(x[:, end, :]))
# output_head is Dense(32 => 1) with initial bias = -4.0
# sigmoid(-4) = 0.018, so default action ≈ 0.18 (close to zero)
# Check what the learned bias is:
output_bias = ps_cpu.output_head.bias
output_weight_norm = sum(abs2, ps_cpu.output_head.weight)
@printf("  output_head.bias = %.4f  (sigmoid = %.4f, action = %.4f)\n",
        output_bias[1], 1/(1+exp(-output_bias[1])), 10/(1+exp(-output_bias[1])))
@printf("  output_head.weight norm = %.4f\n", sqrt(output_weight_norm))
println()

# ---- 2. What does the policy output given zero observations? ----
println("=== Policy behavior with zero input buffer ===")
let input_buffer = zeros(Float32, 2, N_STEPS, 1), st_local = st_cpu
    for step in 1:N_STEPS
        action, st_local = policy(input_buffer, ps_cpu, st_local)
        d = Float32(action[1])
        @printf("  Step %2d: action = %.4f\n", step, d)
        input_buffer[2, step, 1] = d  # feed design back, obs stays 0
    end
end
println()

# ---- 3. Distribution of actions across many rollouts ----
println("=== Action distribution across 2000 rollouts ===")
n_rollouts = 2000
seed = 0
all_designs = Matrix{Float32}(undef, N_STEPS, n_rollouts)
for i in 1:n_rollouts
    rng = MersenneTwister(seed + i)
    d = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng, θT, true_σ;
                                     Cx0=true_Cx0, n_substeps=N_SUBSTEPS)
    all_designs[:, i] .= d
end

println("  Per-step action quantiles:")
@printf("  %5s  %8s  %8s  %8s  %8s  %8s  %8s\n",
        "Step", "min", "p5", "p25", "median", "p75", "p95")
for step in 1:N_STEPS
    vals = all_designs[step, :]
    @printf("  %5d  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n",
            step, minimum(vals), quantile(vals, 0.05), quantile(vals, 0.25),
            median(vals), quantile(vals, 0.75), quantile(vals, 0.95))
end
println()

# ---- 4. Total feed volume distribution ----
total_feed = vec(sum(all_designs; dims=1))
@printf("  Total feed: min=%.3f  p5=%.3f  median=%.3f  p95=%.3f  max=%.3f\n",
        minimum(total_feed), quantile(total_feed, 0.05), median(total_feed),
        quantile(total_feed, 0.95), maximum(total_feed))

low_feed = total_feed .< 3.0
@printf("  Trials with total_feed < 3.0: %d / %d (%.1f%%)\n",
        sum(low_feed), n_rollouts, 100*sum(low_feed)/n_rollouts)

zero_ish = total_feed .< 0.5
@printf("  Trials with total_feed < 0.5: %d / %d (%.1f%%)\n",
        sum(zero_ish), n_rollouts, 100*sum(zero_ish)/n_rollouts)
println()

# ---- 5. What observations trigger the low-feed behavior? ----
println("=== Observation patterns in low-feed vs normal trials ===")
low_idx = findall(total_feed .< 3.0)
high_idx = findall(total_feed .>= quantile(total_feed, 0.5))

# Re-rollout a few low-feed trials and capture observations
println("\n  Low-feed trial examples (observations + actions):")
for (rank, idx) in enumerate(low_idx[1:min(5, length(low_idx))])
    rng = MersenneTwister(seed + idx)
    u = reshape(Float32[3.0f0, true_Cx0, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    st_local = st_cpu
    theta_mat = reshape(θT, 2, 1)

    @printf("  Trial %d (total_feed=%.3f):\n", idx, total_feed[idx])
    local st_loc = st_cpu
    for step in 1:N_STEPS
        action, st_loc = policy(input_buffer, ps_cpu, st_loc)
        d = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        u = integrate_cpu(u, theta_mat, d, DT, N_SUBSTEPS)
        y_obs = u[1, 1] + true_σ * randn(rng, Float32)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
        @printf("    step %2d: Cs=%.4f  obs=%.4f  action=%.4f\n", step, u[1,1], y_obs, d)
    end
    println()
end

# ---- 6. Same θ but different noise — how variable is the policy? ----
println("=== Policy variability: same θ, different noise seeds ===")
n_seeds = 100
designs_same_theta = Matrix{Float32}(undef, N_STEPS, n_seeds)
for i in 1:n_seeds
    rng = MersenneTwister(1000 + i)
    d = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng, θT, true_σ;
                                     Cx0=true_Cx0, n_substeps=N_SUBSTEPS)
    designs_same_theta[:, i] .= d
end
total_same = vec(sum(designs_same_theta; dims=1))
@printf("  Total feed (100 seeds, same θ): mean=%.3f  std=%.3f  min=%.3f  max=%.3f\n",
        mean(total_same), std(total_same), minimum(total_same), maximum(total_same))
@printf("  CV = %.2f\n", std(total_same) / mean(total_same))

# Compare to BIM (fixed, zero variance)
bim_data = deserialize(joinpath(results_dir, "bim_std_design.jls"))
bim_total = sum(bim_data["static_design"])
@printf("  BIM total feed = %.3f (fixed, CV=0)\n", bim_total)

println("\nDone.")
