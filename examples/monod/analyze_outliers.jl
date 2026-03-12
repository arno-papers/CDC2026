#!/usr/bin/env julia
# Analyze outlier trials in the adaptive posterior evaluation.

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Printf
using Random
using Serialization
using Statistics

results_dir = joinpath(@__DIR__, "results")
res = deserialize(joinpath(results_dir, "posterior_results.jls"))

true_μ   = res["true_mu_max"]
true_K   = res["true_K_s"]
true_σ   = res["true_sigma"]
true_Cx0 = res["true_Cx0"]
n_trials = res["n_trials"]
N_post   = res["N_post"]

pm_adapt = res["posterior_means"]["adaptive"]
pm_bim   = res["posterior_means"]["static_std"]
pv_adapt = res["posterior_vars"]["adaptive"]
pv_bim   = res["posterior_vars"]["static_std"]
ess_adapt = res["ess"]["adaptive"]
ess_bim   = res["ess"]["static_std"]

# Compute per-trial RMSE(K) and posterior std(K)
err_K_adapt = abs.(pm_adapt[2, :] .- true_K)
err_K_bim   = abs.(pm_bim[2, :] .- true_K)
std_K_adapt = sqrt.(pv_adapt[2, :])
std_K_bim   = sqrt.(pv_bim[2, :])
std_μ_adapt = sqrt.(pv_adapt[1, :])

# ---- 1. How many outlier trials? ----
q95_std_K = quantile(std_K_adapt, 0.95)
q99_std_K = quantile(std_K_adapt, 0.99)
println("=== Adaptive posterior std(K) distribution ===")
@printf("  median = %.5f\n", median(std_K_adapt))
@printf("  p75    = %.5f\n", quantile(std_K_adapt, 0.75))
@printf("  p90    = %.5f\n", quantile(std_K_adapt, 0.90))
@printf("  p95    = %.5f\n", q95_std_K)
@printf("  p99    = %.5f\n", q99_std_K)
@printf("  max    = %.5f\n", maximum(std_K_adapt))
println()

@printf("BIM std(K): median=%.5f  p95=%.5f  max=%.5f\n",
        median(std_K_bim), quantile(std_K_bim, 0.95), maximum(std_K_bim))
println()

# ---- 2. Identify outlier trials ----
outlier_mask = std_K_adapt .> q95_std_K
outlier_idx = findall(outlier_mask)
normal_idx = findall(.!outlier_mask)

println("=== Outlier trials (top 5% by posterior std(K)) ===")
@printf("  n_outlier = %d / %d\n", sum(outlier_mask), n_trials)
@printf("  Outlier mean |err(K)| = %.5f   Normal mean |err(K)| = %.5f\n",
        mean(err_K_adapt[outlier_idx]), mean(err_K_adapt[normal_idx]))
@printf("  Outlier mean ESS     = %.1f     Normal mean ESS     = %.1f\n",
        mean(ess_adapt[outlier_idx]), mean(ess_adapt[normal_idx]))
@printf("  Outlier median ESS   = %.1f     Normal median ESS   = %.1f\n",
        median(ess_adapt[outlier_idx]), median(ess_adapt[normal_idx]))
println()

# ---- 3. Roll out adaptive designs for outlier vs normal trials ----
ps_cpu, st_cpu, _ = load_checkpoint_cpu(results_dir)
seed = 0
θT = Float32[true_μ, true_K]

# Collect designs for all trials
all_designs = Matrix{Float32}(undef, N_STEPS, n_trials)
for trial_idx in 1:n_trials
    rng_rollout = MersenneTwister(seed + trial_idx)
    d = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng_rollout, θT, true_σ;
                                     Cx0=true_Cx0, n_substeps=N_SUBSTEPS)
    all_designs[:, trial_idx] .= d
end

# ---- 4. Compare designs: outlier vs normal ----
println("=== Design statistics ===")
println("  Step-by-step mean design (outlier vs normal):")
@printf("  %5s  %12s  %12s  %12s\n", "Step", "Outlier", "Normal", "Diff")
for step in 1:N_STEPS
    o_mean = mean(all_designs[step, outlier_idx])
    n_mean = mean(all_designs[step, normal_idx])
    @printf("  %5d  %12.4f  %12.4f  %+12.4f\n", step, o_mean, n_mean, o_mean - n_mean)
end
println()

# Total feed volume
vol_outlier = sum(all_designs[:, outlier_idx]; dims=1)
vol_normal  = sum(all_designs[:, normal_idx]; dims=1)
@printf("  Total feed (outlier): mean=%.3f  std=%.3f\n", mean(vol_outlier), std(vec(vol_outlier)))
@printf("  Total feed (normal):  mean=%.3f  std=%.3f\n", mean(vol_normal), std(vec(vol_normal)))
println()

# ---- 5. Check if outliers have near-zero designs ----
max_feed = maximum(all_designs; dims=1) |> vec
zero_design = max_feed .< 0.1f0
@printf("  Trials with max(design) < 0.1: %d / %d\n", sum(zero_design), n_trials)
@printf("  Overlap with outliers: %d / %d\n",
        sum(zero_design .& outlier_mask), sum(outlier_mask))
println()

# ---- 6. Look at substrate trajectories for outlier trials ----
println("=== Substrate trajectories for 5 worst outlier trials ===")
worst_idx = sortperm(std_K_adapt; rev=true)[1:5]
for (rank, idx) in enumerate(worst_idx)
    design = all_designs[:, idx]
    params = Float64[Float64(true_μ), Float64(true_K), Float64(true_Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=N_SUBSTEPS)

    @printf("  Trial %d (rank=%d): std(K)=%.4f  ESS=%.1f  err(K)=%.4f\n",
            idx, rank, std_K_adapt[idx], ess_adapt[idx], err_K_adapt[idx])
    @printf("    Design:    [%s]\n", join([@sprintf("%.3f", x) for x in design], ", "))
    @printf("    Cs final:  %.4f\n", cs[end])
    @printf("    Cs range:  [%.4f, %.4f]\n", minimum(cs), maximum(cs))
    println()
end

# ---- 7. Is the low ESS the cause? ----
println("=== ESS vs posterior quality ===")
ess_bins = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, Inf)]
@printf("  %15s  %6s  %12s  %12s  %12s\n", "ESS range", "n", "mean|err(K)|", "mean std(K)", "mean std(μ)")
for (lo, hi) in ess_bins
    mask = (ess_adapt .>= lo) .& (ess_adapt .< hi)
    n = sum(mask)
    n == 0 && continue
    @printf("  [%5.1f, %5.1f)  %6d  %12.5f  %12.5f  %12.5f\n",
            lo, hi, n, mean(err_K_adapt[mask]), mean(std_K_adapt[mask]), mean(std_μ_adapt[mask]))
end
println()

# ---- 8. Correlation: design diversity vs posterior quality ----
# How much does the design vary across steps? (measure of "adaptiveness")
design_std_per_trial = [std(all_designs[:, i]) for i in 1:n_trials]
design_max_per_trial = [maximum(all_designs[:, i]) for i in 1:n_trials]
active_steps = [count(x -> x > 0.1, all_designs[:, i]) for i in 1:n_trials]

println("=== Design variability vs posterior quality ===")
@printf("  Correlation: design_std vs std(K)   = %.3f\n",
        cor(design_std_per_trial, Float64.(std_K_adapt)))
@printf("  Correlation: design_max vs std(K)   = %.3f\n",
        cor(design_max_per_trial, Float64.(std_K_adapt)))
@printf("  Correlation: active_steps vs std(K) = %.3f\n",
        cor(Float64.(active_steps), Float64.(std_K_adapt)))
@printf("  Correlation: ESS vs std(K)          = %.3f\n",
        cor(Float64.(ess_adapt), Float64.(std_K_adapt)))

println("\nDone.")
