#!/usr/bin/env julia
# Compare average adaptive design vs cheating-optimized static design
# on the cheating BIM criterion, with per-trial evaluation and plot.
#
# Usage:
#   julia --project compare_avg_adaptive.jl [checkpoint=...] [n_substeps=100]

include("common_core.jl")
include("compare_static_bim.jl")

using Serialization
using Statistics
using Printf

# ============================================================================
#  Main
# ============================================================================

checkpoint   = parse_kwarg(ARGS, "checkpoint"; default="results/joint-nuisance-initfix")
n_substeps   = parse_int(ARGS, "n_substeps"; default=N_SUBSTEPS)
n_cx0        = parse_int(ARGS, "n_cx0"; default=1024)
n_trials     = parse_int(ARGS, "n_trials"; default=200)
seed         = parse_int(ARGS, "seed"; default=42)
true_mu_max  = parse_float64(ARGS, "true_mu_max"; default=Float64(μ_max_lo + μ_max_hi) / 2)
true_K_s     = parse_float64(ARGS, "true_K_s"; default=Float64(K_s_lo + K_s_hi) / 2)
true_sigma   = parse_float64(ARGS, "true_sigma"; default=Float64(σ_lo + σ_hi) / 2)

cheating_dir = joinpath(checkpoint, "bim_comparison_cheating")
output_dir   = joinpath(checkpoint, "bim_avg_adaptive")
mkpath(output_dir)

# Load the saved cheating comparison (contains adaptive_designs matrix)
comp = deserialize(joinpath(cheating_dir, "bim_comparison.jls"))
adaptive_designs = comp["adaptive_designs"]  # N_STEPS × n_trials
static_design    = comp["static_design"]

# Compute mean adaptive design
avg_adaptive = Float32.(vec(mean(adaptive_designs; dims=2)))

println("=== Average Adaptive Design vs Cheating-Optimized Static ===")
println()
println("Cheating static design: [", join(round.(static_design; digits=3), ", "), "]")
println("Avg adaptive design:    [", join(round.(avg_adaptive; digits=3), ", "), "]")
println()

# Evaluate both on the cheating BIM criterion (expectation over Cx0)
theta_T_true = Float32[Float32(true_mu_max), Float32(true_K_s)]
sigma_true   = Float32(true_sigma)

rng = MersenneTwister(seed)
cx0_samples = sample_cx0(rng, n_cx0)

score_static = bim_logdet_cheating(Float64.(static_design), theta_T_true, sigma_true,
                                    cx0_samples; n_substeps=n_substeps)
score_avg    = bim_logdet_cheating(Float64.(avg_adaptive), theta_T_true, sigma_true,
                                    cx0_samples; n_substeps=n_substeps)

@printf("Cheating BIM logdet (static optimized): %.7f\n", score_static)
@printf("Cheating BIM logdet (avg adaptive):     %.7f\n", score_avg)
@printf("Difference (avg - static):              %.7f\n", score_avg - score_static)
println()

# ============================================================================
#  Per-trial evaluation: adaptive vs static vs avg-adaptive
#  Scoring: logdet(Schur(Σ⁻¹_prior + F(θ,σ,Cx0,ξ)) + ridge·I)
# ============================================================================

println("--- Per-trial evaluation ($n_trials trials) ---")
flush(stdout)

ridge = 1e-6
I2 = Matrix{Float64}(I, 2, 2)

# Load model for adaptive rollouts
ps_cpu, st_cpu, _ = load_checkpoint_cpu(checkpoint)

rng_eval = MersenneTwister(seed + 2)

adaptive_scores  = Vector{Float64}(undef, n_trials)
static_scores    = Vector{Float64}(undef, n_trials)
avgadapt_scores  = Vector{Float64}(undef, n_trials)
trial_designs    = Matrix{Float32}(undef, N_STEPS, n_trials)

for i in 1:n_trials
    # Fixed θ_T and σ, resample only Cx0
    Cx0 = Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng_eval, Float32)

    # Adaptive rollout
    d_adapt = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng_eval,
                                           theta_T_true, sigma_true;
                                           Cx0=Cx0, n_substeps=n_substeps)
    trial_designs[:, i] .= d_adapt

    # FIM + prior for each design
    F_a = fim_matrix(theta_T_true, sigma_true, Cx0, d_adapt; n_substeps=n_substeps)
    F_s = fim_matrix(theta_T_true, sigma_true, Cx0, static_design; n_substeps=n_substeps)
    F_v = fim_matrix(theta_T_true, sigma_true, Cx0, avg_adaptive; n_substeps=n_substeps)
    for k in 1:3
        F_a[k, k] += PRIOR_PREC[k]
        F_s[k, k] += PRIOR_PREC[k]
        F_v[k, k] += PRIOR_PREC[k]
    end
    adaptive_scores[i]  = logdet(Symmetric(schur_complement_2x2(F_a) + ridge * I2))
    static_scores[i]    = logdet(Symmetric(schur_complement_2x2(F_s) + ridge * I2))
    avgadapt_scores[i]  = logdet(Symmetric(schur_complement_2x2(F_v) + ridge * I2))

    if i % 50 == 0
        @printf("  trial %d/%d\n", i, n_trials)
        flush(stdout)
    end
end

delta_as = adaptive_scores .- static_scores
delta_av = adaptive_scores .- avgadapt_scores
delta_vs = avgadapt_scores .- static_scores

println()
@printf("  Adaptive:        %.5f ± %.5f\n", mean(adaptive_scores), std(adaptive_scores))
@printf("  Static (opt):    %.5f ± %.5f\n", mean(static_scores), std(static_scores))
@printf("  Avg adaptive:    %.5f ± %.5f\n", mean(avgadapt_scores), std(avgadapt_scores))
println()
@printf("  Adaptive - Static:      %.5f ± %.5f  (win %.1f%%)\n",
        mean(delta_as), std(delta_as), 100 * mean(delta_as .> 0))
@printf("  Adaptive - AvgAdaptive: %.5f ± %.5f  (win %.1f%%)\n",
        mean(delta_av), std(delta_av), 100 * mean(delta_av .> 0))
@printf("  AvgAdaptive - Static:   %.5f ± %.5f  (win %.1f%%)\n",
        mean(delta_vs), std(delta_vs), 100 * mean(delta_vs .> 0))
flush(stdout)

# ============================================================================
#  Plot
# ============================================================================

steps = collect(1:N_STEPS)
n_show = min(50, n_trials)

# --- Top: design rollouts ---
p1 = plot(; xlabel="step", ylabel="Q_in (L/h)",
          title="Design comparison", legend=:topleft, ylims=(-0.5, 10.5))
for j in 1:n_show
    plot!(p1, steps, trial_designs[:, j];
          color=:dodgerblue, alpha=0.2, lw=1,
          label=(j == 1 ? "adaptive rollouts" : ""))
end
plot!(p1, steps, static_design; color=:crimson, lw=3, label="static cheating-BIM")
plot!(p1, steps, avg_adaptive; color=:darkgreen, lw=3, ls=:dash, label="avg adaptive")

# --- Bottom: score scatter (3-way) ---
lo = min(minimum(static_scores), minimum(avgadapt_scores), minimum(adaptive_scores)) - 0.3
hi = max(maximum(static_scores), maximum(avgadapt_scores), maximum(adaptive_scores)) + 0.3
p2 = plot(; xlabel="static design score", ylabel="score",
          title=@sprintf("Per-trial BIM scores (prior included)"),
          legend=:topleft, xlims=(lo, hi), ylims=(lo, hi))
plot!(p2, [lo, hi], [lo, hi]; color=:black, lw=1, ls=:dash, label="")
scatter!(p2, static_scores, adaptive_scores;
         color=:dodgerblue, alpha=0.4, ms=4, msw=0,
         label=@sprintf("adaptive (Δ=%.2f, win %.0f%%)", mean(delta_as), 100*mean(delta_as .> 0)))
scatter!(p2, static_scores, avgadapt_scores;
         color=:darkgreen, alpha=0.4, ms=4, msw=0, markershape=:diamond,
         label=@sprintf("avg adaptive (Δ=%.2f, win %.0f%%)", mean(delta_vs), 100*mean(delta_vs .> 0)))

p = plot(p1, p2; layout=(2, 1), size=(950, 800))
out_png = joinpath(output_dir, "plot_avg_adaptive_comparison.png")
savefig(p, out_png)
println("\nSaved: $out_png")

# Save summary
open(joinpath(output_dir, "avg_adaptive_summary.txt"), "w") do io
    println(io, "# Average Adaptive Design vs Cheating-Optimized Static")
    println(io, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "# Evaluation: logdet(Schur(Σ⁻¹_prior + F) + ridge·I)")
    println(io)
    println(io, "static_design = [", join(round.(static_design; digits=4), ", "), "]")
    println(io, "avg_adaptive  = [", join(round.(avg_adaptive; digits=4), ", "), "]")
    println(io)
    @printf(io, "cheating_bim_static = %.7f\n", score_static)
    @printf(io, "cheating_bim_avg    = %.7f\n", score_avg)
    @printf(io, "cheating_diff       = %.7f\n", score_avg - score_static)
    println(io)
    println(io, "# Per-trial evaluation ($n_trials trials, seed=$seed)")
    @printf(io, "adaptive_mean     = %.7f ± %.7f\n", mean(adaptive_scores), std(adaptive_scores))
    @printf(io, "static_mean       = %.7f ± %.7f\n", mean(static_scores), std(static_scores))
    @printf(io, "avgadapt_mean     = %.7f ± %.7f\n", mean(avgadapt_scores), std(avgadapt_scores))
    println(io)
    @printf(io, "delta_adapt_static  = %.7f ± %.7f  (win %.1f%%)\n",
            mean(delta_as), std(delta_as), 100*mean(delta_as .> 0))
    @printf(io, "delta_adapt_avg     = %.7f ± %.7f  (win %.1f%%)\n",
            mean(delta_av), std(delta_av), 100*mean(delta_av .> 0))
    @printf(io, "delta_avg_static    = %.7f ± %.7f  (win %.1f%%)\n",
            mean(delta_vs), std(delta_vs), 100*mean(delta_vs .> 0))
    println(io)
    println(io, "n_trials = $n_trials")
    println(io, "n_cx0 = $n_cx0")
    println(io, "n_substeps = $n_substeps")
    println(io, "ridge = $ridge")
end

println("Saved: $(joinpath(output_dir, "avg_adaptive_summary.txt"))")
