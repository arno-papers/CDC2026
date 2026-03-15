#!/usr/bin/env julia
# DC motor: evaluate adaptive sPCE policy vs adaptive BIM baseline.
#
# For each trial: sample true parameters, run both strategies, compute sPCE
# scores and posterior means via importance sampling (CPU).
#
# Saves: examples/dcmotor/results/comparison_results.jls
#
# Usage:
#   julia --project=. examples/dcmotor/eval_comparison.jl [n_trials=500] [N_post=5000] [L=5000] [M=5000]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "adaptive_bim.jl"))

using Dates
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU log-likelihood
# ============================================================================

function log_likelihood_cpu(observations::AbstractVector, theta_dyn, sigma,
                             design::AbstractVector; n_substeps::Int=N_SUBSTEPS)
    ω_pred = omega_trajectory_diff_n(Float64[theta_dyn[1], theta_dyn[2], theta_dyn[3]],
                                      Float64.(design); n_substeps=n_substeps)
    σ2 = Float64(sigma)^2
    ll = 0.0
    for k in eachindex(observations)
        residual = Float64(observations[k]) - ω_pred[k]
        ll -= 0.5 * (residual^2 / σ2 + log(σ2))
    end
    return ll
end

# ============================================================================
#  CPU sPCE score (importance sampling)
# ============================================================================

function _logsumexp(x::AbstractVector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function spce_score_cpu(observations::AbstractVector, design::AbstractVector,
                         theta_T_true, f_true, sigma_true,
                         denom_samples, numer_f, numer_sigma;
                         n_substeps::Int=N_SUBSTEPS)
    L_plus_1 = length(denom_samples)
    M = length(numer_f)

    # Denominator: L+1 full prior samples
    ll_denom = Vector{Float64}(undef, L_plus_1)
    for l in 1:L_plus_1
        theta_all, sigma_l = denom_samples[l]
        ll_denom[l] = log_likelihood_cpu(observations, theta_all, sigma_l, design;
                                          n_substeps=n_substeps)
    end
    log_denominator = _logsumexp(ll_denom) - log(Float64(L_plus_1))

    # Numerator: M nuisance samples (fix target k,J; resample f,σ)
    ll_numer = Vector{Float64}(undef, M)
    for m in 1:M
        theta_dyn_m = Float64[theta_T_true[1], theta_T_true[2], numer_f[m]]
        ll_numer[m] = log_likelihood_cpu(observations, theta_dyn_m, numer_sigma[m], design;
                                          n_substeps=n_substeps)
    end
    log_numerator = _logsumexp(ll_numer) - log(Float64(M))

    return log_numerator - log_denominator
end

# ============================================================================
#  CPU posterior mean (importance sampling)
# ============================================================================

function posterior_mean_cpu(observations::AbstractVector, design::AbstractVector,
                             post_samples; n_substeps::Int=N_SUBSTEPS)
    N_p = length(post_samples)
    ll = Vector{Float64}(undef, N_p)
    for i in 1:N_p
        theta_all, sigma_i = post_samples[i]
        ll[i] = log_likelihood_cpu(observations, theta_all, sigma_i, design;
                                    n_substeps=n_substeps)
    end
    # Numerically stable weights
    ll_max = maximum(ll)
    w = exp.(ll .- ll_max)
    w ./= sum(w)

    # Weighted mean of target params (k, J)
    post_k = sum(w[i] * Float64(post_samples[i][1][1]) for i in 1:N_p)
    post_J = sum(w[i] * Float64(post_samples[i][1][2]) for i in 1:N_p)
    ess = 1.0 / sum(w .^ 2)

    return Float64[post_k, post_J], ess
end

# ============================================================================
#  sPCE rollout (CPU, with timing)
# ============================================================================

function rollout_spce_timed(model, ps_cpu, st_cpu, rng,
                             k_true, J_true, f_true, sigma_true;
                             n_substeps::Int=N_SUBSTEPS)
    theta_dyn = reshape(Float32[k_true, J_true, f_true], 3, 1)
    k = Float32(k_true); f = Float32(f_true)
    d = Float32(R_CONST) * f + k^2
    u = reshape(Float32[f * Float32(V_PRE) / d, k * Float32(V_PRE) / d, 0.0f0], 3, 1)

    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float64, N_STEPS)
    observations = zeros(Float64, N_STEPS)
    step_times_s = zeros(Float64, N_STEPS)
    st_local = st_cpu

    @inbounds for step in 1:N_STEPS
        t0 = time_ns()
        action, st_local = model(input_buffer, ps_cpu, st_local)
        t1 = time_ns()
        step_times_s[step] = (t1 - t0) / 1e9

        v_in = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = Float64(v_in)
        u = integrate_cpu(u, theta_dyn, v_in, DT, n_substeps)
        y_obs = Float64(u[2, 1]) + sigma_true * randn(rng)
        observations[step] = y_obs
        input_buffer[1, step, 1] = Float32(y_obs)
        input_buffer[2, step, 1] = v_in
    end

    return (; designs, observations, step_times_s)
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    n_trials    = 500
    L           = 5000
    M           = 5000
    N_post      = 5000
    n_substeps  = N_SUBSTEPS
    seed        = 0
    map_iters   = 100
    n_grid      = 100

    for arg in ARGS
        key, val = split(arg, '='; limit=2)
        if     key == "n_trials";   global n_trials   = parse(Int, val)
        elseif key == "L";          global L          = parse(Int, val)
        elseif key == "M";          global M          = parse(Int, val)
        elseif key == "N_post";     global N_post     = parse(Int, val)
        elseif key == "seed";       global seed       = parse(Int, val)
        elseif key == "map_iters";  global map_iters  = parse(Int, val)
        elseif key == "n_grid";     global n_grid     = parse(Int, val)
        else   @warn "Unknown argument: $arg"
        end
    end

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    checkpoint_path = joinpath(results_dir, "checkpoint.jls")
    @assert isfile(checkpoint_path) "Checkpoint not found: $checkpoint_path. Run training first."

    ckpt = deserialize(checkpoint_path)
    ps_cpu = ckpt["parameters"]
    st_cpu = ckpt["states"]

    println("\n=== DC Motor: Adaptive sPCE vs Adaptive BIM ===")
    println("n_trials   = $n_trials")
    println("L          = $L")
    println("M          = $M")
    println("N_post     = $N_post")
    println("map_iters  = $map_iters")
    println("n_grid     = $n_grid")
    println("seed       = $seed")
    println()
    flush(stdout)

    # Pre-draw shared evaluation samples (denom for sPCE, posterior samples)
    rng_samples = MersenneTwister(seed + 1000)
    denom_samples = draw_prior_samples(rng_samples, L + 1)
    post_samples = draw_prior_samples(rng_samples, N_post)

    # Warm up sPCE policy (JIT compilation)
    warmup_rng = MersenneTwister(999)
    rollout_spce_timed(policy, ps_cpu, st_cpu, warmup_rng,
                       0.5, 0.025, 0.01, 1.0; n_substeps=n_substeps)

    # Storage
    spce_scores_spce = zeros(Float64, n_trials)
    spce_scores_bim  = zeros(Float64, n_trials)
    post_means_spce  = zeros(Float64, 2, n_trials)
    post_means_bim   = zeros(Float64, 2, n_trials)
    ess_spce         = zeros(Float64, n_trials)
    ess_bim          = zeros(Float64, n_trials)
    spce_step_times  = zeros(Float64, N_STEPS, n_trials)
    bim_step_times   = zeros(Float64, N_STEPS, n_trials)

    t_start = time()

    for trial in 1:n_trials
        # Sample true parameters from prior
        rng_trial = MersenneTwister(seed + trial)
        k_t = Float64(k_lo) + (Float64(k_hi) - Float64(k_lo)) * rand(rng_trial)
        J_t = Float64(J_lo) + (Float64(J_hi) - Float64(J_lo)) * rand(rng_trial)
        f_t = Float64(f_lo) + (Float64(f_hi) - Float64(f_lo)) * rand(rng_trial)
        σ_t = Float64(σ_lo) + (Float64(σ_hi) - Float64(σ_lo)) * rand(rng_trial)
        theta_T_true = Float64[k_t, J_t]

        # Per-trial RNGs with fixed offsets (independent of n_trials)
        rng_nuis = MersenneTwister(seed + 100_000 + trial)
        rng_spce = MersenneTwister(seed + 200_000 + trial)
        rng_bim  = MersenneTwister(seed + 300_000 + trial)

        # Nuisance samples for sPCE numerator (fix k,J; resample f,σ)
        numer_f = Float64[Float64(f_lo) + (Float64(f_hi) - Float64(f_lo)) * rand(rng_nuis) for _ in 1:M]
        numer_sigma = Float64[Float64(σ_lo) + (Float64(σ_hi) - Float64(σ_lo)) * rand(rng_nuis) for _ in 1:M]

        # --- sPCE policy rollout ---
        res_spce = rollout_spce_timed(policy, ps_cpu, st_cpu, rng_spce,
                                       k_t, J_t, f_t, σ_t; n_substeps=n_substeps)
        spce_step_times[:, trial] .= res_spce.step_times_s

        spce_scores_spce[trial] = spce_score_cpu(
            res_spce.observations, res_spce.designs, theta_T_true, f_t, σ_t,
            denom_samples, numer_f, numer_sigma; n_substeps=n_substeps)

        pm_spce, ess_s = posterior_mean_cpu(res_spce.observations, res_spce.designs,
                                             post_samples; n_substeps=n_substeps)
        post_means_spce[:, trial] .= pm_spce
        ess_spce[trial] = ess_s

        # --- Adaptive BIM rollout ---
        res_bim = rollout_adaptive_bim(rng_bim, k_t, J_t, f_t, σ_t;
                                        n_grid=n_grid, map_iters=map_iters,
                                        n_substeps=n_substeps)
        bim_step_times[:, trial] .= res_bim.step_times_s

        spce_scores_bim[trial] = spce_score_cpu(
            res_bim.observations, res_bim.designs, theta_T_true, f_t, σ_t,
            denom_samples, numer_f, numer_sigma; n_substeps=n_substeps)

        pm_bim, ess_b = posterior_mean_cpu(res_bim.observations, res_bim.designs,
                                            post_samples; n_substeps=n_substeps)
        post_means_bim[:, trial] .= pm_bim
        ess_bim[trial] = ess_b

        if trial % 50 == 0 || trial == 1
            elapsed = time() - t_start
            @printf("  trial %d/%d (%.1fs)\n", trial, n_trials, elapsed)
            flush(stdout)
        end
    end

    t_total = time() - t_start
    @printf("\nTotal evaluation time: %.1fs\n", t_total)

    # --- Summary ---
    println("\n=== Results ===\n")

    # Sample true params for RMSE (re-draw with same seeds)
    true_k = zeros(Float64, n_trials)
    true_J = zeros(Float64, n_trials)
    for trial in 1:n_trials
        rng_trial = MersenneTwister(seed + trial)
        true_k[trial] = Float64(k_lo) + (Float64(k_hi) - Float64(k_lo)) * rand(rng_trial)
        true_J[trial] = Float64(J_lo) + (Float64(J_hi) - Float64(J_lo)) * rand(rng_trial)
    end

    for (name, scores, pm, ess_vals, step_times) in [
        ("Adaptive (sPCE)", spce_scores_spce, post_means_spce, ess_spce, spce_step_times),
        ("Adaptive (BIM)",  spce_scores_bim,  post_means_bim,  ess_bim,  bim_step_times),
    ]
        m = mean(scores)
        sem = std(scores) / sqrt(length(scores))
        rmse_k = sqrt(mean((pm[1, :] .- true_k) .^ 2))
        rmse_J = sqrt(mean((pm[2, :] .- true_J) .^ 2))
        med_ess = median(ess_vals)
        med_time = median(vec(step_times))
        @printf("  %-20s  sPCE = %6.3f +/- %.3f  RMSE(k) = %.4f  RMSE(J) = %.5f  ESS = %.0f  time/step = %s\n",
                name, m, sem, rmse_k, rmse_J, med_ess,
                med_time < 0.001 ? @sprintf("%.0f us", med_time * 1e6) : @sprintf("%.3f s", med_time))
    end

    # Paired test
    delta = spce_scores_spce .- spce_scores_bim
    t_stat = mean(delta) / (std(delta) / sqrt(length(delta)))
    @printf("\n  Paired: sPCE - BIM  delta = %+.4f +/- %.4f (SEM)  t=%.2f\n",
            mean(delta), std(delta) / sqrt(length(delta)), t_stat)
    flush(stdout)

    # --- Save ---
    results = Dict{String, Any}(
        "spce_scores_spce"  => spce_scores_spce,
        "spce_scores_bim"   => spce_scores_bim,
        "post_means_spce"   => post_means_spce,
        "post_means_bim"    => post_means_bim,
        "ess_spce"          => ess_spce,
        "ess_bim"           => ess_bim,
        "spce_step_times"    => spce_step_times,
        "bim_step_times"     => bim_step_times,
        "true_k"            => true_k,
        "true_J"            => true_J,
        "n_trials"          => n_trials,
        "L"                 => L,
        "M"                 => M,
        "N_post"            => N_post,
        "map_iters"         => map_iters,
        "n_grid"            => n_grid,
        "seed"              => seed,
        "wall_time_s"       => t_total,
    )
    outfile = joinpath(results_dir, "comparison_results.jls")
    serialize(outfile, results)
    println("\nSaved: $outfile")

    # --- Summary file ---
    open(joinpath(results_dir, "comparison_summary.txt"), "w") do io
        println(io, "# DC Motor Comparison: Adaptive sPCE vs Adaptive BIM")
        println(io, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "n_trials = $n_trials")
        println(io, "L = $L, M = $M, N_post = $N_post")
        println(io, "map_iters = $map_iters, n_grid = $n_grid")
        @printf(io, "wall_time_s = %.1f\n", t_total)
        println(io)
        for (name, scores, pm, step_times) in [
            ("Adaptive (sPCE)", spce_scores_spce, post_means_spce, spce_step_times),
            ("Adaptive (BIM)",  spce_scores_bim,  post_means_bim,  bim_step_times),
        ]
            m = mean(scores)
            sem = std(scores) / sqrt(length(scores))
            rmse_k = sqrt(mean((pm[1, :] .- true_k) .^ 2))
            rmse_J = sqrt(mean((pm[2, :] .- true_J) .^ 2))
            med_time = median(vec(step_times))
            @printf(io, "%-20s  sPCE = %6.3f +/- %.3f  RMSE(k) = %.4f  RMSE(J) = %.5f  time/step = %s\n",
                    name, m, sem, rmse_k, rmse_J,
                    med_time < 0.001 ? @sprintf("%.0f us", med_time * 1e6) : @sprintf("%.3f s", med_time))
        end
        @printf(io, "\nPaired: sPCE - BIM  delta = %+.4f +/- %.4f  t=%.2f\n",
                mean(delta), std(delta) / sqrt(length(delta)), t_stat)
    end
end
