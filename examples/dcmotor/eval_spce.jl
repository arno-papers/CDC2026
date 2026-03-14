#!/usr/bin/env julia
# CPU-based sPCE evaluation for DC motor: adaptive vs static BIM design.
#
# Usage:
#   julia --project=. examples/dcmotor/eval_spce.jl [n_trials=500] [L=500] [M=128]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

using Dates
using Plots
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU helpers
# ============================================================================

function steady_state_cpu(theta_dyn::Vector{Float64})
    k, _, f = theta_dyn
    denom = Float64(R_CONST) * f + k^2
    i_ss = f * Float64(V_PRE) / denom
    ω_ss = k * Float64(V_PRE) / denom
    return Float64[i_ss, ω_ss, 0.0]
end

function log_likelihood_cpu(observations::Vector{Float64}, theta_dyn::Vector{Float64},
                            sigma::Float64, design::AbstractVector;
                            n_substeps::Int=N_SUBSTEPS)
    u = reshape(steady_state_cpu(theta_dyn), 3, 1)
    θ_mat = reshape(theta_dyn, 3, 1)
    σ² = sigma^2
    ll = 0.0
    for step in 1:N_STEPS
        u = integrate_cpu(u, θ_mat, Float64(design[step]), Float64(DT), n_substeps)
        residual = observations[step] - u[2, 1]  # observe ω
        ll -= 0.5 * (residual^2 / σ² + log(σ²))
    end
    return ll
end

function _logsumexp(x::Vector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function spce_score_cpu(observations::Vector{Float64}, design::AbstractVector,
                        denom_samples, numer_f, numer_sigma,
                        theta_T_true::Vector{Float64};
                        n_substeps::Int=N_SUBSTEPS)
    L_plus_1 = length(denom_samples)
    M = length(numer_sigma)

    # Denominator: all params from prior
    ll_denom = Vector{Float64}(undef, L_plus_1)
    for ℓ in 1:L_plus_1
        θ_dyn_ℓ, σ_ℓ = denom_samples[ℓ]
        ll_denom[ℓ] = log_likelihood_cpu(observations, θ_dyn_ℓ, σ_ℓ, design;
                                          n_substeps=n_substeps)
    end
    log_denominator = _logsumexp(ll_denom) - log(Float64(L_plus_1))

    # Numerator: fix target (k, J), resample nuisance (f, σ)
    ll_numer = Vector{Float64}(undef, M)
    for m in 1:M
        θ_dyn_m = Float64[theta_T_true[1], theta_T_true[2], numer_f[m]]
        ll_numer[m] = log_likelihood_cpu(observations, θ_dyn_m, numer_sigma[m], design;
                                          n_substeps=n_substeps)
    end
    log_numerator = _logsumexp(ll_numer) - log(Float64(M))

    return log_numerator - log_denominator
end

function generate_observations_cpu(rng, theta_dyn::Vector{Float64}, sigma::Float64,
                                    design::AbstractVector; n_substeps::Int=N_SUBSTEPS)
    u = reshape(steady_state_cpu(theta_dyn), 3, 1)
    θ_mat = reshape(theta_dyn, 3, 1)
    observations = Vector{Float64}(undef, N_STEPS)
    for step in 1:N_STEPS
        u = integrate_cpu(u, θ_mat, Float64(design[step]), Float64(DT), n_substeps)
        observations[step] = u[2, 1] + sigma * randn(rng)
    end
    return observations
end

function rollout_adaptive_cpu(model, ps_cpu, st_cpu, rng,
                              theta_dyn::Vector{Float64}, sigma::Float64;
                              n_substeps::Int=N_SUBSTEPS)
    u = Float32.(reshape(steady_state_cpu(theta_dyn), 3, 1))
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float64, N_STEPS)
    observations = Vector{Float64}(undef, N_STEPS)
    st_local = st_cpu
    θ_mat = reshape(Float32.(theta_dyn), 3, 1)

    for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        v_in = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = Float64(v_in)

        u = integrate_cpu(u, θ_mat, v_in, DT, n_substeps)
        y_obs = Float64(u[2, 1]) + sigma * randn(rng)
        observations[step] = y_obs
        input_buffer[1, step, 1] = Float32(y_obs)
        input_buffer[2, step, 1] = v_in
    end
    return designs, observations
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    n_trials   = 500
    L          = 500
    M          = 128
    seed       = 42

    results_dir = joinpath(@__DIR__, "results")

    # ---- Load checkpoint and static designs ----
    ps_cpu, st_cpu, _ = load_checkpoint_cpu(results_dir)

    static_designs_eval = load_static_designs(results_dir; T=Float64)
    has_spce_opt = any(p -> p.first == "static_spce", static_designs_eval)

    println("\n=== CPU sPCE Evaluation: DC Motor ===")
    println("n_trials   = $n_trials")
    println("L          = $L")
    println("M          = $M")
    println("seed       = $seed")
    println()
    for (name, d) in static_designs_eval
        println("$name: [", join(round.(d; digits=3), ", "), "]")
    end
    println()
    flush(stdout)

    # ---- Pre-sample shared denominator/numerator draws per trial ----
    rng = MersenneTwister(seed)
    n_denom = L + 1

    all_scores = Dict{String, Vector{Float64}}()
    all_scores["adaptive"] = Float64[]
    for (name, _) in static_designs_eval
        all_scores[name] = Float64[]
    end
    all_adaptive_designs = Matrix{Float64}(undef, N_STEPS, n_trials)

    _, st_init = Lux.setup(MersenneTwister(0), policy)

    t_start = time()
    for trial in 1:n_trials
        # Sample true parameters
        θ_true = sample_θ_full(rng, 1)
        theta_dyn_true = Float64[θ_true[1], θ_true[2], θ_true[3]]
        sigma_true = Float64(θ_true[4])

        # Pre-draw shared samples for sPCE evaluation
        denom_samples = [(Float64[s[1][1], s[1][2], s[1][3]], Float64(s[2]))
                         for s in draw_prior_samples(rng, n_denom)]
        numer_f = Float64.(sample_f(rng, M))
        numer_sigma = Float64[σ_lo + (σ_hi - σ_lo) * rand(rng) for _ in 1:M]

        # --- Adaptive ---
        designs_ad, obs_ad = rollout_adaptive_cpu(policy, ps_cpu, st_init, rng,
                                                   theta_dyn_true, sigma_true)
        all_adaptive_designs[:, trial] .= designs_ad
        score_ad = spce_score_cpu(obs_ad, designs_ad, denom_samples, numer_f, numer_sigma,
                                   theta_dyn_true)
        push!(all_scores["adaptive"], score_ad)

        # --- Static designs ---
        for (name, design) in static_designs_eval
            obs_st = generate_observations_cpu(rng, theta_dyn_true, sigma_true, design)
            score_st = spce_score_cpu(obs_st, design, denom_samples, numer_f, numer_sigma,
                                       theta_dyn_true)
            push!(all_scores[name], score_st)
        end

        if trial % 50 == 0 || trial == 1
            t_elapsed = time() - t_start
            @printf("  trial %d/%d (%.1fs)\n", trial, n_trials, t_elapsed)
            flush(stdout)
        end
    end
    t_total = time() - t_start
    @printf("\nTotal time: %.1fs\n", t_total)

    # ---- Report ----
    println("\n=== Results ===")
    for name in ["adaptive"; [p.first for p in static_designs_eval]]
        scores = all_scores[name]
        @printf("%-15s mean = %8.4f ± %.4f  (median = %8.4f)\n",
                name * ":", mean(scores), std(scores) / sqrt(n_trials), median(scores))
    end
    adaptive_mean = mean(all_scores["adaptive"])
    for (name, _) in static_designs_eval
        Δ = adaptive_mean - mean(all_scores[name])
        @printf("Δ(adaptive - %s) = %.4f\n", name, Δ)
    end

    # ---- Save ----
    scores_dict = Dict{String, Any}(
        "adaptive_scores"   => all_scores["adaptive"],
        "static_std_scores" => all_scores["static_std"],
        "adaptive_designs"  => Float32.(all_adaptive_designs),
        "n_trials"          => n_trials,
        "L"                 => L,
        "M"                 => M,
        "seed"              => seed,
        "wall_time_s"       => t_total,
    )
    if has_spce_opt
        scores_dict["static_spce_scores"] = all_scores["static_spce"]
    end
    serialize(joinpath(results_dir, "spce_scores.jls"), scores_dict)

    # ---- Plots ----
    static_designs_plot = Pair{String, Vector{Float32}}[n => Float32.(d) for (n, d) in static_designs_eval]

    score_designs = extract_designs(scores_dict)
    plot_spce_histograms(score_designs;
        output_path = joinpath(results_dir, "plot_spce_histograms.png"),
        title_suffix = " (L=$L, M=$M, $n_trials trials, CPU)")

    plot_design_trajectories(Float32.(all_adaptive_designs), static_designs_plot;
        output_path = joinpath(results_dir, "plot_spce_trajectories.png"),
        design_ylabel = "Voltage (V)")

    println("\nDone. Outputs in: $results_dir")
end
