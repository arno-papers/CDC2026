#!/usr/bin/env julia
# GPU-accelerated sPCE evaluation using Reactant (@jit, forward-only).
#
# Usage:
#   julia --project=. examples/monod/eval_spce.jl [checkpoint=...] [n_trials=500] [L=1000] [M=128] [B=32]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "args.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

using Dates
using Distributions: TDist, ccdf
using Plots
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU helpers
# ============================================================================

function log_likelihood(observations::Vector{Float64}, theta_T, sigma, Cx0,
                         design::AbstractVector; n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    σ² = Float64(sigma)^2
    ll = 0.0
    for k in 1:N_STEPS
        residual = observations[k] - cs[k]
        ll -= 0.5 * (residual^2 / σ² + log(σ²))
    end
    return ll
end

function _logsumexp(x::Vector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function spce_score(observations::Vector{Float64}, design::AbstractVector,
                     theta_T_true, sigma_true,
                     denom_samples, numer_sigma, numer_Cx0;
                     n_substeps::Int=N_SUBSTEPS)
    L_plus_1 = length(denom_samples)
    M = length(numer_sigma)

    ll_denom = Vector{Float64}(undef, L_plus_1)
    for ℓ in 1:L_plus_1
        θT_ℓ, σ_ℓ, Cx0_ℓ = denom_samples[ℓ]
        ll_denom[ℓ] = log_likelihood(observations, θT_ℓ, σ_ℓ, Cx0_ℓ, design;
                                      n_substeps=n_substeps)
    end
    log_denominator = _logsumexp(ll_denom) - log(Float64(L_plus_1))

    ll_numer = Vector{Float64}(undef, M)
    for m in 1:M
        ll_numer[m] = log_likelihood(observations, theta_T_true, numer_sigma[m],
                                      numer_Cx0[m], design; n_substeps=n_substeps)
    end
    log_numerator = _logsumexp(ll_numer) - log(Float64(M))

    return log_numerator - log_denominator
end

function generate_observations(rng, theta_T, sigma, Cx0, design;
                                n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    observations = [cs[k] + Float64(sigma) * randn(rng) for k in 1:N_STEPS]
    return observations
end

# ============================================================================
#  GPU kernels
# ============================================================================

function adaptive_spce_eval(model, ps, st, data)
    θ_full, σ_numer, Cx0_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer, n_substeps_val = data

    B = size(ε, 2)

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_T_true = θ_full[1:2, 1:1, :]
    σ_true = θ_full[3, 1, :]
    Cx0_true = θ_full[4, 1, :]

    u = vcat(
        repeat(u0[1:1, :, :], 1, 1, B),
        reshape(Cx0_true, 1, 1, B),
        repeat(u0[3:3, :, :], 1, 1, B),
    )

    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        Q_in = action
        designs[step, :] .= Q_in[1, :]

        u = integrate(u, θ_T_true, Q_in, DT, n_substeps_val)

        obs = u[1, 1, :]
        noise = ε[step, :]
        y_noisy = obs .+ σ_true .* noise

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= Q_in[1, :]
    end

    n_denom = size(θ_full, 2)
    θ_T_denom = θ_full[1:2, :, :]
    σ²_denom = (θ_full[3, :, :]) .^ 2
    Cx0_denom = θ_full[4:4, :, :]

    u_denom = vcat(
        repeat(u0[1:1, :, :], 1, n_denom, B),
        Cx0_denom,
        repeat(u0[3:3, :, :], 1, n_denom, B),
    )

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]
        u_denom = integrate(u_denom, θ_T_denom, Q_step, DT, n_substeps_val)

        pred_obs = u_denom[1, :, :]
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred_obs
        ll_denom .-= 0.5f0 .* (residual.^2 ./ σ²_denom .+ log.(σ²_denom))
    end

    σ²_numer = σ_numer .^ 2
    M_N = size(ll_numer, 1)

    u_numer = vcat(
        repeat(u0[1:1, :, :], 1, M_N, B),
        reshape(Cx0_numer, 1, M_N, B),
        repeat(u0[3:3, :, :], 1, M_N, B),
    )

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]
        u_numer = integrate(u_numer, θ_T_true, Q_step, DT, n_substeps_val)

        pred_obs = u_numer[1, :, :]
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred_obs
        ll_numer .-= 0.5f0 .* (residual.^2 ./ σ²_numer .+ log.(σ²_numer))
    end

    ll_max_num = maximum(ll_numer; dims=1)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))
    log_numerator = lse_num .- log(Float32(M_N))

    ll_max_den = maximum(ll_denom; dims=1)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))
    log_denominator = lse_den .- log(Float32(n_denom))

    spce_per_episode = log_numerator .- log_denominator
    return spce_per_episode, st, (;)
end

function static_spce_eval(model, ps, st, data)
    θ_full, σ_numer, Cx0_numer, u0, ε, ll_denom, ll_numer, n_substeps_val = data

    B = size(ε, 2)
    design = ps.layer_1.weight

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_T_true = θ_full[1:2, 1:1, :]
    σ_true = θ_full[3, 1, :]
    Cx0_true = θ_full[4, 1, :]

    u = vcat(
        repeat(u0[1:1, :, :], 1, 1, B),
        reshape(Cx0_true, 1, 1, B),
        repeat(u0[3:3, :, :], 1, 1, B),
    )

    observations = similar(ε)

    for step in 1:N_STEPS
        Q_in = design[step:step, :]
        u = integrate(u, θ_T_true, Q_in, DT, n_substeps_val)
        obs = u[1, 1, :]
        y_noisy = obs .+ σ_true .* ε[step, :]
        observations[step, :] .= y_noisy
    end

    n_denom = size(θ_full, 2)
    θ_T_denom = θ_full[1:2, :, :]
    σ²_denom = (θ_full[3, :, :]) .^ 2
    Cx0_denom = θ_full[4:4, :, :]

    u_denom = vcat(
        repeat(u0[1:1, :, :], 1, n_denom, B),
        Cx0_denom,
        repeat(u0[3:3, :, :], 1, n_denom, B),
    )

    for step in 1:N_STEPS
        Q_step = design[step:step, :]
        u_denom = integrate(u_denom, θ_T_denom, Q_step, DT, n_substeps_val)
        pred_obs = u_denom[1, :, :]
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred_obs
        ll_denom .-= 0.5f0 .* (residual .^ 2 ./ σ²_denom .+ log.(σ²_denom))
    end

    σ²_numer = σ_numer .^ 2
    M_N = size(σ_numer, 1)

    u_numer = vcat(
        repeat(u0[1:1, :, :], 1, M_N, B),
        reshape(Cx0_numer, 1, M_N, B),
        repeat(u0[3:3, :, :], 1, M_N, B),
    )

    for step in 1:N_STEPS
        Q_step = design[step:step, :]
        u_numer = integrate(u_numer, θ_T_true, Q_step, DT, n_substeps_val)
        pred_obs = u_numer[1, :, :]
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred_obs
        ll_numer .-= 0.5f0 .* (residual .^ 2 ./ σ²_numer .+ log.(σ²_numer))
    end

    ll_max_num = maximum(ll_numer; dims=1)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))
    log_numerator = lse_num .- log(Float32(M_N))

    ll_max_den = maximum(ll_denom; dims=1)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))
    log_denominator = lse_den .- log(Float32(n_denom))

    spce_per_episode = log_numerator .- log_denominator
    return spce_per_episode, st, (;)
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    checkpoint       = parse_kwarg(ARGS, "checkpoint"; default=joinpath(@__DIR__, "results"))
    spce_design_path = parse_kwarg(ARGS, "spce_design"; default=nothing)
    n_trials         = parse_int(ARGS, "n_trials"; default=500)
    n_substeps       = parse_int(ARGS, "n_substeps"; default=N_SUBSTEPS)
    L                = parse_int(ARGS, "L"; default=L_CONTRASTIVE)
    M                = parse_int(ARGS, "M"; default=M_NUISANCE)
    B                = parse_int(ARGS, "B"; default=32)
    seed             = parse_int(ARGS, "seed"; default=0)

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    # ---- Load model and static designs ----
    ps_cpu, st_cpu, _ = load_checkpoint_cpu(checkpoint)

    cheating_data = deserialize(joinpath(results_dir, "bim_cheat_design.jls"))
    static_cheating = cheating_data["static_design"]

    standard_data = deserialize(joinpath(results_dir, "bim_std_design.jls"))
    static_standard = standard_data["static_design"]

    has_spce_opt = false
    local static_spce_opt
    if spce_design_path !== nothing
        spce_file = spce_design_path
    else
        spce_file = joinpath(results_dir, "spce_static_design.jls")
    end
    if isfile(spce_file)
        spce_data = deserialize(spce_file)
        static_spce_opt = spce_data["design"]
        has_spce_opt = true
        println("Loaded sPCE-optimized design from: $spce_file")
    else
        println("No sPCE-optimized design found (looked at: $spce_file)")
        static_spce_opt = zeros(Float32, N_STEPS)
    end

    static_designs = Pair{String, Vector{Float32}}[
        "static_std"   => Float32.(static_standard),
        "static_cheat" => Float32.(static_cheating),
    ]
    if has_spce_opt
        push!(static_designs, "static_spce" => Float32.(static_spce_opt))
    end

    println("\n=== GPU-Accelerated Targeted sPCE Evaluation ===")
    println("n_trials   = $n_trials")
    println("L          = $L")
    println("M          = $M")
    println("B          = $B")
    println("n_substeps = $n_substeps")
    println("seed       = $seed")
    println()
    for (name, d) in static_designs
        println("$name: [", join(round.(d; digits=3), ", "), "]")
    end
    println()
    flush(stdout)

    # ---- Set up GPU ----
    Reactant.set_default_backend("gpu")
    xdev = reactant_device()
    println("Using device: ", xdev)

    n_denom = L + 1

    u0 = make_u0()
    u0_ra = u0 |> xdev

    rng_setup = MersenneTwister(seed)
    _, st_init = Lux.setup(rng_setup, policy)
    ps_ra = ps_cpu |> xdev
    st_ra = st_init |> xdev

    make_design_model() = Dense(1 => N_STEPS; use_bias=true)
    static_model = make_design_model()
    ps_static_init, st_static_init = Lux.setup(rng_setup, static_model)
    ps_static_cpu = (layer_1 = (weight = reshape(copy(static_designs[1].second), N_STEPS, 1),
                                bias = zeros(Float32, N_STEPS)),)
    ps_static_ra = ps_static_cpu |> xdev
    st_static_ra = st_static_init |> xdev

    # ---- Batch loop ----
    n_batches = cld(n_trials, B)
    rng = MersenneTwister(seed)

    all_scores = Dict{String, Vector{Float64}}()
    all_scores["adaptive"] = Float64[]
    for (name, _) in static_designs
        all_scores[name] = Float64[]
    end
    all_adaptive_designs = Matrix{Float32}(undef, N_STEPS, n_trials)

    t_start = time()
    println("\nStarting evaluation: $n_batches batches of $B episodes")
    println("First batch includes compilation time (~5-15 min)...")
    flush(stdout)

    for batch_idx in 1:n_batches
        actual_B = min(B, n_trials - (batch_idx - 1) * B)

        θ_full = sample_θ_full(rng, n_denom, B)
        σ_numer, Cx0_numer = sample_θ_N_joint(rng, M, B)
        ε_shared = randn(rng, Float32, N_STEPS, B)

        θ_full_ra = θ_full |> xdev
        σ_numer_ra = σ_numer |> xdev
        Cx0_numer_ra = Cx0_numer |> xdev

        input_buffer = zeros(Float32, 2, N_STEPS, B) |> xdev
        observations = zeros(Float32, N_STEPS, B) |> xdev
        designs_buf = zeros(Float32, N_STEPS, B) |> xdev
        ε_ra = ε_shared |> xdev
        ll_denom_buf = zeros(Float32, n_denom, B) |> xdev
        ll_numer_buf = zeros(Float32, M, B) |> xdev

        data_adaptive = (θ_full_ra, σ_numer_ra, Cx0_numer_ra, u0_ra,
                         input_buffer, observations, designs_buf, ε_ra,
                         ll_denom_buf, ll_numer_buf, n_substeps)

        scores_ra, _, _ = @jit adaptive_spce_eval(policy, ps_ra, st_ra, data_adaptive)
        scores_cpu = Array(scores_ra)
        append!(all_scores["adaptive"], Float64.(scores_cpu[1, 1:actual_B]))

        designs_cpu = Array(designs_buf)
        col_start = (batch_idx - 1) * B + 1
        all_adaptive_designs[:, col_start:col_start + actual_B - 1] .= designs_cpu[:, 1:actual_B]

        for (name, design) in static_designs
            ε_static = randn(rng, Float32, N_STEPS, B)
            ε_static_ra = ε_static |> xdev
            ll_denom_s = zeros(Float32, n_denom, B) |> xdev
            ll_numer_s = zeros(Float32, M, B) |> xdev

            copyto!(ps_static_ra.layer_1.weight, reshape(design, N_STEPS, 1))

            data_static = (θ_full_ra, σ_numer_ra, Cx0_numer_ra, u0_ra,
                           ε_static_ra, ll_denom_s, ll_numer_s, n_substeps)

            scores_s_ra, _, _ = @jit static_spce_eval(static_model, ps_static_ra, st_static_ra, data_static)
            scores_s_cpu = Array(scores_s_ra)
            append!(all_scores[name], Float64.(scores_s_cpu[1, 1:actual_B]))
        end

        t_elapsed = time() - t_start
        if batch_idx == 1
            @printf("  batch %d/%d done (%.1fs, includes compilation)\n", batch_idx, n_batches, t_elapsed)
        elseif batch_idx % 5 == 0 || batch_idx == n_batches
            @printf("  batch %d/%d done (%.1fs total)\n", batch_idx, n_batches, t_elapsed)
        end
        flush(stdout)
    end

    # ---- Mean adaptive as static baseline ----
    avg_adaptive = Float32.(vec(mean(all_adaptive_designs; dims=2)))
    push!(static_designs, "static_avg" => avg_adaptive)
    all_scores["static_avg"] = Float64[]

    println("\nEvaluating mean adaptive design as static baseline...")
    flush(stdout)
    rng_avg = MersenneTwister(seed + 999)
    for batch_idx in 1:n_batches
        actual_B = min(B, n_trials - (batch_idx - 1) * B)
        θ_full = sample_θ_full(rng_avg, n_denom, B)
        σ_numer, Cx0_numer = sample_θ_N_joint(rng_avg, M, B)
        ε_avg = randn(rng_avg, Float32, N_STEPS, B)

        θ_full_ra = θ_full |> xdev
        σ_numer_ra = σ_numer |> xdev
        Cx0_numer_ra = Cx0_numer |> xdev
        ε_avg_ra = ε_avg |> xdev
        ll_denom_a = zeros(Float32, n_denom, B) |> xdev
        ll_numer_a = zeros(Float32, M, B) |> xdev

        copyto!(ps_static_ra.layer_1.weight, reshape(avg_adaptive, N_STEPS, 1))

        data_avg = (θ_full_ra, σ_numer_ra, Cx0_numer_ra, u0_ra,
                     ε_avg_ra, ll_denom_a, ll_numer_a, n_substeps)

        scores_a_ra, _, _ = @jit static_spce_eval(static_model, ps_static_ra, st_static_ra, data_avg)
        scores_a_cpu = Array(scores_a_ra)
        append!(all_scores["static_avg"], Float64.(scores_a_cpu[1, 1:actual_B]))

        if batch_idx % 5 == 0 || batch_idx == n_batches
            @printf("  batch %d/%d\n", batch_idx, n_batches)
            flush(stdout)
        end
    end

    t_total = time() - t_start
    @printf("\nTotal evaluation time: %.1fs\n", t_total)

    # ---- Save ----
    scores_dict = Dict{String, Any}(
        "adaptive_scores"      => all_scores["adaptive"],
        "static_avg_scores"    => all_scores["static_avg"],
        "static_std_scores"    => all_scores["static_std"],
        "static_cheat_scores"  => all_scores["static_cheat"],
        "adaptive_designs"     => all_adaptive_designs,
        "n_trials"             => n_trials,
        "L"                    => L,
        "M"                    => M,
        "B"                    => B,
        "n_substeps"           => n_substeps,
        "seed"                 => seed,
        "wall_time_s"          => t_total,
    )
    if has_spce_opt
        scores_dict["static_spce_scores"] = all_scores["static_spce"]
    end
    serialize(joinpath(results_dir, "spce_scores.jls"), scores_dict)

    score_designs = extract_designs(scores_dict)
    summary = compute_summary_stats(score_designs)
    println()
    println(summary)
    flush(stdout)

    plot_spce_histograms(score_designs;
        output_path = joinpath(results_dir, "plot_spce_histograms.png"),
        title_suffix = " (L=$L, M=$M, $n_trials trials, GPU)")

    plot_design_trajectories(all_adaptive_designs, static_designs;
        output_path = joinpath(results_dir, "plot_spce_trajectories.png"))

    open(joinpath(results_dir, "spce_summary.txt"), "w") do io
        println(io, "# Targeted sPCE Evaluation (GPU)")
        println(io, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "n_trials = $n_trials")
        println(io, "L = $L")
        println(io, "M = $M")
        println(io, "B = $B")
        println(io, "n_substeps = $n_substeps")
        println(io, "seed = $seed")
        @printf(io, "wall_time_s = %.1f\n", t_total)
        println(io)
        println(io, "static_avg_design   = [", join(round.(avg_adaptive; digits=4), ", "), "]")
        println(io, "static_std_design   = [", join(round.(static_standard; digits=4), ", "), "]")
        println(io, "static_cheat_design = [", join(round.(static_cheating; digits=4), ", "), "]")
        if has_spce_opt
            println(io, "static_spce_design  = [", join(round.(static_spce_opt; digits=4), ", "), "]")
        end
        println(io)
        println(io, summary)
    end

    println("\nDone. Outputs in: $results_dir")
end
