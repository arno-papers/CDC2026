#!/usr/bin/env julia
# GPU-accelerated paired sPCE evaluation for architecture ablation.
#
# Compares three architectures on identical random seeds per batch:
#   1. Full Transformer (with PE)
#   2. Flat MLP
#   3. Transformer without PE
#
# Usage:
#   julia --project=. examples/weibull/eval_ablation.jl [n_trials=1000] [L=5000] [M=5000] [B=32]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))

using Dates
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  Ablation architectures (must match training scripts exactly)
# ============================================================================

const ablation_flat = @compact(
    mlp = Chain(
        Dense(2 * N_STEPS => 64, gelu),
        Dense(64 => 64, gelu),
        Dense(64 => 1; init_bias=(rng, dims...) -> fill(-2.0f0, dims...)),
    ),
) do x
    x_flat = reshape(x, 2 * N_STEPS, :)
    @return ACTION_HI .* sigmoid.(mlp(x_flat))
end

const ablation_nope = @compact(
    input_proj = Dense(2 => 32),
    rms1 = RMSNorm((32,)),
    mha = MultiHeadAttention(32; nheads = 4),
    rms2 = RMSNorm((32,)),
    ff = Chain(Dense(32 => 64, gelu), Dense(64 => 32)),
    output_head = Dense(32 => 1; init_bias=(rng, dims...) -> fill(-2.0f0, dims...)),
) do x
    x = input_proj(x)
    h = rms1(x)
    attn, _ = mha(h)
    x = x + attn
    x = x + ff(rms2(x))
    @return ACTION_HI .* sigmoid.(output_head(x[:, end, :]))
end

# ============================================================================
#  GPU kernel: adaptive sPCE evaluation (returns per-episode scores)
# ============================================================================

function adaptive_spce_eval(model, ps, st, data)
    θ_full, θ_obs_numer, θ_dyn_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer, n_substeps_val = data

    B = size(ε, 3)

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_dyn_true = θ_full[1:N_PARAMS_DYN, 1:1, :]
    θ_obs_true_3d = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1:1, :]
    θ_obs_true = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1, :]

    u = make_initial_state(u0, θ_dyn_true, θ_obs_true_3d, B)

    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        d = action
        designs[step, :] .= d[1, :]

        u = integrate(u, θ_dyn_true, d, DT, n_substeps_val)

        y_noisy = observe_noisy(u, θ_obs_true, ε, step)

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= d[1, :]
    end

    # DENOMINATOR: all params vary
    n_denom = size(θ_full, 2)
    θ_dyn_denom = θ_full[1:N_PARAMS_DYN, :, :]
    θ_obs_denom = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, :, :]

    u_denom = make_initial_state(u0, θ_dyn_denom, θ_obs_denom, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_denom, observations[step:step, :], u_denom, θ_obs_denom)
    end

    # NUMERATOR: fix k_a, k_tr; resample CL, Q_d, σ_prop, σ_add
    M_N = size(ll_numer, 1)

    u_numer = make_initial_state(u0, θ_dyn_numer, θ_obs_numer, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_numer, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_numer, observations[step:step, :], u_numer, θ_obs_numer)
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
    n_trials    = 1000
    n_substeps  = N_SUBSTEPS
    seed        = 42
    L           = 5000
    M           = 5000
    B           = 32

    for arg in ARGS
        key, val = split(arg, '='; limit=2)
        if     key == "n_trials";  n_trials  = parse(Int, val)
        elseif key == "L";         L         = parse(Int, val)
        elseif key == "M";         M         = parse(Int, val)
        elseif key == "B";         B         = parse(Int, val)
        elseif key == "seed";      seed      = parse(Int, val)
        else   @warn "Unknown argument: $arg"
        end
    end

    Random.seed!(seed)
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    # ---- Model registry ----
    model_defs = [
        (name="full", label="Full Transformer",    model=policy,        subdir="ablation_full"),
        (name="flat", label="Flat MLP",             model=ablation_flat, subdir="ablation_flat"),
        (name="nope", label="Transformer (no PE)",  model=ablation_nope, subdir="ablation_nope"),
    ]

    println("\n=== Architecture Ablation — Paired sPCE Evaluation ===")
    println("n_trials   = $n_trials")
    println("L          = $L")
    println("M          = $M")
    println("B          = $B")
    println("n_substeps = $n_substeps")
    println("seed       = $seed")
    println()
    flush(stdout)

    # ---- Set up GPU ----
    Reactant.set_default_backend("gpu")
    xdev = reactant_device()
    println("Using device: ", xdev)

    n_denom = L + 1
    u0 = make_u0()
    u0_ra = u0 |> xdev

    # ---- Load checkpoints and move to GPU ----
    rng_setup = MersenneTwister(seed)
    model_configs = []
    for md in model_defs
        ckpt_dir = joinpath(@__DIR__, "results", md.subdir)
        @assert isdir(ckpt_dir) "Checkpoint dir not found: $ckpt_dir"
        ps_cpu, _, _ = load_checkpoint_cpu(ckpt_dir)
        _, st_init = Lux.setup(rng_setup, md.model)
        ps_ra = ps_cpu |> xdev
        st_ra = st_init |> xdev
        push!(model_configs, (name=md.name, label=md.label, model=md.model, ps=ps_ra, st=st_ra))
        println("  Loaded $(md.label) from $(md.subdir)/checkpoint.jls")
    end
    println()
    flush(stdout)

    # ---- Batch loop ----
    n_batches = cld(n_trials, B)
    rng = MersenneTwister(seed)

    all_scores = Dict{String, Vector{Float64}}()
    for mc in model_configs
        all_scores[mc.name] = Float64[]
    end

    t_start = time()
    println("Starting evaluation: $n_batches batches of $B episodes")
    println("First batch includes compilation time (~5-15 min per model)...")
    flush(stdout)

    for batch_idx in 1:n_batches
        actual_B = min(B, n_trials - (batch_idx - 1) * B)

        # Shared randomness across all 3 models
        θ_full_cpu = sample_θ_full(rng, n_denom, B)
        θ_dyn_numer_cpu = sample_θ_dyn_numer(rng, θ_full_cpu[1:N_PARAMS_DYN, 1:1, :], M, B)
        θ_obs_numer_cpu = sample_θ_N_joint(rng, M, B)
        ε_cpu = randn(rng, Float32, N_NOISE_CHANNELS, N_STEPS, B)

        θ_full_ra = θ_full_cpu |> xdev
        θ_dyn_numer_ra = θ_dyn_numer_cpu |> xdev
        θ_obs_numer_ra = θ_obs_numer_cpu |> xdev
        ε_ra = ε_cpu |> xdev

        for mc in model_configs
            # Fresh mutable buffers per model
            input_buffer = zeros(Float32, 2, N_STEPS, B) |> xdev
            observations = zeros(Float32, N_STEPS, B) |> xdev
            designs_buf  = zeros(Float32, N_STEPS, B) |> xdev
            ll_denom_buf = zeros(Float32, n_denom, B) |> xdev
            ll_numer_buf = zeros(Float32, M, B) |> xdev

            data = (θ_full_ra, θ_obs_numer_ra, θ_dyn_numer_ra, u0_ra,
                    input_buffer, observations, designs_buf, ε_ra,
                    ll_denom_buf, ll_numer_buf, n_substeps)

            scores_ra, _, _ = @jit adaptive_spce_eval(mc.model, mc.ps, mc.st, data)
            scores_cpu = Array(scores_ra)
            append!(all_scores[mc.name], Float64.(scores_cpu[1, 1:actual_B]))
        end

        t_elapsed = time() - t_start
        if batch_idx == 1
            @printf("  batch %d/%d done (%.1fs, includes compilation)\n", batch_idx, n_batches, t_elapsed)
        elseif batch_idx % 5 == 0 || batch_idx == n_batches
            @printf("  batch %d/%d done (%.1fs total)\n", batch_idx, n_batches, t_elapsed)
        end
        flush(stdout)
    end

    t_total = time() - t_start
    @printf("\nTotal evaluation time: %.1fs\n", t_total)

    # ---- Save raw scores ----
    scores_dict = Dict{String, Any}(
        "n_trials"    => n_trials,
        "L"           => L,
        "M"           => M,
        "B"           => B,
        "n_substeps"  => n_substeps,
        "seed"        => seed,
        "wall_time_s" => t_total,
    )
    for mc in model_configs
        scores_dict["$(mc.name)_scores"] = all_scores[mc.name]
    end
    serialize(joinpath(results_dir, "ablation_scores.jls"), scores_dict)

    # ---- Print and save summary ----
    println("\n=== Results (targeted sPCE, higher = more informative) ===\n")
    summary_lines = String[]

    for mc in model_configs
        scores = all_scores[mc.name]
        m = mean(scores)
        s = std(scores)
        sem = s / sqrt(length(scores))
        line = @sprintf("  %-25s  mean = %8.4f  std = %8.4f  SEM = %6.4f  (n=%d)",
                         mc.label, m, s, sem, length(scores))
        println(line)
        push!(summary_lines, line)
    end
    println()

    # Pairwise paired t-tests (full vs each ablation)
    full_scores = all_scores["full"]
    for mc in model_configs
        mc.name == "full" && continue
        scores = all_scores[mc.name]
        delta = full_scores .- scores
        t_stat = mean(delta) / (std(delta) / sqrt(length(delta)))
        line = @sprintf("  Paired: Full - %-18s  delta = %+.4f ± %.4f (SEM)  t=%6.2f",
                         mc.label, mean(delta), std(delta) / sqrt(length(delta)), t_stat)
        println(line)
        push!(summary_lines, line)
    end
    flush(stdout)

    open(joinpath(results_dir, "ablation_summary.txt"), "w") do io
        println(io, "# Architecture Ablation — Paired sPCE Evaluation")
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
        println(io, "=== Results (targeted sPCE, higher = more informative) ===")
        println(io)
        for line in summary_lines
            println(io, line)
        end
    end

    println("\nDone. Outputs in: $results_dir")
end
