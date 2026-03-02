#!/usr/bin/env julia
# GPU-accelerated posterior mean evaluation.
#
# Usage:
#   julia --project=. examples/monod/eval_posterior.jl [checkpoint=...] [n_trials=200] [N_post=5000] [B=32]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "args.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

using Dates
using LinearAlgebra: norm
using Plots
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU observation generation
# ============================================================================

function generate_observations(rng, theta_T, sigma, Cx0, design;
                                n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    return Float32[cs[k] + Float64(sigma) * randn(rng) for k in 1:N_STEPS]
end

# ============================================================================
#  2D convex hull (Graham scan) + 95% confidence region
# ============================================================================

function convex_hull_2d(px::Vector{T}, py::Vector{T}) where T
    n = length(px)
    n <= 2 && return (copy(px), copy(py))
    idx = sortperm(collect(zip(px, py)))
    sx, sy = px[idx], py[idx]
    cross(ox, oy, ax, ay, bx, by) = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    lower_x, lower_y = T[], T[]
    for i in 1:n
        while length(lower_x) >= 2 && cross(lower_x[end-1], lower_y[end-1],
                lower_x[end], lower_y[end], sx[i], sy[i]) <= 0
            pop!(lower_x); pop!(lower_y)
        end
        push!(lower_x, sx[i]); push!(lower_y, sy[i])
    end
    upper_x, upper_y = T[], T[]
    for i in n:-1:1
        while length(upper_x) >= 2 && cross(upper_x[end-1], upper_y[end-1],
                upper_x[end], upper_y[end], sx[i], sy[i]) <= 0
            pop!(upper_x); pop!(upper_y)
        end
        push!(upper_x, sx[i]); push!(upper_y, sy[i])
    end
    hx = vcat(lower_x[1:end-1], upper_x[1:end-1])
    hy = vcat(lower_y[1:end-1], upper_y[1:end-1])
    push!(hx, hx[1]); push!(hy, hy[1])
    return (hx, hy)
end

function confidence_hull_95(px::AbstractVector, py::AbstractVector)
    cx, cy = mean(px), mean(py)
    dists = [norm((px[i] - cx, py[i] - cy)) for i in eachindex(px)]
    keep = sortperm(dists)[1:round(Int, 0.95 * length(px))]
    return convex_hull_2d(Float64.(px[keep]), Float64.(py[keep]))
end

# ============================================================================
#  GPU kernel: importance-sampling posterior mean
# ============================================================================

function posterior_mean_eval(model, ps, st, data)
    θ_post, u0, observations, design_mat, ll_buf, n_substeps_val = data

    B = size(observations, 2)
    N_p = size(θ_post, 2)

    ll_buf .= 0.0f0

    θ_T = θ_post[1:2, :, :]
    σ² = θ_post[3, :, :] .^ 2
    Cx0 = θ_post[4:4, :, :]

    u = vcat(
        repeat(u0[1:1, :, :], 1, N_p, B),
        Cx0,
        repeat(u0[3:3, :, :], 1, N_p, B),
    )

    for step in 1:N_STEPS
        Q_step = design_mat[step:step, :]
        u = integrate(u, θ_T, Q_step, DT, n_substeps_val)

        pred_obs = u[1, :, :]
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred_obs
        ll_buf .-= 0.5f0 .* (residual .^ 2 ./ σ² .+ log.(σ²))
    end

    ll_max = maximum(ll_buf; dims=1)
    w = exp.(ll_buf .- ll_max)
    w = w ./ sum(w; dims=1)

    μ_vals = θ_post[1, :, :]
    K_vals = θ_post[2, :, :]

    post_μ = sum(w .* μ_vals; dims=1)
    post_K = sum(w .* K_vals; dims=1)
    ess = 1.0f0 ./ sum(w .^ 2; dims=1)

    return vcat(post_μ, post_K, ess), st, (;)
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    checkpoint       = parse_kwarg(ARGS, "checkpoint"; default=joinpath(@__DIR__, "results"))
    spce_design_path = parse_kwarg(ARGS, "spce_design"; default=nothing)
    n_trials         = parse_int(ARGS, "n_trials"; default=200)
    N_post           = parse_int(ARGS, "N_post"; default=5000)
    B                = parse_int(ARGS, "B"; default=32)
    n_substeps       = parse_int(ARGS, "n_substeps"; default=N_SUBSTEPS)
    seed             = parse_int(ARGS, "seed"; default=0)

    true_μ = Float32(parse_float64(ARGS, "true_mu_max"; default=Float64(μ_max_lo + μ_max_hi) / 2))
    true_K = Float32(parse_float64(ARGS, "true_K_s";    default=Float64(K_s_lo + K_s_hi) / 2))
    true_σ = Float32(parse_float64(ARGS, "true_sigma";  default=Float64(σ_lo + σ_hi) / 2))
    true_Cx0 = Float32(parse_float64(ARGS, "true_Cx0";  default=Float64(Cx0_lo + Cx0_hi) / 2))
    θT = Float32[true_μ, true_K]

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

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

    println("\n=== GPU-Accelerated Posterior Mean Evaluation ===")
    println("n_trials   = $n_trials")
    println("N_post     = $N_post")
    println("B          = $B")
    println("n_substeps = $n_substeps")
    println("seed       = $seed")
    @printf("true_mu_max = %.4f\n", true_μ)
    @printf("true_K_s    = %.4f\n", true_K)
    @printf("true_sigma  = %.4f\n", true_σ)
    @printf("true_Cx0    = %.4f\n", true_Cx0)
    println()
    for (name, d) in static_designs
        println("$name: [", join(round.(d; digits=3), ", "), "]")
    end
    println()
    flush(stdout)

    Reactant.set_default_backend("gpu")
    xdev = reactant_device()
    println("Using device: ", xdev)

    u0 = make_u0()
    u0_ra = u0 |> xdev

    rng_setup = MersenneTwister(seed)
    dummy_model = Dense(1 => 1)
    ps_dummy, st_dummy = Lux.setup(rng_setup, dummy_model)
    ps_dummy_ra = ps_dummy |> xdev
    st_dummy_ra = st_dummy |> xdev

    n_batches = cld(n_trials, B)
    rng = MersenneTwister(seed)
    rng_post = MersenneTwister(seed + 42)

    design_names = String["adaptive"; [n for (n, _) in static_designs]]
    all_post_means = Dict(name => Matrix{Float32}(undef, 2, n_trials) for name in design_names)
    all_ess        = Dict(name => Vector{Float32}(undef, n_trials) for name in design_names)
    all_adaptive_designs = Matrix{Float32}(undef, N_STEPS, n_trials)

    t_start = time()
    println("\nStarting evaluation: $n_batches batches of $B episodes")
    println("First batch includes compilation time (~5-15 min)...")
    flush(stdout)

    for batch_idx in 1:n_batches
        trial_start = (batch_idx - 1) * B + 1
        actual_B = min(B, n_trials - trial_start + 1)

        obs_dict = Dict(name => zeros(Float32, N_STEPS, B) for name in design_names)
        design_dict = Dict(name => zeros(Float32, N_STEPS, B) for name in design_names)

        for b in 1:actual_B
            trial_idx = trial_start + b - 1

            rng_rollout = MersenneTwister(seed + trial_idx)
            d_adapt = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng_rollout, θT, true_σ;
                                                   Cx0=true_Cx0, n_substeps=n_substeps)
            all_adaptive_designs[:, trial_idx] .= d_adapt
            design_dict["adaptive"][:, b] .= d_adapt

            obs_adapt = generate_observations(MersenneTwister(seed + trial_idx),
                                               θT, true_σ, true_Cx0, d_adapt; n_substeps=n_substeps)
            obs_dict["adaptive"][:, b] .= obs_adapt

            for (i, (name, design)) in enumerate(static_designs)
                obs_static = generate_observations(MersenneTwister(seed + i * n_trials + trial_idx),
                                                    θT, true_σ, true_Cx0, design; n_substeps=n_substeps)
                obs_dict[name][:, b] .= obs_static
                design_dict[name][:, b] .= design
            end
        end

        θ_post = sample_θ_full(rng_post, N_post, B)
        θ_post_ra = θ_post |> xdev
        ll_buf = zeros(Float32, N_post, B) |> xdev

        for name in design_names
            obs_ra = obs_dict[name] |> xdev
            design_ra = design_dict[name] |> xdev

            data = (θ_post_ra, u0_ra, obs_ra, design_ra, ll_buf, n_substeps)
            result_ra, _, _ = @jit posterior_mean_eval(dummy_model, ps_dummy_ra, st_dummy_ra, data)
            result_cpu = Array(result_ra)

            for b in 1:actual_B
                trial_idx = trial_start + b - 1
                all_post_means[name][:, trial_idx] .= result_cpu[1:2, b]
                all_ess[name][trial_idx] = result_cpu[3, b]
            end
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
    push!(design_names, "static_avg")
    all_post_means["static_avg"] = Matrix{Float32}(undef, 2, n_trials)
    all_ess["static_avg"] = Vector{Float32}(undef, n_trials)

    println("\nEvaluating mean adaptive design as static baseline...")
    println("Mean adaptive: [", join(round.(avg_adaptive; digits=3), ", "), "]")
    flush(stdout)

    rng_post_avg = MersenneTwister(seed + 42)
    for batch_idx in 1:n_batches
        trial_start = (batch_idx - 1) * B + 1
        actual_B = min(B, n_trials - trial_start + 1)

        obs_avg = zeros(Float32, N_STEPS, B)
        design_avg = repeat(reshape(avg_adaptive, N_STEPS, 1), 1, B)

        for b in 1:actual_B
            trial_idx = trial_start + b - 1
            obs = generate_observations(MersenneTwister(seed + 5 * n_trials + trial_idx),
                                         θT, true_σ, true_Cx0, avg_adaptive; n_substeps=n_substeps)
            obs_avg[:, b] .= obs
        end

        θ_post = sample_θ_full(rng_post_avg, N_post, B)
        θ_post_ra = θ_post |> xdev
        ll_buf = zeros(Float32, N_post, B) |> xdev

        obs_ra = obs_avg |> xdev
        design_ra = design_avg |> xdev

        data = (θ_post_ra, u0_ra, obs_ra, design_ra, ll_buf, n_substeps)
        result_ra, _, _ = @jit posterior_mean_eval(dummy_model, ps_dummy_ra, st_dummy_ra, data)
        result_cpu = Array(result_ra)

        for b in 1:actual_B
            trial_idx = trial_start + b - 1
            all_post_means["static_avg"][:, trial_idx] .= result_cpu[1:2, b]
            all_ess["static_avg"][trial_idx] = result_cpu[3, b]
        end

        if batch_idx % 5 == 0 || batch_idx == n_batches
            @printf("  batch %d/%d\n", batch_idx, n_batches)
            flush(stdout)
        end
    end

    t_total = time() - t_start
    @printf("\nTotal evaluation time: %.1fs\n", t_total)

    # ---- Summary ----
    println("\n=== Posterior Mean Quality ===\n")

    header = @sprintf("  %-30s  %10s %10s %10s %10s %10s",
                       "Design", "RMSE(mu)", "RMSE(K)", "Bias(mu)", "Bias(K)", "med ESS")
    sep    = @sprintf("  %-30s  %10s %10s %10s %10s %10s",
                       "-"^30, "-"^10, "-"^10, "-"^10, "-"^10, "-"^10)
    println(header)
    println(sep)

    summary_lines = [header, sep]

    for name in DESIGN_ORDER
        haskey(all_post_means, name) || continue
        pm = all_post_means[name]
        ess_vals = all_ess[name]
        style = get(DESIGN_STYLES, name, (label = name, color = :black))

        δ_μ = pm[1, :] .- true_μ
        δ_K = pm[2, :] .- true_K
        rmse_μ = sqrt(mean(δ_μ .^ 2))
        rmse_K = sqrt(mean(δ_K .^ 2))
        bias_μ = mean(δ_μ)
        bias_K = mean(δ_K)
        med_ess = median(ess_vals)

        line = @sprintf("  %-30s  %10.6f %10.6f %+10.6f %+10.6f %10.1f",
                         style.label, rmse_μ, rmse_K, bias_μ, bias_K, med_ess)
        println(line)
        push!(summary_lines, line)
    end
    flush(stdout)

    # ---- Save ----
    results = Dict{String, Any}(
        "true_mu_max"          => true_μ,
        "true_K_s"             => true_K,
        "true_sigma"           => true_σ,
        "true_Cx0"             => true_Cx0,
        "posterior_means"      => Dict(name => all_post_means[name] for name in keys(all_post_means)),
        "ess"                  => Dict(name => all_ess[name] for name in keys(all_ess)),
        "adaptive_designs"     => all_adaptive_designs,
        "avg_adaptive_design"  => avg_adaptive,
        "n_trials"             => n_trials,
        "N_post"               => N_post,
        "B"                    => B,
        "n_substeps"           => n_substeps,
        "seed"                 => seed,
        "wall_time_s"          => t_total,
    )
    serialize(joinpath(results_dir, "posterior_results.jls"), results)

    p = plot(; xlabel = "Posterior mean mu_max", ylabel = "Posterior mean K_s",
               title = @sprintf("Posterior means (%d trials, N=%d)", n_trials, N_post),
               legend = :outertopright, size = (750, 600))

    for name in DESIGN_ORDER
        haskey(all_post_means, name) || continue
        style = get(DESIGN_STYLES, name, (label = name, color = :black))
        pm = all_post_means[name]

        scatter!(p, pm[1, :], pm[2, :];
                 color = style.color, alpha = 0.4, ms = 3, msw = 0,
                 label = style.label)

        hx, hy = confidence_hull_95(pm[1, :], pm[2, :])
        plot!(p, Shape(hx, hy); color = style.color, fillalpha = 0.1,
              lw = 2, linealpha = 0.8, label = "")
    end

    scatter!(p, [true_μ], [true_K];
             color = :black, shape = :xcross, ms = 10, msw = 3, label = "True value")

    savefig(p, joinpath(results_dir, "plot_posterior.png"))
    println("\nSaved: $(joinpath(results_dir, "plot_posterior.png"))")

    open(joinpath(results_dir, "posterior_summary.txt"), "w") do io
        println(io, "# Posterior Mean Evaluation (GPU)")
        println(io, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "n_trials = $n_trials")
        println(io, "N_post = $N_post")
        println(io, "B = $B")
        println(io, "n_substeps = $n_substeps")
        println(io, "seed = $seed")
        @printf(io, "true_mu_max = %.4f\n", true_μ)
        @printf(io, "true_K_s    = %.4f\n", true_K)
        @printf(io, "true_sigma  = %.4f\n", true_σ)
        @printf(io, "true_Cx0    = %.4f\n", true_Cx0)
        @printf(io, "wall_time_s = %.1f\n", t_total)
        println(io)
        println(io, "avg_adaptive_design = [", join(round.(avg_adaptive; digits=4), ", "), "]")
        for (name, d) in static_designs
            println(io, "$name = [", join(round.(d; digits=4), ", "), "]")
        end
        println(io)
        for line in summary_lines
            println(io, line)
        end
    end

    println("\nDone. Outputs in: $results_dir")
end
