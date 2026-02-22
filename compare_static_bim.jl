include("common_core.jl")

using Dates
using ForwardDiff
using LinearAlgebra
using Plots
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  Argument parsing
# ============================================================================

function parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        if startswith(a, prefix)
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_bool(args, key; default=false)
    v = parse_kwarg(args, key; default=nothing)
    v === nothing && return default
    return lowercase(v) in ("1", "true", "t", "yes", "y")
end

function parse_int(args, key; default::Int)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Int, v)
end

function parse_float64(args, key; default::Float64)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Float64, v)
end

# ============================================================================
#  Prior sampling
# ============================================================================

function draw_prior_samples(rng, n::Int)
    θ = sample_θ_full(rng, n)
    samples = Vector{Tuple{Vector{Float32}, Float32, Float32}}(undef, n)
    @inbounds for i in 1:n
        samples[i] = (Float32[θ[1, i], θ[2, i]], Float32(θ[3, i]), Float32(θ[4, i]))
    end
    return samples
end

# ============================================================================
#  ForwardDiff-compatible trajectory
#
#  params = [μ_max, K_s, Cx0] — may carry Dual{InnerTag} from inner ForwardDiff
#  design = [Q1, ..., Q14]    — may carry Dual{OuterTag} from optimizer gradient
#
#  The inner and outer ForwardDiff calls use different closure types, so their
#  tags are automatically distinct → no perturbation confusion.
# ============================================================================

function substrate_trajectory_diff(params::AbstractVector, design::AbstractVector;
                                    n_substeps::Int=N_SUBSTEPS)
    T = promote_type(eltype(params), eltype(design))
    u = zeros(T, 3, 1)
    u[1, 1] = T(3.0)
    u[2, 1] = params[3]    # Cx0
    u[3, 1] = T(7.0)
    θ_mat = reshape(T[params[1], params[2]], 2, 1)
    c_s = Vector{T}(undef, N_STEPS)
    for step in 1:N_STEPS
        u = integrate_cpu(u, θ_mat, T(design[step]), T(DT), n_substeps)
        c_s[step] = u[1, 1]
    end
    return c_s
end

# ============================================================================
#  Fisher Information Matrix (3×3, for [μ_max, K_s, Cx0])
# ============================================================================

# Prior precision for uniform priors: 12 / width²
const PRIOR_PREC = Float64[
    12.0 / (Float64(μ_max_hi) - Float64(μ_max_lo))^2,   # μ_max
    12.0 / (Float64(K_s_hi) - Float64(K_s_lo))^2,        # K_s
    12.0 / (Float64(Cx0_hi) - Float64(Cx0_lo))^2,        # Cx0
]

function fim_matrix(theta_T, sigma, Cx0, design::AbstractVector;
                     n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    J = ForwardDiff.jacobian(
        p -> substrate_trajectory_diff(p, design; n_substeps=n_substeps),
        params
    )  # N_STEPS × 3; elements carry outer Dual type if design is Dual
    T = eltype(J)
    σ2 = T(Float64(sigma))^2
    return (one(T) / σ2) .* (J' * J)   # 3×3
end

function schur_complement_2x2(B::AbstractMatrix)
    B_TT = B[1:2, 1:2]
    B_TN = B[1:2, 3:3]     # 2×1
    B_NN = B[3, 3]
    return B_TT - B_TN * B_TN' / B_NN
end

# ============================================================================
#  Bayesian Information Matrix objective (differentiable w.r.t. design)
#
#  BIM = (1/N) Σ_i F(θ_i, ξ) + Σ^{-1}_prior
#  Marginalise Cx0 via Schur complement → 2×2 for θ_T = (μ_max, K_s)
#  Objective: logdet(BIM_marginal)
# ============================================================================

function bim_logdet(design::AbstractVector, prior_samples;
                     n_substeps::Int=N_SUBSTEPS)
    T = promote_type(Float64, eltype(design))
    B_full = zeros(T, 3, 3)
    for (theta_T, sigma, Cx0) in prior_samples
        B_full .+= fim_matrix(theta_T, sigma, Cx0, design; n_substeps=n_substeps)
    end
    B_full ./= length(prior_samples)
    B_full[1, 1] += T(PRIOR_PREC[1])
    B_full[2, 2] += T(PRIOR_PREC[2])
    B_full[3, 3] += T(PRIOR_PREC[3])
    B_marg = schur_complement_2x2(B_full)
    return logdet(Symmetric(B_marg))
end

# ============================================================================
#  Projected gradient ascent with box constraints [0, 10]
# ============================================================================

function init_design(restart::Int)
    designs = [
        fill(0.1, N_STEPS),                                        # near-zero
        vcat(fill(0.1, N_STEPS - 3), [4.0, 7.0, 10.0]),           # late burst
        fill(5.0, N_STEPS),                                        # midpoint
        collect(range(10.0, 0.0; length=N_STEPS)),                 # decreasing ramp
    ]
    return restart <= length(designs) ? copy(designs[restart]) : 10.0 .* rand(N_STEPS)
end

function optimize_static_design_grad(prior_samples;
        n_iters::Int=300, lr_max::Float64=1.0, lr_min::Float64=0.01,
        n_restarts::Int=4, n_substeps::Int=N_SUBSTEPS)

    best_design = zeros(Float64, N_STEPS)
    best_score = -Inf
    best_restart = 0

    for r in 1:n_restarts
        design = init_design(r)
        velocity = zeros(Float64, N_STEPS)
        local_best = copy(design)
        local_best_score = -Inf

        for iter in 1:n_iters
            g = ForwardDiff.gradient(
                ξ -> bim_logdet(ξ, prior_samples; n_substeps=n_substeps),
                design
            )
            lr = cosine_lr(iter, n_iters, lr_max, lr_min, 10)
            velocity .= 0.5 .* velocity .+ g
            design .+= lr .* velocity
            clamp!(design, 0.0, 10.0)

            # Evaluate every iteration (cheap relative to gradient)
            score = bim_logdet(design, prior_samples; n_substeps=n_substeps)
            if score > local_best_score
                local_best_score = score
                local_best .= design
            end

            if iter % 25 == 0 || iter == 1 || iter == n_iters
                @printf("[GRAD r%d] iter %3d/%3d | lr=%.4f | score=%.5f | best=%.5f\n",
                        r, iter, n_iters, lr, score, local_best_score)
                flush(stdout)
            end
        end

        @printf("[GRAD] restart %d/%d → best = %.5f\n", r, n_restarts, local_best_score)
        flush(stdout)
        if local_best_score > best_score
            best_score = local_best_score
            best_design .= local_best
            best_restart = r
        end
    end

    @printf("[GRAD] selected restart %d with score = %.5f\n", best_restart, best_score)
    flush(stdout)
    return Float32.(best_design), best_score
end

# ============================================================================
#  Adaptive policy rollout (CPU, Float32)
# ============================================================================

function rollout_adaptive_design_cpu(model, ps_cpu, st_cpu, rng,
        theta_T::Vector{Float32}, sigma::Float32;
        Cx0::Float32=0.25f0, n_substeps::Int=N_SUBSTEPS)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float32, N_STEPS)
    st_local = st_cpu
    theta_mat = reshape(theta_T, 2, 1)
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        q_in = clamp(Float32(action[1]), 0.0f0, 10.0f0)
        designs[step] = q_in
        u = integrate_cpu(u, theta_mat, q_in, DT, n_substeps)
        y_obs = u[1, 1] + sigma * randn(rng, Float32)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = q_in
    end
    return designs
end

# ============================================================================
#  Unified evaluation: rollout adaptive, score both, collect everything
# ============================================================================

function evaluate_adaptive_vs_static(model, ps_cpu, st_cpu, rng, static_design;
        n_trials::Int=200, n_substeps::Int=N_SUBSTEPS, ridge::Float64=1e-6,
        resample_true_each_trial::Bool=true)

    adaptive_scores  = Vector{Float64}(undef, n_trials)
    static_scores    = Vector{Float64}(undef, n_trials)
    adaptive_designs = Matrix{Float32}(undef, N_STEPS, n_trials)

    θ0 = sample_θ_full(rng, 1)
    fixed_θT  = Float32[θ0[1, 1], θ0[2, 1]]
    fixed_σ   = Float32(θ0[3, 1])
    fixed_Cx0 = Float32(θ0[4, 1])

    I2 = Matrix{Float64}(I, 2, 2)

    for i in 1:n_trials
        θT, σ, Cx0 = if resample_true_each_trial
            tf = sample_θ_full(rng, 1)
            (Float32[tf[1, 1], tf[2, 1]], Float32(tf[3, 1]), Float32(tf[4, 1]))
        else
            (copy(fixed_θT), fixed_σ, fixed_Cx0)
        end

        d_adapt = rollout_adaptive_design_cpu(model, ps_cpu, st_cpu, rng, θT, σ;
                                               Cx0=Cx0, n_substeps=n_substeps)
        adaptive_designs[:, i] .= d_adapt

        # 3×3 FIM → Schur complement → logdet(2×2 + ridge·I)
        F_a = fim_matrix(θT, σ, Cx0, d_adapt; n_substeps=n_substeps)
        F_s = fim_matrix(θT, σ, Cx0, static_design; n_substeps=n_substeps)
        adaptive_scores[i] = logdet(Symmetric(schur_complement_2x2(F_a) + ridge * I2))
        static_scores[i]   = logdet(Symmetric(schur_complement_2x2(F_s) + ridge * I2))

        if i % 50 == 0
            @printf("  trial %d/%d\n", i, n_trials)
            flush(stdout)
        end
    end

    delta = adaptive_scores .- static_scores
    wins  = delta .> 0.0
    return (; adaptive_scores, static_scores, delta, wins, adaptive_designs)
end

# ============================================================================
#  Combined figure: design rollouts (colored by win/loss) + score scatter
# ============================================================================

function plot_combined_comparison(adaptive_designs, static_design,
        adaptive_scores, static_scores, wins, out_png;
        n_show::Int=50)
    steps = collect(1:N_STEPS)
    n_trials = length(wins)
    n_show = clamp(n_show, 1, n_trials)
    win_rate = mean(wins)
    delta_mean = mean(adaptive_scores .- static_scores)

    # --- Top: design rollouts ---
    p1 = plot(; xlabel="step", ylabel="Q_in (L/h)",
              title="Design rollouts", legend=:topleft, ylims=(-0.5, 10.5))
    n_blue = 0; n_orange = 0
    for j in 1:n_show
        if wins[j]
            plot!(p1, steps, adaptive_designs[:, j];
                  color=:dodgerblue, alpha=0.25, lw=1,
                  label=(n_blue == 0 ? "adaptive (win)" : ""))
            n_blue += 1
        else
            plot!(p1, steps, adaptive_designs[:, j];
                  color=:darkorange, alpha=0.25, lw=1,
                  label=(n_orange == 0 ? "adaptive (loss)" : ""))
            n_orange += 1
        end
    end
    plot!(p1, steps, static_design; color=:crimson, lw=3, label="static BIM-optimal")

    # --- Bottom: score scatter ---
    lo = min(minimum(adaptive_scores), minimum(static_scores)) - 0.5
    hi = max(maximum(adaptive_scores), maximum(static_scores)) + 0.5
    p2 = plot(; xlabel="static logdet(F_marg + rI)", ylabel="adaptive logdet(F_marg + rI)",
              title=@sprintf("Win rate %.1f%% | mean Δ = %.3f", 100 * win_rate, delta_mean),
              legend=:bottomright, xlims=(lo, hi), ylims=(lo, hi), aspect_ratio=:equal)
    plot!(p2, [lo, hi], [lo, hi]; color=:black, lw=1, ls=:dash, label="")
    wins_idx = findall(wins)
    loss_idx = findall(.!wins)
    if !isempty(wins_idx)
        scatter!(p2, static_scores[wins_idx], adaptive_scores[wins_idx];
                 color=:dodgerblue, alpha=0.5, ms=4, msw=0, label="adaptive wins")
    end
    if !isempty(loss_idx)
        scatter!(p2, static_scores[loss_idx], adaptive_scores[loss_idx];
                 color=:darkorange, alpha=0.5, ms=4, msw=0, label="static wins")
    end

    p = plot(p1, p2; layout=(2, 1), size=(950, 800))
    savefig(p, out_png)
    println("Saved: $out_png")
end

# ============================================================================
#  Checkpoint loading
# ============================================================================

function load_checkpoint_cpu(path::AbstractString)
    if isdir(path)
        r = load_results(path)
        return r.parameters, r.states, path
    end
    @assert isfile(path) "Checkpoint not found: $path"
    ckpt = deserialize(path)
    return ckpt["parameters"], ckpt["states"], dirname(path)
end

# ============================================================================
#  Summary text file
# ============================================================================

function save_summary_txt(path, cfg, eval_res, static_score, static_design)
    open(path, "w") do io
        println(io, "# Static BIM vs Adaptive Policy (ForwardDiff + gradient)")
        println(io, "date = $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "# Configuration")
        for k in sort(collect(keys(cfg)))
            println(io, "$k = $(cfg[k])")
        end
        println(io)
        println(io, "# BIM definition")
        println(io, "FIM: 3×3 for [μ_max, K_s, Cx0], inner Jacobian via ForwardDiff")
        println(io, "BIM = avg(FIM) + diag(prior_precision)")
        println(io, "Marginal: Schur complement → 2×2 for θ_T = (μ_max, K_s)")
        @printf(io, "prior_prec = [%.3f, %.3f, %.3f]  (12/width² for uniform)\n", PRIOR_PREC...)
        println(io)
        println(io, "# Static design")
        @printf(io, "static_bim_logdet = %.7f\n", static_score)
        println(io, "static_design = [", join(round.(static_design; digits=4), ", "), "]")
        println(io)
        println(io, "# Comparison (per-trial logdet of marginal FIM + ridge)")
        @printf(io, "adaptive_mean = %.7f\n", mean(eval_res.adaptive_scores))
        @printf(io, "adaptive_std  = %.7f\n", std(eval_res.adaptive_scores))
        @printf(io, "static_mean   = %.7f\n", mean(eval_res.static_scores))
        @printf(io, "static_std    = %.7f\n", std(eval_res.static_scores))
        @printf(io, "delta_mean    = %.7f\n", mean(eval_res.delta))
        @printf(io, "delta_std     = %.7f\n", std(eval_res.delta))
        @printf(io, "delta_median  = %.7f\n", median(eval_res.delta))
        @printf(io, "win_rate      = %.4f\n", mean(eval_res.wins))
    end
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    checkpoint    = parse_kwarg(ARGS, "checkpoint"; default="checkpoint.jls")
    output_dir_arg = parse_kwarg(ARGS, "output_dir"; default=nothing)
    seed          = parse_int(ARGS, "seed"; default=0)
    n_prior_opt   = parse_int(ARGS, "n_prior_opt"; default=512)
    n_prior_report = parse_int(ARGS, "n_prior_report"; default=1024)
    n_trials      = parse_int(ARGS, "n_trials"; default=200)
    n_substeps    = parse_int(ARGS, "n_substeps"; default=N_SUBSTEPS)
    ridge         = parse_float64(ARGS, "ridge"; default=1e-6)
    grad_iters    = parse_int(ARGS, "grad_iters"; default=300)
    grad_restarts = parse_int(ARGS, "grad_restarts"; default=4)
    lr_max        = parse_float64(ARGS, "lr_max"; default=0.1)
    lr_min        = parse_float64(ARGS, "lr_min"; default=0.001)
    resample      = parse_bool(ARGS, "resample_true_each_trial"; default=true)
    save_plot     = parse_bool(ARGS, "save_plot"; default=true)
    n_plot_show   = parse_int(ARGS, "n_plot_runouts"; default=50)

    ps_cpu, st_cpu, ckpt_base = load_checkpoint_cpu(checkpoint)
    output_dir = output_dir_arg === nothing ? joinpath(ckpt_base, "bim_comparison") : output_dir_arg
    mkpath(output_dir)

    println("\n=== Bayesian D-optimal Static Design (ForwardDiff + gradient) ===")
    println("checkpoint        = $checkpoint")
    println("output_dir        = $output_dir")
    println("seed              = $seed")
    println("n_prior_opt       = $n_prior_opt")
    println("n_prior_report    = $n_prior_report")
    println("n_trials          = $n_trials")
    println("grad              = $grad_iters iters × $grad_restarts restarts")
    println("lr                = [$lr_min, $lr_max] cosine")
    println("ridge             = $ridge")
    println("resample_each     = $resample")
    println("BIM: 3×3 FIM + prior, Schur complement → 2×2")
    @printf("Prior precision: μ_max=%.1f  K_s=%.1f  Cx0=%.1f\n", PRIOR_PREC...)
    flush(stdout)

    # ---- Phase 1: optimize static design ----
    println("\n--- Phase 1: optimize static design ---")
    flush(stdout)
    rng_opt = MersenneTwister(seed)
    prior_opt = draw_prior_samples(rng_opt, n_prior_opt)

    static_design, static_obj = optimize_static_design_grad(
        prior_opt; n_iters=grad_iters, lr_max=lr_max, lr_min=lr_min,
        n_restarts=grad_restarts, n_substeps=n_substeps)

    rng_report = MersenneTwister(seed + 1)
    prior_report = draw_prior_samples(rng_report, n_prior_report)
    static_obj_report = bim_logdet(Float64.(static_design), prior_report; n_substeps=n_substeps)

    println("\nStatic design optimized.")
    @printf("  Train BIM logdet:  %.5f\n", static_obj)
    @printf("  Report BIM logdet: %.5f\n", static_obj_report)
    println("  Design: [", join(round.(static_design; digits=3), ", "), "]")
    flush(stdout)

    # ---- Phase 2: evaluate adaptive vs static ----
    println("\n--- Phase 2: evaluate adaptive vs static ($n_trials trials) ---")
    flush(stdout)
    rng_eval = MersenneTwister(seed + 2)

    eval_res = evaluate_adaptive_vs_static(
        policy, ps_cpu, st_cpu, rng_eval, static_design;
        n_trials=n_trials, n_substeps=n_substeps, ridge=ridge,
        resample_true_each_trial=resample)

    @printf("\n  Adaptive: %.5f ± %.5f\n", mean(eval_res.adaptive_scores), std(eval_res.adaptive_scores))
    @printf("  Static:   %.5f ± %.5f\n", mean(eval_res.static_scores), std(eval_res.static_scores))
    @printf("  Delta:    %.5f ± %.5f\n", mean(eval_res.delta), std(eval_res.delta))
    @printf("  Win rate: %.1f%%\n", 100 * mean(eval_res.wins))
    flush(stdout)

    # ---- Save ----
    serialize(joinpath(output_dir, "bim_static_design.jls"), Dict(
        "static_design"     => static_design,
        "static_obj_opt"    => static_obj,
        "static_obj_report" => static_obj_report,
    ))
    serialize(joinpath(output_dir, "bim_comparison.jls"), Dict(
        "adaptive_scores"  => eval_res.adaptive_scores,
        "static_scores"    => eval_res.static_scores,
        "delta"            => eval_res.delta,
        "wins"             => eval_res.wins,
        "adaptive_designs" => eval_res.adaptive_designs,
        "static_design"    => static_design,
        "n_trials"         => n_trials,
        "ridge"            => ridge,
    ))

    cfg = Dict{String, Any}(
        "checkpoint" => checkpoint, "seed" => seed,
        "n_prior_opt" => n_prior_opt, "n_prior_report" => n_prior_report,
        "n_trials" => n_trials, "n_substeps" => n_substeps, "ridge" => ridge,
        "grad_iters" => grad_iters, "grad_restarts" => grad_restarts,
        "lr_max" => lr_max, "lr_min" => lr_min,
        "resample_true_each_trial" => resample,
    )
    save_summary_txt(joinpath(output_dir, "bim_summary.txt"), cfg, eval_res,
                     static_obj_report, static_design)

    if save_plot
        plot_combined_comparison(
            eval_res.adaptive_designs, static_design,
            eval_res.adaptive_scores, eval_res.static_scores, eval_res.wins,
            joinpath(output_dir, "plot_bim_comparison.png");
            n_show=n_plot_show)
    end

    println("\nDone. Outputs in: $output_dir")
end
