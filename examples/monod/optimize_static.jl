#!/usr/bin/env julia
# GPU-accelerated static sPCE design optimizer.
# Initialized from the mean of adaptive policy rollouts.
#
# Usage:
#   julia --project=. examples/monod/optimize_static.jl

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

using Dates
using Plots
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  Trivial Lux model: Dense(1 => N_STEPS) whose weight IS the design vector.
# ============================================================================

function make_design_model()
    return Dense(1 => N_STEPS; use_bias=true)
end

# ============================================================================
#  Static sPCE loss
# ============================================================================

function static_spce_loss(model, ps, st, data)
    θ_full, θ_obs_numer, u0, observations, ε, ll_denom, ll_numer, n_substeps_val = data

    B = size(ε, 3)
    design = ps.layer_1.weight

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_dyn_true = θ_full[1:N_PARAMS_DYN, 1:1, :]
    θ_obs_true_3d = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1:1, :]
    θ_obs_true = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1, :]

    u = make_initial_state(u0, θ_dyn_true, θ_obs_true_3d, B)

    for step in 1:N_STEPS
        d_step = design[step:step, :]
        u = integrate(u, θ_dyn_true, d_step, DT, n_substeps_val)
        y_noisy = observe_noisy(u, θ_obs_true, ε, step)
        observations[step, :] .= y_noisy
    end

    n_denom = size(θ_full, 2)
    θ_dyn_denom = θ_full[1:N_PARAMS_DYN, :, :]
    θ_obs_denom = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, :, :]

    u_denom = make_initial_state(u0, θ_dyn_denom, θ_obs_denom, B)

    for step in 1:N_STEPS
        d_step = design[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_denom, observations[step:step, :], u_denom, θ_obs_denom)
    end

    M_N = size(ll_numer, 1)

    u_numer = make_initial_state(u0, θ_dyn_true, θ_obs_numer, B)

    for step in 1:N_STEPS
        d_step = design[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_true, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_numer, observations[step:step, :], u_numer, θ_obs_numer)
    end

    ll_max_num = maximum(ll_numer; dims=1)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))
    log_numerator = lse_num .- log(Float32(M_N))

    ll_max_den = maximum(ll_denom; dims=1)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))
    log_denominator = lse_den .- log(Float32(n_denom))

    loss_per_episode = -(log_numerator .- log_denominator)
    loss = sum(loss_per_episode) / Float32(B)

    return loss, st, (;)
end

# ============================================================================
#  Training loop (in function scope to avoid soft-scope issues)
# ============================================================================

function optimize_static_spce(design_init, xdev;
    n_iters::Int, L::Int, M::Int, B_micro::Int, n_substeps::Int,
    lr_max::Float32, lr_min::Float32, warmup::Int, grad_accum::Int,
    loss_png_every::Int, seed::Int, results_dir::String)

    model = make_design_model()
    n_denom = L + 1

    u0 = make_u0()
    u0_ra = u0 |> xdev

    rng = MersenneTwister(seed)
    ps_model, st_model = Lux.setup(rng, model)
    ps_model = (layer_1 = (weight = reshape(copy(design_init), N_STEPS, 1),
                           bias = zeros(Float32, N_STEPS)),)

    ps_ra = ps_model |> xdev
    st_ra = st_model |> xdev

    opt = Adam(lr_min)
    train_state = Lux.Training.TrainState(model, ps_ra, st_ra, opt)

    loss_history = Float32[]
    best_loss = Inf32
    best_design_cpu = copy(design_init)

    on_iteration = loss_plot_callback(;
        title="Static sPCE Optimization",
        output_path=joinpath(results_dir, "plot_spce_optimize_loss.png"),
        save_every=loss_png_every, n_iters)

    t_start = time()

    for iter in 1:n_iters
        ga = grad_accum
        lr_t = cosine_lr(iter, n_iters, Float64(lr_max), Float64(lr_min), warmup)
        Optimisers.adjust!(train_state.optimizer_state;
                           eta = Float32(lr_t / ga))

        total_loss = 0.0f0
        for _k in 1:ga
            θ_full = sample_θ_full(rng, n_denom, B_micro) |> xdev
            θ_obs_numer = sample_θ_N_joint(rng, M, B_micro) |> xdev
            ε = randn(rng, Float32, N_NOISE_CHANNELS, N_STEPS, B_micro) |> xdev
            observations = zeros(Float32, N_STEPS, B_micro) |> xdev
            ll_denom_buf = zeros(Float32, n_denom, B_micro) |> xdev
            ll_numer_buf = zeros(Float32, M, B_micro) |> xdev

            data = (θ_full, θ_obs_numer, u0_ra, observations, ε,
                    ll_denom_buf, ll_numer_buf, n_substeps)

            _, loss_k, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(), static_spce_loss, data, train_state
            )
            total_loss += loss_k
        end

        design_cpu = Array(train_state.parameters.layer_1.weight)
        clamp!(design_cpu, 0.0f0, 10.0f0)
        copyto!(train_state.parameters.layer_1.weight, design_cpu)

        avg_loss = total_loss / Float32(ga)
        push!(loss_history, avg_loss)

        design_flat = vec(design_cpu)
        if avg_loss < best_loss
            best_loss = avg_loss
            best_design_cpu .= design_flat
        end

        if on_iteration !== nothing
            on_iteration(iter, avg_loss, loss_history, train_state)
        end

        if iter % 10 == 0 || iter == 1 || iter == n_iters
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\tlr: %.6f\tga: %d\n",
                    iter, n_iters, avg_loss, lr_t, ga)
            flush(stdout)
        end
    end

    t_total = time() - t_start
    return best_design_cpu, best_loss, loss_history, t_total
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    seed        = 0
    n_iters     = 1000
    n_rollouts  = 1000
    ode_budget  = ODE_BUDGET_TRAJ
    n_substeps  = N_SUBSTEPS
    lr_max      = 0.003f0
    lr_min      = 1f-5
    warmup      = 50
    grad_accum  = GRAD_ACCUM_STEPS
    loss_png_every = 10
    results_dir = joinpath(@__DIR__, "results")

    L, M, B_micro = allocate_budget(ode_budget; B_multiplier=grad_accum)
    B_total = B_micro * grad_accum

    Reactant.set_default_backend("gpu")
    xdev = reactant_device()
    println("Using device: ", xdev)

    mkpath(results_dir)

    # ---- Compute initial design from adaptive policy rollouts ----
    println("\n--- Computing initial design from adaptive policy ---")
    flush(stdout)
    ps_cpu, st_cpu, _ = load_checkpoint_cpu(results_dir)
    rng_init = MersenneTwister(42)
    _, st_cpu = Lux.setup(rng_init, policy)

    init = adaptive_mean_design(policy, ps_cpu, st_cpu;
                                 n_rollouts=n_rollouts, seed=seed, n_substeps=n_substeps)
    design_init = Float32.(clamp.(init, ACTION_LO, ACTION_HI))
    println("  Init: [", join([@sprintf("%.3f", x) for x in design_init], ", "), "]")
    flush(stdout)

    println("\n=== Static sPCE Design Optimizer (Reactant + Enzyme) ===")
    println("n_iters     = $n_iters")
    println("B_micro     = $B_micro  (ga ramps from $grad_accum)")
    println("L           = $L")
    println("M           = $M")
    println("n_substeps  = $n_substeps")
    println("lr          = [$lr_min, $lr_max] cosine, warmup=$warmup")
    println("init        = mean of $n_rollouts adaptive rollouts")
    println("seed        = $seed")
    println("results_dir = $results_dir")
    println()
    flush(stdout)

    open(joinpath(results_dir, "spce_optimize_summary.txt"), "w") do io
        println(io, "# Static sPCE-optimal design (GPU + Enzyme)")
        println(io, "date = $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "# Configuration")
        println(io, "n_iters = $n_iters")
        println(io, "B_micro = $B_micro")
        println(io, "L = $L")
        println(io, "M = $M")
        println(io, "n_substeps = $n_substeps")
        println(io, "grad_accum = $grad_accum (ramps)")
        println(io, "lr_max = $lr_max")
        println(io, "lr_min = $lr_min")
        println(io, "warmup = $warmup")
        println(io, "ode_budget = $ode_budget")
        println(io, "seed = $seed")
        println(io, "init = adaptive_mean ($n_rollouts rollouts)")
    end

    best_design_cpu, best_loss, loss_history, t_total = optimize_static_spce(
        design_init, xdev;
        n_iters, L, M, B_micro, n_substeps, lr_max, lr_min, warmup, grad_accum,
        loss_png_every, seed, results_dir)

    @printf("\n[sPCE] best loss = %.7f\n", best_loss)
    println("Final design: [", join([@sprintf("%.4f", x) for x in best_design_cpu], ", "), "]")
    @printf("Total wall time: %.1fs (%.1fs/iter)\n", t_total, t_total / n_iters)
    flush(stdout)

    serialize(joinpath(results_dir, "spce_static_design.jls"), Dict(
        "design"       => best_design_cpu,
        "loss"         => best_loss,
        "loss_history" => loss_history,
    ))

    open(joinpath(results_dir, "spce_optimize_summary.txt"), "a") do io
        println(io)
        println(io, "# Result")
        @printf(io, "best_loss = %.7f\n", best_loss)
        println(io, "design = [", join([@sprintf("%.6f", x) for x in best_design_cpu], ", "), "]")
        @printf(io, "wall_time_s = %.1f\n", t_total)
    end

    println("\nDone. Outputs in: $results_dir")
end
