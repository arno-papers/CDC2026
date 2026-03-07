#!/usr/bin/env julia
# GPU-accelerated static sPCE design optimizer.
#
# Usage:
#   julia --project=. examples/monod/optimize_static.jl [n_iters=300] [B=64] [L=15] ...

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))

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
        d_step = design[step:step, :]
        u = integrate(u, θ_T_true, d_step, DT, n_substeps_val)
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
        d_step = design[step:step, :]
        u_denom = integrate(u_denom, θ_T_denom, d_step, DT, n_substeps_val)
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
        d_step = design[step:step, :]
        u_numer = integrate(u_numer, θ_T_true, d_step, DT, n_substeps_val)
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

    loss_per_episode = -(log_numerator .- log_denominator)
    loss = sum(loss_per_episode) / Float32(B)

    return loss, st, (;)
end

# ============================================================================
#  Multi-restart optimization
# ============================================================================

function optimize_static_spce(;
    n_iters::Int = 1000,
    n_restarts::Int = 1,
    B::Int = 64,
    L::Int = L_CONTRASTIVE,
    M::Int = M_NUISANCE,
    n_substeps::Int = N_SUBSTEPS,
    lr_max::Float32 = 0.01f0,
    lr_min::Float32 = 1f-5,
    warmup::Int = 20,
    grad_accum::Int = 1,
    seed::Int = 0,
    results_dir::AbstractString = joinpath(@__DIR__, "results"),
)
    Reactant.set_default_backend("gpu")
    xdev = reactant_device()
    println("Using device: ", xdev)

    model = make_design_model()
    n_denom = L + 1
    B_micro = B ÷ grad_accum

    u0 = make_u0()
    u0_ra = u0 |> xdev

    n_substeps_val = n_substeps

    best_design = fill(5.0f0, N_STEPS)
    best_loss = Inf32
    best_restart = 0
    all_results = []

    for r in 1:n_restarts
        rng = MersenneTwister(seed + r)
        design_init = fill(0.0f0, N_STEPS)

        ps_init, st_init = Lux.setup(rng, model)
        ps_init = (layer_1 = (weight = reshape(copy(design_init), N_STEPS, 1),
                              bias = zeros(Float32, N_STEPS)),)

        ps_ra = ps_init |> xdev
        st_ra = st_init |> xdev

        opt = Adam(lr_min)
        train_state = Lux.Training.TrainState(model, ps_ra, st_ra, opt)

        loss_history = Float32[]
        local_best_loss = Inf32
        local_best_design_cpu = copy(design_init)

        for iter in 1:n_iters
            lr_t = cosine_lr(iter, n_iters, Float64(lr_max), Float64(lr_min), warmup)
            Optimisers.adjust!(train_state.optimizer_state;
                               eta = Float32(lr_t / grad_accum))

            total_loss = 0.0f0
            for _k in 1:grad_accum
                θ_full = sample_θ_full(rng, n_denom, B_micro) |> xdev
                σ_numer, Cx0_numer = sample_θ_N_joint(rng, M, B_micro)
                σ_numer = σ_numer |> xdev
                Cx0_numer = Cx0_numer |> xdev
                ε = randn(rng, Float32, N_STEPS, B_micro) |> xdev
                ll_denom_buf = zeros(Float32, n_denom, B_micro) |> xdev
                ll_numer_buf = zeros(Float32, M, B_micro) |> xdev

                data = (θ_full, σ_numer, Cx0_numer, u0_ra, ε,
                        ll_denom_buf, ll_numer_buf, n_substeps_val)

                _, loss_k, _, train_state = Lux.Training.single_train_step!(
                    AutoEnzyme(), static_spce_loss, data, train_state
                )
                total_loss += loss_k
            end

            design_cpu = Array(train_state.parameters.layer_1.weight)
            clamp!(design_cpu, 0.0f0, 10.0f0)
            copyto!(train_state.parameters.layer_1.weight, design_cpu)

            avg_loss = total_loss / Float32(grad_accum)
            push!(loss_history, avg_loss)

            design_flat = vec(design_cpu)
            if avg_loss < local_best_loss
                local_best_loss = avg_loss
                local_best_design_cpu .= design_flat
            end

            if iter % 25 == 0 || iter == 1 || iter == n_iters
                @printf("[sPCE r%d] iter %3d/%3d | lr=%.6f | loss=%.6f | best=%.6f | design=[%s]\n",
                        r, iter, n_iters, lr_t, avg_loss, local_best_loss,
                        join([@sprintf("%.2f", x) for x in design_flat], ", "))
                flush(stdout)
            end
        end

        @printf("[sPCE] restart %d/%d -> best loss = %.6f\n", r, n_restarts, local_best_loss)
        println("  design = [", join([@sprintf("%.3f", x) for x in local_best_design_cpu], ", "), "]")
        flush(stdout)

        push!(all_results, (;
            restart = r,
            design = copy(local_best_design_cpu),
            loss = local_best_loss,
            loss_history = loss_history,
        ))

        if local_best_loss < best_loss
            best_loss = local_best_loss
            best_design .= local_best_design_cpu
            best_restart = r
        end
    end

    @printf("\n[sPCE] Selected restart %d with loss = %.6f\n", best_restart, best_loss)
    println("Final design: [", join([@sprintf("%.4f", x) for x in best_design], ", "), "]")
    flush(stdout)

    mkpath(results_dir)
    serialize(joinpath(results_dir, "spce_static_design.jls"), Dict(
        "design"       => best_design,
        "loss"         => best_loss,
        "best_restart" => best_restart,
        "all_results"  => [(restart=r.restart, design=r.design, loss=r.loss, loss_history=r.loss_history) for r in all_results],
    ))

    open(joinpath(results_dir, "spce_optimize_summary.txt"), "w") do io
        println(io, "# Static sPCE-optimal design (GPU + Enzyme)")
        println(io, "date = $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "# Configuration")
        println(io, "n_iters = $n_iters")
        println(io, "n_restarts = $n_restarts")
        println(io, "B = $B")
        println(io, "L = $L")
        println(io, "M = $M")
        println(io, "n_substeps = $n_substeps")
        println(io, "grad_accum = $grad_accum")
        println(io, "lr_max = $lr_max")
        println(io, "lr_min = $lr_min")
        println(io, "warmup = $warmup")
        println(io, "seed = $seed")
        println(io)
        println(io, "# Result")
        @printf(io, "best_loss = %.7f\n", best_loss)
        println(io, "best_restart = $best_restart")
        println(io, "design = [", join([@sprintf("%.6f", x) for x in best_design], ", "), "]")
        println(io)
        for r in all_results
            @printf(io, "restart_%d_loss = %.7f\n", r.restart, r.loss)
            println(io, "restart_$(r.restart)_design = [",
                    join([@sprintf("%.4f", x) for x in r.design], ", "), "]")
        end
    end

    p = plot(; xlabel="Iteration", ylabel="sPCE Loss (negated)",
               title="Static sPCE Optimization", legend=:topright)
    for r in all_results
        plot!(p, r.loss_history; label="restart $(r.restart)", lw=1.5)
    end
    savefig(p, joinpath(results_dir, "plot_spce_optimize_loss.png"))
    println("Saved: $(joinpath(results_dir, "plot_spce_optimize_loss.png"))")

    println("\nDone. Outputs in: $results_dir")
    return best_design, best_loss
end

# ============================================================================
#  Main
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    n_iters     = 1000
    n_restarts  = 1
    B           = 64
    L           = L_CONTRASTIVE
    M           = M_NUISANCE
    n_substeps  = N_SUBSTEPS
    lr_max      = 0.01f0
    lr_min      = 1f-5
    warmup      = 20
    grad_accum  = 1
    seed        = 0
    results_dir = joinpath(@__DIR__, "results")

    println("\n=== Static sPCE Design Optimizer (Reactant + Enzyme) ===")
    println("n_iters     = $n_iters")
    println("n_restarts  = $n_restarts")
    println("B           = $B ($(grad_accum)x$(B ÷ grad_accum) micro)")
    println("L           = $L")
    println("M           = $M")
    println("n_substeps  = $n_substeps")
    println("lr          = [$lr_min, $lr_max] cosine, warmup=$warmup")
    println("grad_accum  = $grad_accum")
    println("seed        = $seed")
    println("results_dir = $results_dir")
    println()
    flush(stdout)

    t_start = time()
    optimize_static_spce(;
        n_iters, n_restarts, B, L, M, n_substeps,
        lr_max, lr_min, warmup, grad_accum, seed, results_dir,
    )
    t_total = time() - t_start
    @printf("\nTotal wall time: %.1fs\n", t_total)
end
