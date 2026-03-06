# ============================================================================
# Common definitions for DADS experiments.
#
# This file is meant to be `include`d AFTER model.jl. It loads:
#   1. src/common_core.jl (CPU-safe infrastructure)
#   2. Reactant-dependent pieces: integrate(), targeted_spce_loss(), train_policy()
#
# model.jl must define: dynamics(), N_STEPS, DT, N_SUBSTEPS, N_PARAMS_DYN,
# sampling functions (incl. sample_θ_dyn_numer), policy network,
# budget constants (L_CONTRASTIVE, M_NUISANCE, GRAD_BATCH, GRAD_ACCUM_STEPS)
# ============================================================================

include(joinpath(@__DIR__, "common_core.jl"))

using Reactant
using Reactant: @trace

# ============================================================================
#  Reactant-compiled ODE integrator
# ============================================================================

function integrate(u, θ, Q_in, dt, n_substeps)
    dt_sub = dt / n_substeps
    @trace mincut=true for _ in 1:n_substeps
        u = rk4_step(u, θ, Q_in, dt_sub)
    end
    return u
end

# ============================================================================
#  Targeted sPCE Loss — Joint Nuisance Sampling
# ============================================================================

function targeted_spce_loss(model, ps, st, data)
    θ_full, σ_numer, Cx0_numer, θ_dyn_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer = data

    B = size(ε, 2)

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_dyn_true = θ_full[1:N_PARAMS_DYN, 1:1, :]
    σ_true = θ_full[N_PARAMS_DYN+1, 1, :]
    Cx0_true = θ_full[N_PARAMS_DYN+2, 1, :]

    u = vcat(
        repeat(u0[1:1, :, :], 1, 1, B),
        reshape(Cx0_true, 1, 1, B),
        repeat(u0[3:3, :, :], 1, 1, B),
    )

    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        Q_in = action
        designs[step, :] .= Q_in[1, :]

        u = integrate(u, θ_dyn_true, Q_in, DT, N_SUBSTEPS)

        obs = u[1, 1, :]
        noise = ε[step, :]
        y_noisy = obs .+ σ_true .* noise

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= Q_in[1, :]
    end

    # DENOMINATOR
    n_denom = size(θ_full, 2)
    θ_dyn_denom = θ_full[1:N_PARAMS_DYN, :, :]
    σ²_denom = (θ_full[N_PARAMS_DYN+1, :, :]) .^ 2
    Cx0_denom = θ_full[N_PARAMS_DYN+2:N_PARAMS_DYN+2, :, :]

    u_denom = vcat(
        repeat(u0[1:1, :, :], 1, n_denom, B),
        Cx0_denom,
        repeat(u0[3:3, :, :], 1, n_denom, B),
    )

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, Q_step, DT, N_SUBSTEPS)

        pred_obs = u_denom[1, :, :]
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred_obs
        ll_denom .-= 0.5f0 .* (residual.^2 ./ σ²_denom .+ log.(σ²_denom))
    end

    # NUMERATOR
    σ²_numer = σ_numer .^ 2
    M_N = size(ll_numer, 1)

    u_numer = vcat(
        repeat(u0[1:1, :, :], 1, M_N, B),
        reshape(Cx0_numer, 1, M_N, B),
        repeat(u0[3:3, :, :], 1, M_N, B),
    )

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_numer, Q_step, DT, N_SUBSTEPS)

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

    loss_per_episode = -(log_numerator .- log_denominator)
    loss = sum(loss_per_episode) / Float32(B)

    return loss, st, (;)
end

# ============================================================================
#  Training
# ============================================================================

function _prepare_batch_default(rng, n_denom, M, B_micro, u0, xdev)
    θ_full_cpu = sample_θ_full(rng, n_denom, B_micro)
    θ_dyn_numer_cpu = sample_θ_dyn_numer(rng, θ_full_cpu[1:N_PARAMS_DYN, 1:1, :], M, B_micro)
    θ_full = θ_full_cpu |> xdev
    θ_dyn_numer = θ_dyn_numer_cpu |> xdev
    σ_numer, Cx0_numer = sample_θ_N_joint(rng, M, B_micro)
    σ_numer = σ_numer |> xdev
    Cx0_numer = Cx0_numer |> xdev

    input_buffer = zeros(Float32, 2, N_STEPS, B_micro) |> xdev
    observations = zeros(Float32, N_STEPS, B_micro) |> xdev
    designs = zeros(Float32, N_STEPS, B_micro) |> xdev
    ε = randn(rng, Float32, N_STEPS, B_micro) |> xdev
    ll_denom = zeros(Float32, n_denom, B_micro) |> xdev
    ll_numer = zeros(Float32, M, B_micro) |> xdev

    return (θ_full, σ_numer, Cx0_numer, θ_dyn_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)
end

function train_policy(model, ps, st, rng;
    loss_fn = targeted_spce_loss,
    prepare_batch = _prepare_batch_default,
    xdev = identity,
    n_iters = 50,
    on_iteration = nothing,
    lr_max = 0.003f0,
    lr_min = 1f-5,
    warmup = 50,
    grad_accum = GRAD_ACCUM_STEPS,
    grad_batch = GRAD_BATCH,
    L = L_CONTRASTIVE,
    M = M_NUISANCE,
    clip_norm = 1.0f0,
    save_dir = ".",
)
    B_micro = grad_batch ÷ grad_accum
    n_denom = L + 1

    opt = Adam(lr_min)
    train_state = Lux.Training.TrainState(model, ps, st, opt)
    loss_history = Float32[]
    diagnostics = Dict{String, Vector{Float32}}()

    u0 = make_u0()
    u0 = u0 |> xdev

    grads_last = nothing
    for iteration in 1:n_iters
        ga = grad_accum + (iteration - 1) ÷ 10
        lr_t = cosine_lr(iteration, n_iters, lr_max, lr_min, warmup)
        Optimisers.adjust!(train_state.optimizer_state; eta = Float32(lr_t / ga))

        total_loss = 0.0f0

        for _k in 1:ga
            data = prepare_batch(rng, n_denom, M, B_micro, u0, xdev)

            grads_last, loss_k, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(), loss_fn, data, train_state
            )
            total_loss += loss_k
        end

        if iteration % 10 == 0 || iteration == 1
            collect_diagnostics!(diagnostics, train_state, grads_last)
        end

        avg_loss = total_loss / Float32(ga)
        push!(loss_history, avg_loss)

        if on_iteration !== nothing
            on_iteration(iteration, avg_loss, loss_history, train_state)
        end

        if iteration % 10 == 0 || iteration == 1
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\tlr: %.6f\tga: %d\n", iteration, n_iters, avg_loss, lr_t, ga)
        end
    end

    save_results(save_dir, train_state, loss_history, diagnostics)

    return train_state, loss_history, diagnostics
end
