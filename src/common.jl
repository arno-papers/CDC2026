ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") * " --xla_gpu_deterministic_ops=true"

# ============================================================================
# Common definitions for DADS experiments.
#
# This file is meant to be `include`d AFTER model.jl. It loads:
#   1. src/common_core.jl (CPU-safe infrastructure)
#   2. Reactant-dependent pieces: integrate(), targeted_spce_loss(), train_policy()
#
# model.jl must define:
#   Constants: dynamics(), N_STEPS, DT, N_SUBSTEPS, N_PARAMS_DYN,
#     N_PARAMS_OBS, N_NOISE_CHANNELS, policy network,
#     budget constants (L_CONTRASTIVE, M_NUISANCE, GRAD_BATCH, GRAD_ACCUM_STEPS)
#   Sampling: sample_θ_full(), sample_θ_dyn_numer(), sample_θ_N_joint()
#   Observation model: make_initial_state(), observe_noisy(), log_likelihood_step!()
#   Initial state: make_u0()
# ============================================================================

include(joinpath(@__DIR__, "common_core.jl"))

using Lux, Optimisers, Printf
using Reactant
using Reactant: @trace

# ============================================================================
#  Reactant-compiled ODE integrator
# ============================================================================

function integrate(u, θ, d, dt, n_substeps)
    dt_sub = dt / n_substeps
    @trace mincut=true for _ in 1:n_substeps
        u = rk4_step(u, θ, d, dt_sub)
    end
    return u
end

# ============================================================================
#  Targeted sPCE Loss — Joint Nuisance Sampling
# ============================================================================

function targeted_spce_loss(model, ps, st, data)
    θ_full, θ_obs_numer, θ_dyn_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer = data

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

        u = integrate(u, θ_dyn_true, d, DT, N_SUBSTEPS)

        y_noisy = observe_noisy(u, θ_obs_true, ε, step)

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= d[1, :]
    end

    # DENOMINATOR
    n_denom = size(θ_full, 2)
    θ_dyn_denom = θ_full[1:N_PARAMS_DYN, :, :]
    θ_obs_denom = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, :, :]

    u_denom = make_initial_state(u0, θ_dyn_denom, θ_obs_denom, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, d_step, DT, N_SUBSTEPS)
        log_likelihood_step!(ll_denom, observations[step:step, :], u_denom, θ_obs_denom)
    end

    # NUMERATOR
    M_N = size(ll_numer, 1)

    u_numer = make_initial_state(u0, θ_dyn_numer, θ_obs_numer, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_numer, d_step, DT, N_SUBSTEPS)
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
#  Training
# ============================================================================

function _prepare_batch_default(rng, n_denom, M, B_micro, u0, xdev)
    θ_full_cpu = sample_θ_full(rng, n_denom, B_micro)
    θ_dyn_numer_cpu = sample_θ_dyn_numer(rng, θ_full_cpu[1:N_PARAMS_DYN, 1:1, :], M, B_micro)
    θ_full = θ_full_cpu |> xdev
    θ_dyn_numer = θ_dyn_numer_cpu |> xdev
    θ_obs_numer = sample_θ_N_joint(rng, M, B_micro) |> xdev

    input_buffer = zeros(Float32, 2, N_STEPS, B_micro) |> xdev
    observations = zeros(Float32, N_STEPS, B_micro) |> xdev
    designs = zeros(Float32, N_STEPS, B_micro) |> xdev
    ε = randn(rng, Float32, N_NOISE_CHANNELS, N_STEPS, B_micro) |> xdev
    ll_denom = zeros(Float32, n_denom, B_micro) |> xdev
    ll_numer = zeros(Float32, M, B_micro) |> xdev

    return (θ_full, θ_obs_numer, θ_dyn_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)
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
    save_dir = ".",
)
    B_micro = grad_batch ÷ grad_accum
    n_denom = L + 1

    opt = Adam(lr_min)
    train_state = Lux.Training.TrainState(model, ps, st, opt)
    loss_history = Float32[]

    u0 = make_u0()
    u0 = u0 |> xdev

    for iteration in 1:n_iters
        ga = grad_accum
        lr_t = cosine_lr(iteration, n_iters, lr_max, lr_min, warmup)
        Optimisers.adjust!(train_state.optimizer_state; eta = Float32(lr_t / ga))

        total_loss = 0.0f0

        for _k in 1:ga
            data = prepare_batch(rng, n_denom, M, B_micro, u0, xdev)

            _, loss_k, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(), loss_fn, data, train_state
            )
            total_loss += loss_k
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

    save_results(save_dir, train_state, loss_history)

    return train_state, loss_history
end
