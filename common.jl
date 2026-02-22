# ============================================================================
# Common definitions for DADS experiments.
#
# This file is meant to be `include`d by scripts (e.g. `training.jl`,
# `profiler.jl`). It should not execute training as a side-effect.
#
# Includes common_core.jl (CPU-safe code) and adds Reactant-dependent pieces:
#   - integrate() with @trace for XLA compilation
#   - targeted_spce_loss() with joint nuisance sampling
#   - train_policy()
# ============================================================================

include("common_core.jl")

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
#
#  Each of M_NUISANCE samples is a jointly drawn (σ, Cx0) pair.
#  Every sample gets its own ODE trajectory → clean 2D (M_NUISANCE, B)
#  logsumexp for the numerator.
# ============================================================================

function targeted_spce_loss(model, ps, st, data)
    # Batch layout (episode minibatch size B ≤ GRAD_BATCH)
    # θ_full:    (4, L+1, B) - full parameter samples [μ_max, K_s, σ, Cx0]
    # σ_numer:   (M_NUISANCE, B) - jointly sampled σ for numerator
    # Cx0_numer: (M_NUISANCE, B) - jointly sampled Cx0 for numerator
    # ll_numer:  (M_NUISANCE, B) - preallocated log-likelihood accumulator
    θ_full, σ_numer, Cx0_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer = data

    # Infer batch size from data (supports micro-batching)
    B = size(ε, 2)

    # Ensure accumulators start at zero (keeps allocation device-safe)
    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    # True parameters per episode (first contrastive sample)
    θ_T_true = θ_full[1:2, 1:1, :]   # (2, 1, B)
    σ_true = θ_full[3, 1, :]         # (B,)
    Cx0_true = θ_full[4, 1, :]       # (B,)

    # Build initial state with per-episode C_x(0)
    u = vcat(
        repeat(u0[1:1, :, :], 1, 1, B),        # C_s(0) = 3.0
        reshape(Cx0_true, 1, 1, B),             # C_x(0) varies per episode
        repeat(u0[3:3, :, :], 1, 1, B),         # V(0) = 7.0
    )                                            # (3, 1, B)

    # Rollout with true θ (B episodes in parallel)
    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        Q_in = action                              # (1, B)
        designs[step, :] .= Q_in[1, :]

        u = integrate(u, θ_T_true, Q_in, DT, N_SUBSTEPS)

        # Noisy observation: y = C_s + σ_true * ε
        obs = u[1, 1, :]                             # (B,)
        noise = ε[step, :]                           # (B,)
        y_noisy = obs .+ σ_true .* noise             # (B,)

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= Q_in[1, :]
    end

    # ========================================================================
    # DENOMINATOR: (1/(L+1)) * Σ_ℓ p(h_K | θ_ℓ, π)
    # Each contrastive sample gets its own (μ_max, K_s, σ, Cx0)
    # ========================================================================
    n_denom = L_CONTRASTIVE + 1
    θ_T_denom = θ_full[1:2, :, :]                               # (2, n_denom, B)
    σ²_denom = (θ_full[3, :, :]) .^ 2                           # (n_denom, B)
    Cx0_denom = θ_full[4:4, :, :]                               # (1, n_denom, B)

    u_denom = vcat(
        repeat(u0[1:1, :, :], 1, n_denom, B),                   # (1, n_denom, B)
        Cx0_denom,                                                # (1, n_denom, B)
        repeat(u0[3:3, :, :], 1, n_denom, B),                   # (1, n_denom, B)
    )                                                             # (3, n_denom, B)

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]                          # (1, B)
        u_denom = integrate(u_denom, θ_T_denom, Q_step, DT, N_SUBSTEPS)

        pred_obs = u_denom[1, :, :]                              # (n_denom, B)
        actual_obs = observations[step:step, :]                  # (1, B)
        residual = actual_obs .- pred_obs
        ll_denom .-= 0.5f0 .* (residual.^2 ./ σ²_denom .+ log.(σ²_denom))
    end

    # ========================================================================
    # NUMERATOR: (1/M_NUISANCE) * Σ_m p(h_K | θ_{T,0}, σ^(m), Cx0^(m), π)
    # Joint (σ, Cx0) samples → each gets its own ODE trajectory
    # Clean 2D (M_NUISANCE, B) logsumexp
    # ========================================================================
    σ²_numer = σ_numer .^ 2                                      # (M_NUISANCE, B)

    u_numer = vcat(
        repeat(u0[1:1, :, :], 1, M_NUISANCE, B),                 # (1, M_N, B)
        reshape(Cx0_numer, 1, M_NUISANCE, B),                    # (1, M_N, B)
        repeat(u0[3:3, :, :], 1, M_NUISANCE, B),                 # (1, M_N, B)
    )                                                              # (3, M_N, B)

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]                             # (1, B)
        u_numer = integrate(u_numer, θ_T_true, Q_step, DT, N_SUBSTEPS)
        # θ_T_true (2,1,B) broadcasts to (2,M_NUISANCE,B) ✓

        pred_obs = u_numer[1, :, :]                                # (M_NUISANCE, B)
        actual_obs = observations[step:step, :]                    # (1, B)
        residual = actual_obs .- pred_obs
        ll_numer .-= 0.5f0 .* (residual.^2 ./ σ²_numer .+ log.(σ²_numer))
    end

    ll_max_num = maximum(ll_numer; dims=1)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))
    log_numerator = lse_num .- log(Float32(M_NUISANCE))

    # ========================================================================
    # Targeted sPCE loss
    # ========================================================================
    ll_max_den = maximum(ll_denom; dims=1)                                         # (1, B)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))        # (1, B)
    log_denominator = lse_den .- log(Float32(n_denom))

    loss_per_episode = -(log_numerator .- log_denominator)                         # (1, B)
    loss = sum(loss_per_episode) / Float32(B)

    return loss, st, (;)
end

# ============================================================================
#  Training
# ============================================================================

function train_policy(model, ps, st, rng;
    xdev = identity,
    n_iters = 50,
    on_iteration = nothing,
    lr_max = 0.003f0,
    lr_min = 1f-5,
    warmup = 50,
    grad_accum = GRAD_ACCUM_STEPS,
    clip_norm = 1.0f0,
    save_dir = ".",
)
    B_micro = GRAD_BATCH ÷ grad_accum
    n_denom = L_CONTRASTIVE + 1

    # Start at lr_min (warmup will ramp up); adjust! updates lr each iteration
    opt = Adam(lr_min)
    train_state = Lux.Training.TrainState(model, ps, st, opt)
    loss_history = Float32[]
    diagnostics = Dict{String, Vector{Float32}}()

    # Pre-allocate u0 once (same across all micro-batches)
    u0 = zeros(Float32, 3, 1, 1)
    u0[1, 1, 1] = 3.0f0
    u0[2, 1, 1] = 0.25f0
    u0[3, 1, 1] = 7.0f0
    u0 = u0 |> xdev

    grads_last = nothing
    for iteration in 1:n_iters
        # Cosine LR schedule with warmup (scale by 1/K for micro-batch steps)
        lr_t = cosine_lr(iteration, n_iters, lr_max, lr_min, warmup)
        Optimisers.adjust!(train_state.optimizer_state; eta = Float32(lr_t / grad_accum))

        # K fused micro-batch steps (forward + backward + optimizer in one XLA program)
        total_loss = 0.0f0

        for _k in 1:grad_accum
            θ_full = sample_θ_full(rng, n_denom, B_micro) |> xdev
            σ_numer, Cx0_numer = sample_θ_N_joint(rng, M_NUISANCE, B_micro)
            σ_numer = σ_numer |> xdev
            Cx0_numer = Cx0_numer |> xdev

            input_buffer = zeros(Float32, 2, N_STEPS, B_micro) |> xdev
            observations = zeros(Float32, N_STEPS, B_micro) |> xdev
            designs = zeros(Float32, N_STEPS, B_micro) |> xdev
            ε = randn(rng, Float32, N_STEPS, B_micro) |> xdev
            ll_denom = zeros(Float32, n_denom, B_micro) |> xdev
            ll_numer = zeros(Float32, M_NUISANCE, B_micro) |> xdev

            data = (θ_full, σ_numer, Cx0_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)

            grads_last, loss_k, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(), targeted_spce_loss, data, train_state
            )
            total_loss += loss_k
        end

        # Collect diagnostics every 10 iterations (GPU→CPU sync is expensive)
        if iteration % 10 == 0 || iteration == 1
            collect_diagnostics!(diagnostics, train_state, grads_last)
        end

        avg_loss = total_loss / Float32(grad_accum)
        push!(loss_history, avg_loss)

        if on_iteration !== nothing
            on_iteration(iteration, avg_loss, loss_history, train_state)
        end

        if iteration % 10 == 0 || iteration == 1
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\tlr: %.6f\n", iteration, n_iters, avg_loss, lr_t)
        end
    end

    save_results(save_dir, train_state, loss_history, diagnostics)

    return train_state, loss_history, diagnostics
end
