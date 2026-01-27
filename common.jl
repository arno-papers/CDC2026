# ============================================================================
# Common definitions for DADS experiments.
#
# This file is meant to be `include`d by scripts (e.g. `training.jl`,
# `profiler.jl`). It should not execute training as a side-effect.
# ============================================================================

using Lux, Reactant, Random
using Reactant: @trace
using Optimisers
using Printf

# ============================================================================
#  Bioreactor Dynamics
# ============================================================================

function bioreactor_dynamics(u, θ, Q_in)
    # Layout: (state_dim, ...batch_dims)
    # Keep the batch dimensions intact (e.g. (3, B) or (3, n_denom, B)).
    C_s = selectdim(u, 1, 1)
    C_x = selectdim(u, 1, 2)
    V = selectdim(u, 1, 3)
    μ_max = selectdim(θ, 1, 1)
    K_s = selectdim(θ, 1, 2)
    μ = @. μ_max * C_s / (K_s + C_s)
    σ = @. μ / 0.777f0
    du1 = @. -σ * C_x + (Q_in ./ V) * (50.0f0 - C_s)
    du2 = @. μ * C_x - (Q_in ./ V) * C_x
    du3 = @. 0.0f0 * V + Q_in

    du = similar(u)
    selectdim(du, 1, 1) .= du1
    selectdim(du, 1, 2) .= du2
    selectdim(du, 1, 3) .= du3
    return du
end

function rk4_step(u, θ, Q_in, dt)
    k1 = bioreactor_dynamics(u, θ, Q_in)
    k2 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k1, θ, Q_in)
    k3 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k2, θ, Q_in)
    k4 = bioreactor_dynamics(u .+ dt .* k3, θ, Q_in)
    return u .+ (dt / 6.0f0) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
end

function integrate(u, θ, Q_in, dt, n_substeps)
    dt_sub = dt / n_substeps
    @trace for _ in 1:n_substeps
        u = rk4_step(u, θ, Q_in, dt_sub)
    end
    return u
end

function integrate_cpu(u, θ, Q_in, dt, n_substeps)
    dt_sub = dt / n_substeps
    for _ in 1:n_substeps
        u = rk4_step(u, θ, Q_in, dt_sub)
    end
    return u
end

# ============================================================================
#  Positional Encoding
# ============================================================================

function sinusoidal_pe(seq_len::Int)
    position = reshape(Float32.(0:(seq_len - 1)), 1, seq_len)
    div_term = exp.(Float32.(0:2:31) .* -(log(1000.0f0) / 32.0f0))
    angles = div_term * position
    pe = zeros(Float32, 32, seq_len)
    pe[1:2:end, :] .= sin.(angles)
    pe[2:2:end, :] .= cos.(angles[1:16, :])
    return pe
end

# ============================================================================
#  Policy
# ============================================================================

const policy = @compact(
    input_proj = Dense(2 => 32),
    rms1 = RMSNorm((32,)),
    mha = MultiHeadAttention(32; nheads = 4),
    rms2 = RMSNorm((32,)),
    ff = Chain(Dense(32 => 64, gelu), Dense(64 => 32)),
    output_head = Dense(32 => 1),
) do x
    seq_len = size(x, 2)
    x = input_proj(x)
    x = x .+ reshape(sinusoidal_pe(seq_len), 32, seq_len, 1)
    h = rms1(x)
    attn, _ = mha(h)
    x = x + attn
    x = x + ff(rms2(x))
    @return 10.0f0 .* sigmoid.(output_head(x[:, end, :]))
end

# ============================================================================
#  Targeted sPCE Loss with Nuisance Parameters
#
#  θ = (μ_max, K_s, σ) - full parameters (3D)
#       θ_T = (μ_max, K_s) - target parameters
#       θ_N = σ             - nuisance parameter
#
#  Numerator: average over M nuisance samples given θ_{T,0}
#  Denominator: average over L+1 full θ samples
# ============================================================================

const N_STEPS = 10
const DT = 1.0f0            # Total time per control interval
const N_SUBSTEPS = 500      # Integration substeps per control interval
const L_CONTRASTIVE = 256   # Contrastive samples for denominator
const M_NUISANCE = 128      # Nuisance samples for numerator
const GRAD_BATCH = 16       # Gradient minibatch (episodes) per update

# Prior bounds
const μ_max_lo, μ_max_hi = 0.3f0, 0.5f0
const K_s_lo, K_s_hi = 0.3f0, 0.6f0
const σ_lo, σ_hi = 0.05f0, 0.15f0  # Nuisance: measurement noise std

# ASCII aliases (useful for scripts / CLI usage)
const mu_max_lo, mu_max_hi = μ_max_lo, μ_max_hi
const sigma_lo, sigma_hi = σ_lo, σ_hi

function targeted_spce_loss(model, ps, st, data)
    # Batch layout (episode minibatch size B = GRAD_BATCH)
    # θ_full: (3, L+1, B) - full parameter samples [μ_max, K_s, σ]
    # θ_N_numer: (M, B) - nuisance samples for numerator
    θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer = data

    # Ensure accumulators start at zero (keeps allocation device-safe)
    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    # True parameters per episode (first contrastive sample)
    θ_T_true = θ_full[1:2, 1:1, :]   # (2, 1, B)
    σ_true = θ_full[3, 1, :]         # (B,)

    B = GRAD_BATCH
    u = repeat(u0, 1, 1, B)  # (3, 1, B)

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
    # ========================================================================
    n_denom = L_CONTRASTIVE + 1
    θ_T_denom = θ_full[1:2, :, :]                               # (2, n_denom, B)
    σ²_denom = (θ_full[3, :, :]) .^ 2                              # (n_denom, B)

    u_denom = repeat(u0, 1, n_denom, B)                          # (3, n_denom, B)

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]                          # (1, B)
        u_denom = integrate(u_denom, θ_T_denom, Q_step, DT, N_SUBSTEPS)

        pred_obs = u_denom[1, :, :]                              # (n_denom, B)
        actual_obs = observations[step:step, :]                  # (1, B)
        residual = actual_obs .- pred_obs
        ll_denom .-= 0.5f0 .* (residual.^2 ./ σ²_denom .+ log.(σ²_denom))
    end

    # ========================================================================
    # NUMERATOR: (1/M) * Σ_m p(h_K | θ_{T,0}, θ_N^{(m)}, π)
    # All M samples share θ_T_true, only σ differs - simulate once, broadcast likelihood
    # ========================================================================
    σ²_numer = θ_N_numer .^ 2                        # (M, B)
    u_numer = repeat(u0, 1, 1, B)                    # (3, 1, B)

    for step in 1:N_STEPS
        Q_step = designs[step:step, :]                          # (1, B)
        u_numer = integrate(u_numer, θ_T_true, Q_step, DT, N_SUBSTEPS)

        pred_obs = u_numer[1:1, 1, :]                       # (1, B)
        actual_obs = observations[step:step, :]             # (1, B)
        residual² = (actual_obs .- pred_obs) .^ 2    # (1, B)

        ll_numer .-= 0.5f0 .* (residual² ./ σ²_numer .+ log.(σ²_numer))
    end

    # ========================================================================
    # Targeted sPCE loss
    # ========================================================================
    ll_max_num = maximum(ll_numer; dims=1)                                         # (1, B)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))        # (1, B)
    log_numerator = lse_num .- log(Float32(M_NUISANCE))

    ll_max_den = maximum(ll_denom; dims=1)                                         # (1, B)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))        # (1, B)
    log_denominator = lse_den .- log(Float32(n_denom))

    loss_per_episode = -(log_numerator .- log_denominator)                         # (1, B)
    loss = sum(loss_per_episode) / Float32(B)

    return loss, st, (;)
end

# ============================================================================
#  Sampling
# ============================================================================

function sample_θ_full(rng, n_samples)
    # Full parameters: (μ_max, K_s, σ)
    θ = rand(rng, Float32, 3, n_samples)
    θ[1, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :]
    θ[2, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :]
    θ[3, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[3, :]
    return θ
end

function sample_θ_full(rng, n_denom::Int, B::Int)
    θ = rand(rng, Float32, 3, n_denom, B)
    @views begin
        θ[1, :, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :, :]
        θ[2, :, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :, :]
        θ[3, :, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[3, :, :]
    end
    return θ
end

function sample_θ_N(rng, n_samples)
    θ = rand(rng, Float32, n_samples)
    θ .= σ_lo .+ (σ_hi - σ_lo) .* θ
    return θ
end

function sample_θ_N(rng, M::Int, B::Int)
    θ = rand(rng, Float32, M, B)
    θ .= σ_lo .+ (σ_hi - σ_lo) .* θ
    return θ
end

# ============================================================================
#  Training
# ============================================================================

function train_policy(model, ps, st, rng;
    xdev = identity,
    n_iters = 50,
    on_iteration = nothing,
)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
    loss_history = Float32[]

    for iteration in 1:n_iters
        # θ_full: (3, L+1, B) full samples (first in each episode generates data)
        n_denom = L_CONTRASTIVE + 1
        θ_full = sample_θ_full(rng, n_denom, GRAD_BATCH) |> xdev

        # θ_N_numer: (M, B) nuisance samples for numerator
        θ_N_numer = sample_θ_N(rng, M_NUISANCE, GRAD_BATCH) |> xdev

        # Buffers
        u0 = zeros(Float32, 3, 1, 1)
        u0[1, 1, 1] = 3.0f0
        u0[2, 1, 1] = 0.25f0
        u0[3, 1, 1] = 7.0f0
        u0 = u0 |> xdev
        input_buffer = zeros(Float32, 2, N_STEPS, GRAD_BATCH) |> xdev
        observations = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
        designs = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
        ε = randn(rng, Float32, N_STEPS, GRAD_BATCH) |> xdev
        ll_denom = zeros(Float32, n_denom, GRAD_BATCH) |> xdev   # (n_denom, B)
        ll_numer = zeros(Float32, M_NUISANCE, GRAD_BATCH) |> xdev

        data = (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)

        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoEnzyme(), targeted_spce_loss, data, train_state
        )
        push!(loss_history, loss)

        if on_iteration !== nothing
            on_iteration(iteration, loss, loss_history, train_state)
        end

        if iteration % 10 == 0 || iteration == 1
            @printf("Iter: [%4d/%4d]\tTargeted sPCE Loss: %.8f\n", iteration, n_iters, loss)
        end
    end

    return train_state, loss_history
end
