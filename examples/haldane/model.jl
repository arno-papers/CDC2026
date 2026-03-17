# ============================================================================
# Haldane bioreactor model — dynamics, constants, sampling, policy.
#
# Haldane kinetics add substrate inhibition via α = 1/K_i.
# This file must be included BEFORE src/common_core.jl or src/common.jl.
# It defines everything model-specific that the generic infrastructure needs.
# ============================================================================

using Lux, Random
using Printf
using Serialization

# ============================================================================
#  Haldane Bioreactor Dynamics
# ============================================================================

function dynamics(u, θ, Q_in)
    C_s = selectdim(u, 1, 1)
    C_x = selectdim(u, 1, 2)
    V = selectdim(u, 1, 3)
    μ_max = selectdim(θ, 1, 1)
    K_s = selectdim(θ, 1, 2)
    α = selectdim(θ, 1, 3)
    μ = @. μ_max * C_s / (K_s + C_s + α * C_s^2)
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

# ============================================================================
#  Experiment Constants
# ============================================================================

const N_STEPS = 14
const DT = 1.0f0
const N_SUBSTEPS = 50

const ACTION_LO = 0.0f0
const ACTION_HI = 10.0f0

# ============================================================================
#  Prior Bounds
# ============================================================================

const μ_max_lo, μ_max_hi = 0.39f0, 0.41f0    # Nuisance: max growth rate (tight)
const K_s_lo, K_s_hi = 0.44f0, 0.46f0        # Nuisance: Monod constant (tight)
const σ_lo, σ_hi = 0.05f0, 0.15f0             # Nuisance: measurement noise std
const Cx0_lo, Cx0_hi = 0.22f0, 0.28f0         # Nuisance: initial biomass (tight)

# ASCII aliases
const mu_max_lo, mu_max_hi = μ_max_lo, μ_max_hi
const sigma_lo, sigma_hi = σ_lo, σ_hi

const N_TARGET = 1
const N_PARAMS_DYN = 3
const N_PARAMS_OBS = 2      # σ, Cx0
const N_NOISE_CHANNELS = 1  # single additive Gaussian noise

# Uniform prior on α = 1/K_i (substrate inhibition strength)
const α_lo, α_hi = 0.0f0, 0.15f0

# ============================================================================
#  Training Budget Allocation
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))

const ODE_BUDGET_TRAJ = 6_365_184
const GRAD_ACCUM_STEPS = 8
const (L_CONTRASTIVE, M_NUISANCE, _B_MICRO) = allocate_budget(ODE_BUDGET_TRAJ; B_multiplier=GRAD_ACCUM_STEPS)
const GRAD_BATCH = _B_MICRO * GRAD_ACCUM_STEPS

# ============================================================================
#  Sampling
# ============================================================================

function sample_θ_full(rng, n_samples)
    θ = rand(rng, Float32, 5, n_samples)
    θ[1, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :]
    θ[2, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :]
    θ[3, :] .= α_lo .+ (α_hi - α_lo) .* θ[3, :]
    θ[4, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[4, :]
    θ[5, :] .= Cx0_lo .+ (Cx0_hi - Cx0_lo) .* θ[5, :]
    return θ
end

function sample_θ_full(rng, n_denom::Int, B::Int)
    θ = rand(rng, Float32, 5, n_denom, B)
    @views begin
        θ[1, :, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :, :]
        θ[2, :, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :, :]
        θ[3, :, :] .= α_lo .+ (α_hi - α_lo) .* θ[3, :, :]
        θ[4, :, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[4, :, :]
        θ[5, :, :] .= Cx0_lo .+ (Cx0_hi - Cx0_lo) .* θ[5, :, :]
    end
    return θ
end

function sample_θ_N_joint(rng, M::Int, B::Int)
    θ_obs = rand(rng, Float32, N_PARAMS_OBS, M, B)
    @views begin
        θ_obs[1, :, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ_obs[1, :, :]
        θ_obs[2, :, :] .= Cx0_lo .+ (Cx0_hi - Cx0_lo) .* θ_obs[2, :, :]
    end
    return θ_obs
end

function sample_θ_dyn_numer(rng, θ_dyn_true, M, B)
    # θ_dyn_true: (3, 1, B) from θ_full[1:3, 1:1, :]
    # Fix α (row 3), resample μ_max (row 1) and K_s (row 2)
    θ_dyn = zeros(Float32, N_PARAMS_DYN, M, B)
    for b in 1:B
        θ_dyn[1, :, b] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* rand(rng, Float32, M)
        θ_dyn[2, :, b] .= K_s_lo .+ (K_s_hi - K_s_lo) .* rand(rng, Float32, M)
        θ_dyn[3, :, b] .= θ_dyn_true[3, 1, b]  # α fixed from true sample
    end
    return θ_dyn
end

function draw_prior_samples(rng, n::Int)
    θ = sample_θ_full(rng, n)
    samples = Vector{Tuple{Vector{Float32}, Float32, Float32}}(undef, n)
    @inbounds for i in 1:n
        samples[i] = (Float32[θ[3, i]], Float32(θ[4, i]), Float32(θ[5, i]))
    end
    return samples
end

# ============================================================================
#  Initial State
# ============================================================================

function make_u0()
    u0 = zeros(Float32, 3, 1, 1)
    u0[1, 1, 1] = 3.0f0
    u0[2, 1, 1] = 0.25f0
    u0[3, 1, 1] = 7.0f0
    return u0
end

# ============================================================================
#  Observation Model Callbacks
# ============================================================================

function make_initial_state(u0, _θ_dyn, θ_obs, B)
    n_samples = size(θ_obs, 2)
    return vcat(
        repeat(u0[1:1, :, :], 1, n_samples, B),
        θ_obs[2:2, :, :],
        repeat(u0[3:3, :, :], 1, n_samples, B),
    )
end

function observe_noisy(u, θ_obs_true, ε, step)
    obs = u[1, 1, :]
    return obs .+ θ_obs_true[1, :] .* ε[1, step, :]
end

function log_likelihood_step!(ll, y_broadcast, u_pred, θ_obs)
    pred_obs = u_pred[1, :, :]
    σ² = θ_obs[1, :, :] .^ 2
    residual = y_broadcast .- pred_obs
    ll .-= 0.5f0 .* (residual .^ 2 ./ σ² .+ log.(σ²))
    return nothing
end

# ============================================================================
#  Policy Network
# ============================================================================

const policy = @compact(
    input_proj = Dense(2 => 32),
    rms1 = RMSNorm((32,)),
    mha = MultiHeadAttention(32; nheads = 4),
    rms2 = RMSNorm((32,)),
    ff = Chain(Dense(32 => 64, gelu), Dense(64 => 32)),
    output_head = Dense(32 => 1; init_bias=(rng, dims...) -> fill(-1.0f0, dims...)),
) do x
    seq_len = size(x, 2)
    x = input_proj(x)
    x = x .+ reshape(sinusoidal_pe(seq_len), 32, seq_len, 1)
    h = rms1(x)
    attn, _ = mha(h)
    x = x + attn
    x = x + ff(rms2(x))
    @return ACTION_HI .* sigmoid.(output_head(x[:, end, :]))
end

# ============================================================================
#  Adaptive policy rollout (CPU, Float32)
# ============================================================================

function rollout_adaptive_design_cpu(model, ps_cpu, st_cpu, rng,
        theta_dyn::Vector{Float32}, sigma::Float32;
        Cx0::Float32=0.25f0, n_substeps::Int=N_SUBSTEPS)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float32, N_STEPS)
    st_local = st_cpu
    theta_mat = reshape(theta_dyn, N_PARAMS_DYN, 1)
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = d
        u = integrate_cpu(u, theta_mat, d, DT, n_substeps)
        y_obs = u[1, 1] + sigma * randn(rng, Float32)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end
    return designs
end
