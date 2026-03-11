# ============================================================================
# Weibull PK model — 2-compartment with transit absorption.
#
# State: [A_t1, A_t2, A_t3, A_central, A_peripheral] (amounts in mg)
# Observation: C_c = A_central / V_C (central concentration, mg/L)
# Noise: proportional + additive
# Target params: k_a, k_tr (absorption)
# Nuisance params: CL, Q_d (elimination), sigma_prop, sigma_add (noise)
#
# This file must be included BEFORE src/common_core.jl or src/common.jl.
# ============================================================================

using Lux, Random
using Printf
using Serialization

# ============================================================================
#  2-Compartment PK Dynamics with Transit Absorption
# ============================================================================

const V_C = 10.0f0   # Central volume (L)
const V_P = 30.0f0   # Peripheral volume (L)

function dynamics(u, θ, R_inf)
    A_t1 = selectdim(u, 1, 1)
    A_t2 = selectdim(u, 1, 2)
    A_t3 = selectdim(u, 1, 3)
    A_c  = selectdim(u, 1, 4)
    A_p  = selectdim(u, 1, 5)

    k_a  = selectdim(θ, 1, 1)
    k_tr = selectdim(θ, 1, 2)
    CL   = selectdim(θ, 1, 3)
    Q_d  = selectdim(θ, 1, 4)

    du1 = @. R_inf - k_tr * A_t1
    du2 = @. k_tr * A_t1 - k_tr * A_t2
    du3 = @. k_tr * A_t2 - k_a * A_t3
    du4 = @. k_a * A_t3 - (CL + Q_d) / V_C * A_c + Q_d / V_P * A_p
    du5 = @. Q_d / V_C * A_c - Q_d / V_P * A_p

    du = similar(u)
    selectdim(du, 1, 1) .= du1
    selectdim(du, 1, 2) .= du2
    selectdim(du, 1, 3) .= du3
    selectdim(du, 1, 4) .= du4
    selectdim(du, 1, 5) .= du5
    return du
end

# ============================================================================
#  Experiment Constants
# ============================================================================

const N_STEPS = 24
const DT = 1.0f0
const N_SUBSTEPS = 10

const ACTION_LO = 0.0f0
const ACTION_HI = 10.0f0    # max 10 mg/hr

# ============================================================================
#  Prior Bounds
# ============================================================================

# Target parameters (absorption)
const K_A_LO,  K_A_HI  = 0.5f0, 3.0f0   # hr^-1
const K_TR_LO, K_TR_HI = 0.5f0, 3.0f0   # hr^-1

# Nuisance dynamics
const CL_LO, CL_HI = 1.0f0, 5.0f0       # L/hr
const Q_D_LO, Q_D_HI = 0.5f0, 3.0f0     # L/hr

# Nuisance observation noise
const SIGMA_PROP_LO, SIGMA_PROP_HI = 0.05f0, 0.20f0   # proportional (5-20%)
const SIGMA_ADD_LO, SIGMA_ADD_HI = 0.01f0, 0.10f0      # additive (mg/L)

const N_TARGET = 2       # k_a, k_tr
const N_PARAMS_DYN = 4   # k_a, k_tr, CL, Q_d
const N_PARAMS_OBS = 2   # sigma_prop, sigma_add
const N_NOISE_CHANNELS = 2  # proportional + additive noise

# ============================================================================
#  Training Budget Allocation
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))

const ODE_BUDGET_TRAJ = 12_000_000
const (L_CONTRASTIVE, M_NUISANCE, GRAD_BATCH) = allocate_budget(ODE_BUDGET_TRAJ)
const GRAD_ACCUM_STEPS = 1

# ============================================================================
#  Sampling
# ============================================================================

function sample_θ_full(rng, n_samples::Int)
    θ = rand(rng, Float32, 6, n_samples)
    @views begin
        θ[1, :] .= K_A_LO  .+ (K_A_HI  - K_A_LO)  .* θ[1, :]
        θ[2, :] .= K_TR_LO .+ (K_TR_HI - K_TR_LO) .* θ[2, :]
        θ[3, :] .= CL_LO   .+ (CL_HI   - CL_LO)   .* θ[3, :]
        θ[4, :] .= Q_D_LO  .+ (Q_D_HI  - Q_D_LO)  .* θ[4, :]
        θ[5, :] .= SIGMA_PROP_LO .+ (SIGMA_PROP_HI - SIGMA_PROP_LO) .* θ[5, :]
        θ[6, :] .= SIGMA_ADD_LO  .+ (SIGMA_ADD_HI  - SIGMA_ADD_LO)  .* θ[6, :]
    end
    return θ
end

function sample_θ_full(rng, n_denom::Int, B::Int)
    θ = rand(rng, Float32, 6, n_denom, B)
    @views begin
        θ[1, :, :] .= K_A_LO  .+ (K_A_HI  - K_A_LO)  .* θ[1, :, :]
        θ[2, :, :] .= K_TR_LO .+ (K_TR_HI - K_TR_LO) .* θ[2, :, :]
        θ[3, :, :] .= CL_LO   .+ (CL_HI   - CL_LO)   .* θ[3, :, :]
        θ[4, :, :] .= Q_D_LO  .+ (Q_D_HI  - Q_D_LO)  .* θ[4, :, :]
        θ[5, :, :] .= SIGMA_PROP_LO .+ (SIGMA_PROP_HI - SIGMA_PROP_LO) .* θ[5, :, :]
        θ[6, :, :] .= SIGMA_ADD_LO  .+ (SIGMA_ADD_HI  - SIGMA_ADD_LO)  .* θ[6, :, :]
    end
    return θ
end

function sample_θ_N_joint(rng, M::Int, B::Int)
    θ_obs = rand(rng, Float32, N_PARAMS_OBS, M, B)
    @views begin
        θ_obs[1, :, :] .= SIGMA_PROP_LO .+ (SIGMA_PROP_HI - SIGMA_PROP_LO) .* θ_obs[1, :, :]
        θ_obs[2, :, :] .= SIGMA_ADD_LO .+ (SIGMA_ADD_HI - SIGMA_ADD_LO) .* θ_obs[2, :, :]
    end
    return θ_obs
end

function sample_θ_dyn_numer(rng, θ_dyn_true, M, B)
    # θ_dyn_true: (4, 1, B) — fix k_a (row 1), k_tr (row 2); resample CL, Q_d
    θ_dyn = zeros(Float32, N_PARAMS_DYN, M, B)
    for b in 1:B
        θ_dyn[1, :, b] .= θ_dyn_true[1, 1, b]   # k_a fixed
        θ_dyn[2, :, b] .= θ_dyn_true[2, 1, b]   # k_tr fixed
        θ_dyn[3, :, b] .= CL_LO  .+ (CL_HI  - CL_LO)  .* rand(rng, Float32, M)
        θ_dyn[4, :, b] .= Q_D_LO .+ (Q_D_HI - Q_D_LO) .* rand(rng, Float32, M)
    end
    return θ_dyn
end

# ============================================================================
#  Initial State
# ============================================================================

function make_u0()
    return zeros(Float32, 5, 1, 1)
end

# ============================================================================
#  Observation Model Callbacks
# ============================================================================

function make_initial_state(u0, _θ_dyn, θ_obs, B)
    n_samples = size(θ_obs, 2)
    return repeat(u0, 1, n_samples, B)
end

function observe_noisy(u, θ_obs_true, ε, step)
    obs = u[4, 1, :] ./ V_C
    return obs .+ θ_obs_true[1, :] .* obs .* ε[1, step, :] .+ θ_obs_true[2, :] .* ε[2, step, :]
end

function log_likelihood_step!(ll, y_broadcast, u_pred, θ_obs)
    pred = u_pred[4, :, :] ./ V_C
    σ²_total = (θ_obs[1, :, :] .* pred) .^ 2 .+ θ_obs[2, :, :] .^ 2
    residual = y_broadcast .- pred
    ll .-= 0.5f0 .* (residual .^ 2 ./ σ²_total .+ log.(σ²_total))
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
    output_head = Dense(32 => 1; init_bias=(rng, dims...) -> fill(-2.0f0, dims...)),
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
