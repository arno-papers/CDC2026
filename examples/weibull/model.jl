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

# ============================================================================
#  Training Budget Allocation
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))

const ODE_BUDGET_TRAJ = 24_000_000
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
    σ_prop = rand(rng, Float32, M, B)
    σ_prop .= SIGMA_PROP_LO .+ (SIGMA_PROP_HI - SIGMA_PROP_LO) .* σ_prop
    σ_add = rand(rng, Float32, M, B)
    σ_add .= SIGMA_ADD_LO .+ (SIGMA_ADD_HI - SIGMA_ADD_LO) .* σ_add
    return σ_prop, σ_add
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

# ============================================================================
#  Targeted sPCE Loss — PK (proportional + additive noise)
# ============================================================================

function targeted_spce_loss_pk(model, ps, st, data)
    θ_full, σ_prop_numer, σ_add_numer, θ_dyn_numer, u0,
        input_buffer, observations, designs, ε_prop, ε_add,
        ll_denom, ll_numer = data

    B = size(ε_prop, 2)

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_dyn_true = θ_full[1:N_PARAMS_DYN, 1:1, :]
    σ_prop_true = θ_full[N_PARAMS_DYN+1, 1, :]
    σ_add_true  = θ_full[N_PARAMS_DYN+2, 1, :]

    u = repeat(u0, 1, 1, B)

    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        d = action
        designs[step, :] .= d[1, :]

        u = integrate(u, θ_dyn_true, d, DT, N_SUBSTEPS)

        obs = u[4, 1, :] ./ V_C
        y_noisy = obs .+ σ_prop_true .* obs .* ε_prop[step, :] .+ σ_add_true .* ε_add[step, :]

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= d[1, :]
    end

    # DENOMINATOR
    n_denom = size(θ_full, 2)
    θ_dyn_denom  = θ_full[1:N_PARAMS_DYN, :, :]
    σ_prop_denom = θ_full[N_PARAMS_DYN+1, :, :]
    σ_add_denom  = θ_full[N_PARAMS_DYN+2, :, :]

    u_denom = repeat(u0, 1, n_denom, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, d_step, DT, N_SUBSTEPS)

        pred = u_denom[4, :, :] ./ V_C
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred
        σ²_total = (σ_prop_denom .* pred) .^ 2 .+ σ_add_denom .^ 2
        ll_denom .-= 0.5f0 .* (residual .^ 2 ./ σ²_total .+ log.(σ²_total))
    end

    # NUMERATOR
    M_N = size(ll_numer, 1)

    u_numer = repeat(u0, 1, M_N, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_numer, d_step, DT, N_SUBSTEPS)

        pred = u_numer[4, :, :] ./ V_C
        actual_obs = observations[step:step, :]
        residual = actual_obs .- pred
        σ²_total = (σ_prop_numer .* pred) .^ 2 .+ σ_add_numer .^ 2
        ll_numer .-= 0.5f0 .* (residual .^ 2 ./ σ²_total .+ log.(σ²_total))
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
#  Batch Preparation — PK
# ============================================================================

function prepare_batch_pk(rng, n_denom, M, B_micro, u0, xdev)
    θ_full_cpu = sample_θ_full(rng, n_denom, B_micro)
    θ_dyn_numer_cpu = sample_θ_dyn_numer(rng, θ_full_cpu[1:N_PARAMS_DYN, 1:1, :], M, B_micro)
    θ_full = θ_full_cpu |> xdev
    θ_dyn_numer = θ_dyn_numer_cpu |> xdev
    σ_prop_numer, σ_add_numer = sample_θ_N_joint(rng, M, B_micro)
    σ_prop_numer = σ_prop_numer |> xdev
    σ_add_numer = σ_add_numer |> xdev

    input_buffer = zeros(Float32, 2, N_STEPS, B_micro) |> xdev
    observations = zeros(Float32, N_STEPS, B_micro) |> xdev
    designs = zeros(Float32, N_STEPS, B_micro) |> xdev
    ε_prop = randn(rng, Float32, N_STEPS, B_micro) |> xdev
    ε_add  = randn(rng, Float32, N_STEPS, B_micro) |> xdev
    ll_denom = zeros(Float32, n_denom, B_micro) |> xdev
    ll_numer = zeros(Float32, M, B_micro) |> xdev

    return (θ_full, σ_prop_numer, σ_add_numer, θ_dyn_numer, u0,
            input_buffer, observations, designs, ε_prop, ε_add,
            ll_denom, ll_numer)
end
