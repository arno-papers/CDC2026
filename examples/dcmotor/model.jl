# ============================================================================
# DC motor model — dynamics, constants, sampling, policy, BIM, observation model.
#
# This file must be included BEFORE src/common_core.jl or src/common.jl.
# It defines everything model-specific that the generic infrastructure needs.
#
# DC motor equations:
#   L · di/dt = V - R·i - k·ω      (electrical)
#   J · dω/dt = k·i - f·ω          (mechanical)
#
# States: u = [i, ω, 0] (current, angular velocity, dummy)
#   Note: a 3rd dummy state is required to work around a Reactant tracing
#   issue where selectdim .= fails for 2-element first dimensions inside
#   @trace for loops.
# Target params: k (motor constant), J (inertia)
# Nuisance dynamics: f (friction)
# Nuisance observation: σ (measurement noise)
# Observation: y = ω + σ·ε (angular velocity with noise)
# Design: V(t) ∈ [0, 10] (voltage)
# ============================================================================

using Lux, Random
using ForwardDiff
using LinearAlgebra
using Printf
using Serialization

# ============================================================================
#  DC Motor Dynamics
# ============================================================================

# Known electrical constants
const R_CONST = 0.5f0       # Resistance (Ω)
const L_CONST = 4.5f-3      # Inductance (H)

function dynamics(u, θ, d)
    i_curr = selectdim(u, 1, 1)
    ω = selectdim(u, 1, 2)
    k = selectdim(θ, 1, 1)
    J_m = selectdim(θ, 1, 2)
    f = selectdim(θ, 1, 3)

    di = @. (d - R_CONST * i_curr - k * ω) / L_CONST
    dω = @. (k * i_curr - f * ω) / J_m

    du = similar(u)
    selectdim(du, 1, 1) .= di
    selectdim(du, 1, 2) .= dω
    selectdim(du, 1, 3) .= 0.0f0   # dummy state
    return du
end

# ============================================================================
#  Experiment Constants
# ============================================================================

const N_STEPS = 10
const DT = 0.01f0              # 10ms per step → 100ms total, captures transient (τ_slow ≈ 10–216ms)
const N_SUBSTEPS = 10           # dt_sub = 1ms, RK4 stable (|λ·dt| ≈ 0.1 << 2.8), Float32-accurate

const ACTION_LO = 0.0f0
const ACTION_HI = 10.0f0

# ============================================================================
#  Prior Bounds
# ============================================================================

const k_lo, k_hi = 0.3f0, 0.7f0          # Motor constant
const J_lo, J_hi = 0.01f0, 0.04f0        # Moment of inertia
const f_lo, f_hi = 0.005f0, 0.02f0       # Friction coefficient (nuisance, enters dynamics)
const σ_lo, σ_hi = 0.5f0, 2.0f0           # Measurement noise std (nuisance, observation only)

# ASCII aliases
const sigma_lo, sigma_hi = σ_lo, σ_hi

const N_TARGET = 2       # k, J
const N_PARAMS_DYN = 3   # k, J, f (f is nuisance but enters dynamics)
const N_PARAMS_OBS = 1   # σ
const N_NOISE_CHANNELS = 1

# ============================================================================
#  Training Budget Allocation
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))

const ODE_BUDGET_TRAJ = 2_000_000
const (L_CONTRASTIVE, M_NUISANCE, GRAD_BATCH) = allocate_budget(ODE_BUDGET_TRAJ)
const GRAD_ACCUM_STEPS = 16

# ============================================================================
#  Sampling
# ============================================================================

# θ_full layout: [k, J, f, σ] — 4 params
# Indices 1:3 = dynamics (k, J, f), index 4 = observation (σ)

function sample_θ_full(rng, n_samples)
    θ = rand(rng, Float32, 4, n_samples)
    θ[1, :] .= k_lo .+ (k_hi - k_lo) .* θ[1, :]
    θ[2, :] .= J_lo .+ (J_hi - J_lo) .* θ[2, :]
    θ[3, :] .= f_lo .+ (f_hi - f_lo) .* θ[3, :]
    θ[4, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[4, :]
    return θ
end

function sample_θ_full(rng, n_denom::Int, B::Int)
    θ = rand(rng, Float32, 4, n_denom, B)
    @views begin
        θ[1, :, :] .= k_lo .+ (k_hi - k_lo) .* θ[1, :, :]
        θ[2, :, :] .= J_lo .+ (J_hi - J_lo) .* θ[2, :, :]
        θ[3, :, :] .= f_lo .+ (f_hi - f_lo) .* θ[3, :, :]
        θ[4, :, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[4, :, :]
    end
    return θ
end

function sample_θ_dyn_numer(rng, θ_dyn_true, M, B)
    # θ_dyn_true: (3, 1, B) = [k, J, f]
    # Fix k, J (target); resample f (nuisance)
    θ_dyn = zeros(Float32, N_PARAMS_DYN, M, B)
    for b in 1:B
        θ_dyn[1, :, b] .= θ_dyn_true[1, 1, b]   # k fixed
        θ_dyn[2, :, b] .= θ_dyn_true[2, 1, b]   # J fixed
        θ_dyn[3, :, b] .= f_lo .+ (f_hi - f_lo) .* rand(rng, Float32, M)  # f resampled
    end
    return θ_dyn
end

function sample_θ_N_joint(rng, M::Int, B::Int)
    θ_obs = rand(rng, Float32, N_PARAMS_OBS, M, B)
    @views θ_obs[1, :, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ_obs[1, :, :]
    return θ_obs
end

function draw_prior_samples(rng, n::Int)
    θ = sample_θ_full(rng, n)
    samples = Vector{Tuple{Vector{Float32}, Float32}}(undef, n)
    @inbounds for i in 1:n
        # (dynamics_params=[k, J, f], sigma)
        samples[i] = (Float32[θ[1, i], θ[2, i], θ[3, i]], Float32(θ[4, i]))
    end
    return samples
end

function sample_f(rng, n::Int)
    return Float32[f_lo + (f_hi - f_lo) * rand(rng, Float32) for _ in 1:n]
end

# ============================================================================
#  Initial State
# ============================================================================

# Pre-experiment voltage: motor is running at steady state before experiment begins.
const V_PRE = 5.0f0

function make_u0()
    # Placeholder — actual initial state is computed in make_initial_state from θ_dyn
    return zeros(Float32, 3, 1, 1)
end

# ============================================================================
#  Observation Model Callbacks
# ============================================================================

function make_initial_state(u0, θ_dyn, θ_obs, B)
    # Steady-state at V_PRE: depends on k and f (J drops out at equilibrium)
    #   ω_ss = k·V / (R·f + k²)
    #   i_ss = f·V / (R·f + k²)
    n_samples = size(θ_dyn, 2)
    k = θ_dyn[1:1, :, :]    # (1, n_samples, B)
    f = θ_dyn[3:3, :, :]    # (1, n_samples, B)
    denom = R_CONST .* f .+ k .^ 2
    i_ss = f .* V_PRE ./ denom
    ω_ss = k .* V_PRE ./ denom
    dummy = zero(ω_ss)
    return cat(i_ss, ω_ss, dummy; dims=1)   # (3, n_samples, B)
end

function observe_noisy(u, θ_obs_true, ε, step)
    # Observe angular velocity ω (state index 2) with additive Gaussian noise
    obs = u[2, 1, :]
    σ_true = θ_obs_true[1, :]
    return obs .+ σ_true .* ε[1, step, :]
end

function log_likelihood_step!(ll, y_broadcast, u_pred, θ_obs)
    pred_obs = u_pred[2, :, :]        # ω prediction
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
    output_head = Dense(32 => 1; init_bias=(rng, dims...) -> fill(0.0f0, dims...)),
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
#  ForwardDiff-compatible trajectory (for BIM)
# ============================================================================

function omega_trajectory_diff(params::AbstractVector, design::AbstractVector;
                                n_substeps::Int=N_SUBSTEPS)
    T = promote_type(eltype(params), eltype(design))
    k, _, f_val = params[1], params[2], params[3]
    denom = T(R_CONST) * f_val + k^2
    i_ss = f_val * T(V_PRE) / denom
    ω_ss = k * T(V_PRE) / denom
    u = reshape(T[i_ss, ω_ss, zero(T)], 3, 1)
    θ_mat = reshape(T[params[1], params[2], params[3]], 3, 1)   # [k, J, f]
    ω_traj = Vector{T}(undef, N_STEPS)
    for step in 1:N_STEPS
        u = integrate_cpu(u, θ_mat, T(design[step]), T(DT), n_substeps)
        ω_traj[step] = u[2, 1]
    end
    return ω_traj
end

# ============================================================================
#  Fisher Information Matrix (3×3, for [k, J, f])
# ============================================================================

# Prior precision for uniform priors: 12 / width²
const PRIOR_PREC = Float64[
    12.0 / (Float64(k_hi) - Float64(k_lo))^2,
    12.0 / (Float64(J_hi) - Float64(J_lo))^2,
    12.0 / (Float64(f_hi) - Float64(f_lo))^2,
]

function fim_matrix(theta_all, sigma, design::AbstractVector;
                     n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_all[1]), Float64(theta_all[2]), Float64(theta_all[3])]
    J = ForwardDiff.jacobian(
        p -> omega_trajectory_diff(p, design; n_substeps=n_substeps),
        params
    )
    T = eltype(J)
    σ2 = T(Float64(sigma))^2
    return (one(T) / σ2) .* (J' * J)
end

function schur_complement_2x2(B::AbstractMatrix)
    B_TT = B[1:2, 1:2]
    B_TN = B[1:2, 3:3]
    B_NN = B[3, 3]
    return B_TT - B_TN * B_TN' / B_NN
end

# ============================================================================
#  Bayesian D-optimal objectives
# ============================================================================

function bim_logdet(design::AbstractVector, prior_samples;
                     n_substeps::Int=N_SUBSTEPS)
    T = promote_type(Float64, eltype(design))
    score = zero(T)
    for (theta_all, sigma) in prior_samples
        F = fim_matrix(theta_all, sigma, design; n_substeps=n_substeps)
        for k in 1:3
            F[k, k] += T(PRIOR_PREC[k])
        end
        score += logdet(Symmetric(schur_complement_2x2(F)))
    end
    return score / length(prior_samples)
end

# ============================================================================
#  Design optimization helpers
# ============================================================================

function init_design(restart::Int)
    designs = [
        fill(0.1, N_STEPS),
        vcat(fill(0.1, N_STEPS - 3), [4.0, 7.0, 10.0]),
        fill(5.0, N_STEPS),
        collect(range(10.0, 0.0; length=N_STEPS)),
    ]
    return restart <= length(designs) ? copy(designs[restart]) : 10.0 .* rand(N_STEPS)
end

function optimize_design_grad(objective;
        n_iters::Int=300, lr_max::Float64=1.0, lr_min::Float64=0.01,
        n_restarts::Int=4, results_dir::Union{String,Nothing}=nothing,
        prefix::String="bim")

    best_design = zeros(Float64, N_STEPS)
    best_score = -Inf
    best_restart = 0
    all_histories = Vector{Vector{Float64}}()

    for r in 1:n_restarts
        design = init_design(r)
        velocity = zeros(Float64, N_STEPS)
        local_best = copy(design)
        local_best_score = -Inf
        score_history = Float64[]

        for iter in 1:n_iters
            g = ForwardDiff.gradient(objective, design)
            lr = cosine_lr(iter, n_iters, lr_max, lr_min, 10)
            velocity .= 0.5 .* velocity .+ g
            design .+= lr .* velocity
            clamp!(design, 0.0, 10.0)

            score = objective(design)
            push!(score_history, score)
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

        push!(all_histories, score_history)
        @printf("[GRAD] restart %d/%d -> best = %.5f\n", r, n_restarts, local_best_score)
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

function optimize_static_design_grad(prior_samples;
        n_iters::Int=300, lr_max::Float64=1.0, lr_min::Float64=0.01,
        n_restarts::Int=4, n_substeps::Int=N_SUBSTEPS,
        results_dir::Union{String,Nothing}=nothing, prefix::String="bim")
    objective = ξ -> bim_logdet(ξ, prior_samples; n_substeps=n_substeps)
    return optimize_design_grad(objective; n_iters, lr_max, lr_min, n_restarts,
                                results_dir, prefix)
end

# ============================================================================
#  Adaptive policy rollout (CPU, Float32)
# ============================================================================

function rollout_adaptive_design_cpu(model, ps_cpu, st_cpu, rng,
        theta_T::Vector{Float32}, sigma::Float32, f_val::Float32;
        n_substeps::Int=N_SUBSTEPS)
    theta_dyn = reshape(Float32[theta_T[1], theta_T[2], f_val], 3, 1)
    k = theta_T[1]; f = f_val
    d = Float32(R_CONST) * f + k^2
    u = reshape(Float32[f * Float32(V_PRE) / d, k * Float32(V_PRE) / d, 0.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float32, N_STEPS)
    st_local = st_cpu
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        v_in = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = v_in
        u = integrate_cpu(u, theta_dyn, v_in, DT, n_substeps)
        y_obs = u[2, 1] + sigma * randn(rng, Float32)   # observe ω
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = v_in
    end
    return designs
end
