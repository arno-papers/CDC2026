# ============================================================================
# Monod bioreactor model — dynamics, constants, sampling, policy, BIM.
#
# This file must be included BEFORE src/common_core.jl or src/common.jl.
# It defines everything model-specific that the generic infrastructure needs.
# ============================================================================

using Lux, Random
using ForwardDiff
using LinearAlgebra
using Optimisers
using Printf
using Serialization

# ============================================================================
#  Monod Bioreactor Dynamics
# ============================================================================

function dynamics(u, θ, Q_in)
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

const μ_max_lo, μ_max_hi = 0.3f0, 0.5f0
const K_s_lo, K_s_hi = 0.3f0, 0.6f0
const σ_lo, σ_hi = 0.05f0, 0.15f0       # Nuisance: measurement noise std
const Cx0_lo, Cx0_hi = 0.10f0, 0.50f0    # Nuisance: initial biomass

# ASCII aliases
const mu_max_lo, mu_max_hi = μ_max_lo, μ_max_hi
const sigma_lo, sigma_hi = σ_lo, σ_hi

const N_TARGET = 2
const N_PARAMS_DYN = N_TARGET
const N_PARAMS_OBS = 2      # σ, Cx0
const N_NOISE_CHANNELS = 1  # single additive Gaussian noise

# ============================================================================
#  Training Budget Allocation
# ============================================================================

include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))

const ODE_BUDGET_TRAJ = 1250000
const (L_CONTRASTIVE, M_NUISANCE, GRAD_BATCH) = allocate_budget(ODE_BUDGET_TRAJ)
const GRAD_ACCUM_STEPS = 1

# ============================================================================
#  Sampling
# ============================================================================

function sample_θ_full(rng, n_samples)
    θ = rand(rng, Float32, 4, n_samples)
    θ[1, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :]
    θ[2, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :]
    θ[3, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[3, :]
    θ[4, :] .= Cx0_lo .+ (Cx0_hi - Cx0_lo) .* θ[4, :]
    return θ
end

function sample_θ_full(rng, n_denom::Int, B::Int)
    θ = rand(rng, Float32, 4, n_denom, B)
    @views begin
        θ[1, :, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :, :]
        θ[2, :, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :, :]
        θ[3, :, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[3, :, :]
        θ[4, :, :] .= Cx0_lo .+ (Cx0_hi - Cx0_lo) .* θ[4, :, :]
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
    return repeat(θ_dyn_true, 1, M, 1)  # all dynamics params are target → just repeat
end

function draw_prior_samples(rng, n::Int)
    θ = sample_θ_full(rng, n)
    samples = Vector{Tuple{Vector{Float32}, Float32, Float32}}(undef, n)
    @inbounds for i in 1:n
        samples[i] = (Float32[θ[1, i], θ[2, i]], Float32(θ[3, i]), Float32(θ[4, i]))
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

function make_initial_state(u0, θ_obs, B)
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
    output_head = Dense(32 => 1; init_bias=(rng, dims...) -> fill(-4.0f0, dims...)),
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
#  ForwardDiff-compatible trajectory
# ============================================================================

function substrate_trajectory_diff(params::AbstractVector, design::AbstractVector;
                                    n_substeps::Int=N_SUBSTEPS)
    T = promote_type(eltype(params), eltype(design))
    u = zeros(T, 3, 1)
    u[1, 1] = T(3.0)
    u[2, 1] = params[3]    # Cx0
    u[3, 1] = T(7.0)
    θ_mat = reshape(T[params[1], params[2]], 2, 1)
    c_s = Vector{T}(undef, N_STEPS)
    for step in 1:N_STEPS
        u = integrate_cpu(u, θ_mat, T(design[step]), T(DT), n_substeps)
        c_s[step] = u[1, 1]
    end
    return c_s
end

# ============================================================================
#  Fisher Information Matrix (3x3, for [mu_max, K_s, Cx0])
# ============================================================================

# Prior precision for uniform priors: 12 / width^2
const PRIOR_PREC = Float64[
    12.0 / (Float64(μ_max_hi) - Float64(μ_max_lo))^2,
    12.0 / (Float64(K_s_hi) - Float64(K_s_lo))^2,
    12.0 / (Float64(Cx0_hi) - Float64(Cx0_lo))^2,
]

function fim_matrix(theta_T, sigma, Cx0, design::AbstractVector;
                     n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    J = ForwardDiff.jacobian(
        p -> substrate_trajectory_diff(p, design; n_substeps=n_substeps),
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
    for (theta_T, sigma, Cx0) in prior_samples
        F = fim_matrix(theta_T, sigma, Cx0, design; n_substeps=n_substeps)
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

function optimize_design_grad(objective;
        n_iters::Int=250, lr_max::Float64=0.003, lr_min::Float64=1e-5,
        warmup::Int=50, results_dir::Union{String,Nothing}=nothing,
        prefix::String="bim")

    design = fill(0.0, N_STEPS)
    opt_state = Optimisers.setup(Adam(lr_min), design)

    best_design = copy(design)
    best_score = -Inf
    score_history = Float64[]

    for iter in 1:n_iters
        g = ForwardDiff.gradient(objective, design)
        lr = cosine_lr(iter, n_iters, lr_max, lr_min, warmup)
        Optimisers.adjust!(opt_state; eta = lr)
        # Adam minimizes, but we maximize: negate gradient
        opt_state, design = Optimisers.update!(opt_state, design, -g)
        clamp!(design, 0.0, 10.0)

        score = objective(design)
        push!(score_history, score)
        if score > best_score
            best_score = score
            best_design .= design
        end

        if iter % 25 == 0 || iter == 1 || iter == n_iters
            @printf("[GRAD] iter %3d/%3d | lr=%.6f | score=%.5f | best=%.5f\n",
                    iter, n_iters, lr, score, best_score)
            flush(stdout)

            if results_dir !== nothing && isdefined(Main, :Plots)
                p = Plots.plot(; xlabel="Iteration", ylabel="BIM logdet",
                                 title="BIM Optimization ($prefix)", legend=:bottomright)
                Plots.plot!(p, score_history; label="score", lw=1.5)
                Plots.savefig(p, joinpath(results_dir, "plot_$(prefix)_loss.png"))
            end
        end
    end

    @printf("[GRAD] best score = %.5f\n", best_score)
    flush(stdout)
    return Float32.(best_design), best_score
end

function optimize_static_design_grad(prior_samples;
        n_iters::Int=250, lr_max::Float64=0.003, lr_min::Float64=1e-5,
        warmup::Int=50, n_substeps::Int=N_SUBSTEPS,
        results_dir::Union{String,Nothing}=nothing, prefix::String="bim")
    objective = ξ -> bim_logdet(ξ, prior_samples; n_substeps=n_substeps)
    return optimize_design_grad(objective; n_iters, lr_max, lr_min, warmup,
                                results_dir, prefix)
end

# ============================================================================
#  Adaptive policy rollout (CPU, Float32)
# ============================================================================

function rollout_adaptive_design_cpu(model, ps_cpu, st_cpu, rng,
        theta_T::Vector{Float32}, sigma::Float32;
        Cx0::Float32=0.25f0, n_substeps::Int=N_SUBSTEPS)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float32, N_STEPS)
    st_local = st_cpu
    theta_mat = reshape(theta_T, 2, 1)
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
