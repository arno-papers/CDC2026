# ============================================================================
# Adaptive BIM baseline for the DC motor example.
#
# Myopic (1-step-ahead) strategy: at each step k, compute MAP estimate from
# observations so far, then grid-search for the next voltage that maximizes
# log det of the marginal posterior precision for (k, J).
#
# Requires: model.jl and src/common_core.jl already included.
# ============================================================================

# ============================================================================
#  Variable-length trajectory (ForwardDiff-compatible)
# ============================================================================

function omega_trajectory_diff_n(params::AbstractVector, design::AbstractVector;
                                  n_substeps::Int=N_SUBSTEPS)
    T = promote_type(eltype(params), eltype(design))
    k, _, f_val = params[1], params[2], params[3]
    denom = T(R_CONST) * f_val + k^2
    i_ss = f_val * T(V_PRE) / denom
    ω_ss = k * T(V_PRE) / denom
    u = reshape(T[i_ss, ω_ss, zero(T)], 3, 1)
    θ_mat = reshape(T[params[1], params[2], params[3]], 3, 1)
    n = length(design)
    ω_traj = Vector{T}(undef, n)
    for step in 1:n
        u = integrate_cpu(u, θ_mat, T(design[step]), T(DT), n_substeps)
        ω_traj[step] = u[2, 1]
    end
    return ω_traj
end

# ============================================================================
#  FIM for variable-length design
# ============================================================================

function fim_matrix_n(theta_all, sigma, design::AbstractVector;
                       n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_all[1]), Float64(theta_all[2]), Float64(theta_all[3])]
    J = ForwardDiff.jacobian(
        p -> omega_trajectory_diff_n(p, design; n_substeps=n_substeps),
        params
    )
    T = eltype(J)
    σ2 = T(Float64(sigma))^2
    return (one(T) / σ2) .* (J' * J)
end

# ============================================================================
#  Negative log-posterior (for MAP estimation)
# ============================================================================

function neg_log_posterior(theta_4d::AbstractVector, observations::AbstractVector,
                           designs::AbstractVector, k_steps::Int;
                           n_substeps::Int=N_SUBSTEPS)
    T = eltype(theta_4d)
    k_val, J_val, f_val, σ_val = theta_4d[1], theta_4d[2], theta_4d[3], theta_4d[4]

    # Predicted omega trajectory
    params = T[k_val, J_val, f_val]
    ω_pred = omega_trajectory_diff_n(params, designs[1:k_steps]; n_substeps=n_substeps)

    # Negative log-likelihood
    σ2 = σ_val^2
    nll = zero(T)
    for j in 1:k_steps
        residual = observations[j] - ω_pred[j]
        nll += T(0.5) * (residual^2 / σ2 + log(σ2))
    end
    return nll
end

# ============================================================================
#  MAP estimate via Adam + ForwardDiff
# ============================================================================

function map_estimate(observations::AbstractVector, designs::AbstractVector,
                      k_steps::Int, theta_init::Vector{Float64};
                      n_iters::Int=100, lr::Float64=0.005,
                      n_substeps::Int=N_SUBSTEPS)
    theta = copy(theta_init)
    # Adam state
    m = zeros(Float64, 4)
    v = zeros(Float64, 4)
    β1, β2, ε_adam = 0.9, 0.999, 1e-8

    bounds_lo = Float64[k_lo, J_lo, f_lo, σ_lo]
    bounds_hi = Float64[k_hi, J_hi, f_hi, σ_hi]

    obs_f64 = Float64.(observations)
    des_f64 = Float64.(designs)

    for t in 1:n_iters
        g = ForwardDiff.gradient(
            θ -> neg_log_posterior(θ, obs_f64, des_f64, k_steps; n_substeps=n_substeps),
            theta
        )
        m .= β1 .* m .+ (1 - β1) .* g
        v .= β2 .* v .+ (1 - β2) .* g .^ 2
        m_hat = m ./ (1 - β1^t)
        v_hat = v ./ (1 - β2^t)
        theta .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε_adam)
        @. theta = clamp(theta, bounds_lo, bounds_hi)
    end
    return theta
end

# ============================================================================
#  Adaptive BIM rollout
# ============================================================================

function rollout_adaptive_bim(rng, k_true::Float64, J_true::Float64,
                               f_true::Float64, sigma_true::Float64;
                               n_grid::Int=100, map_iters::Int=100,
                               n_substeps::Int=N_SUBSTEPS)
    # True dynamics parameters for simulation
    theta_dyn_true = reshape(Float32[k_true, J_true, f_true], 3, 1)
    denom_true = Float32(R_CONST) * Float32(f_true) + Float32(k_true)^2
    u = reshape(Float32[Float32(f_true) * Float32(V_PRE) / denom_true,
                         Float32(k_true) * Float32(V_PRE) / denom_true,
                         0.0f0], 3, 1)

    designs = zeros(Float64, N_STEPS)
    observations = zeros(Float64, N_STEPS)
    step_times_s = zeros(Float64, N_STEPS)

    # Prior midpoints for warm-start
    theta_hat = Float64[(k_lo + k_hi) / 2, (J_lo + J_hi) / 2,
                         (f_lo + f_hi) / 2, (σ_lo + σ_hi) / 2]

    design_grid = collect(range(Float64(ACTION_LO), Float64(ACTION_HI); length=n_grid))

    for step in 1:N_STEPS
        t_start = time_ns()

        # 1. MAP estimate (skip at step 1 — no data yet)
        if step > 1
            theta_hat = map_estimate(observations, designs, step - 1, theta_hat;
                                     n_iters=map_iters, n_substeps=n_substeps)
        end

        # 2. Greedy grid search: maximize log det B_T over candidate voltages
        best_d = design_grid[1]
        best_score = -Inf
        for d_cand in design_grid
            trial_design = vcat(designs[1:step-1], d_cand)
            F = fim_matrix_n(theta_hat[1:3], theta_hat[4], trial_design;
                             n_substeps=n_substeps)
            for idx in 1:3
                F[idx, idx] += PRIOR_PREC[idx]
            end
            B_T = schur_complement_2x2(F)
            score = logdet(Symmetric(B_T))
            if score > best_score
                best_score = score
                best_d = d_cand
            end
        end
        designs[step] = best_d

        t_end = time_ns()
        step_times_s[step] = (t_end - t_start) / 1e9

        # 3. Simulate true system and observe
        u = integrate_cpu(u, theta_dyn_true, Float32(best_d), DT, n_substeps)
        y_obs = Float64(u[2, 1]) + sigma_true * randn(rng)
        observations[step] = y_obs
    end

    return (; designs, observations, step_times_s, theta_hat)
end
