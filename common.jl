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
using Serialization

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
    @trace mincut=true for _ in 1:n_substeps
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
#  Policy Network
# ============================================================================
#
# A causal transformer that maps observation-action history to control actions.
#
# Architecture overview:
#   1. Input projection: Lifts (observation, action) pairs to embedding space
#   2. Positional encoding: Adds temporal structure via sinusoidal embeddings
#   3. Pre-norm transformer block: Self-attention + feed-forward with residuals
#   4. Output head: Projects final token embedding to scalar action
#
# Input:  x ∈ ℝ^(2 × T × B)  where T = sequence length, B = batch size
#         - x[1, t, :] = observation at step t (e.g., substrate concentration)
#         - x[2, t, :] = action taken at step t (e.g., flow rate Q_in)
#
# Output: action ∈ ℝ^(1 × B), scaled to [0, 10] via sigmoid
# ============================================================================

const policy = @compact(
    # -------------------------------------------------------------------------
    # Layer 1: Input Projection
    # Maps each (obs, action) pair to a 32-dim embedding
    # Input:  (2, T, B)  →  Output: (32, T, B)
    # Conceptually: Lifts raw features into a richer representation space
    # -------------------------------------------------------------------------
    input_proj = Dense(2 => 32),

    # -------------------------------------------------------------------------
    # Layer 2: RMSNorm (Pre-Attention)
    # Normalizes embeddings before attention (Pre-LN transformer style)
    # Input:  (32, T, B)  →  Output: (32, T, B)
    # Conceptually: Stabilizes activations, improves gradient flow
    # -------------------------------------------------------------------------
    rms1 = RMSNorm((32,)),

    # -------------------------------------------------------------------------
    # Layer 3: Multi-Head Self-Attention
    # 4 heads, each with 8-dim queries/keys/values (32/4 = 8)
    # Input:  (32, T, B)  →  Output: (32, T, B)
    # Conceptually: Allows each timestep to attend to all previous timesteps,
    #               learning temporal dependencies in the observation history
    # -------------------------------------------------------------------------
    mha = MultiHeadAttention(32; nheads = 4),

    # -------------------------------------------------------------------------
    # Layer 4: RMSNorm (Pre-FFN)
    # Input:  (32, T, B)  →  Output: (32, T, B)
    # Conceptually: Normalizes before feed-forward, same as rms1
    # -------------------------------------------------------------------------
    rms2 = RMSNorm((32,)),

    # -------------------------------------------------------------------------
    # Layer 5: Feed-Forward Network (MLP)
    # Two-layer MLP with GELU activation and 2x expansion
    # Input:  (32, T, B)  →  hidden: (64, T, B)  →  Output: (32, T, B)
    # Conceptually: Adds non-linear transformations, increases model capacity
    # -------------------------------------------------------------------------
    ff = Chain(Dense(32 => 64, gelu), Dense(64 => 32)),

    # -------------------------------------------------------------------------
    # Layer 6: Output Head
    # Projects the final token's embedding to a scalar action
    # Input:  (32, B)  →  Output: (1, B)
    # Conceptually: Decodes the accumulated history into a control decision
    # -------------------------------------------------------------------------
    output_head = Dense(32 => 1),
) do x
    # x: (2, T, B) - input sequence of (observation, action) pairs

    seq_len = size(x, 2)  # T = number of timesteps in history

    # Step 1: Project inputs to embedding space
    # (2, T, B) → (32, T, B)
    x = input_proj(x)

    # Step 2: Add sinusoidal positional encoding
    # PE: (32, T) broadcast to (32, T, B)
    # Injects temporal order information since attention is permutation-invariant
    x = x .+ reshape(sinusoidal_pe(seq_len), 32, seq_len, 1)

    # Step 3: Pre-norm self-attention with residual connection
    # h = RMSNorm(x): (32, T, B)
    # attn = MHA(h): (32, T, B)
    # x = x + attn: residual connection
    h = rms1(x)
    attn, _ = mha(h)
    x = x + attn

    # Step 4: Pre-norm feed-forward with residual connection
    # x = x + FFN(RMSNorm(x)): (32, T, B)
    x = x + ff(rms2(x))

    # Step 5: Extract final timestep and project to action
    # x[:, end, :]: (32, B) - embedding of the last (most recent) timestep
    # output_head: (32, B) → (1, B)
    # sigmoid scales to (0, 1), then multiply by 10 → action ∈ (0, 10)
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

const N_STEPS = 14
const DT = 1.0f0            # Total time per control interval
const N_SUBSTEPS = 500      # Integration substeps per control interval

# -----------------------------------------------------------------------------
# Training budget allocation (see Appendix, Section "Computational Budget Trade-offs")
#
# Budget: C_traj = B * (L + 3) trajectory rollouts per optimizer update.
# Optimal (L, B) minimizes MSE proxy: 1/B + λ/(L+1)² subject to budget constraint.
# This yields the scaling B* ∝ (L*+1)².
# -----------------------------------------------------------------------------
const ODE_BUDGET_TRAJ = 1060864

const (L_CONTRASTIVE, GRAD_BATCH) = let
    C = ODE_BUDGET_TRAJ
    λ = 1.0  # equal weight on variance (1/B) vs squared bias (1/(L+1)²)
    best_L, best_B, best_obj = 1, fld(C, 4), Inf
    for L in 1:(C - 3)
        B = fld(C, L + 3)
        B < 1 && break
        obj = 1.0/B + λ/(L+1)^2
        if obj < best_obj
            best_obj, best_L, best_B = obj, L, B
        end
    end
    (best_L, best_B)
end

# Gradient accumulation: split B into micro-batches to fit in GPU memory.
# Each micro-batch processes B/K episodes; K optimizer steps per iteration
# with lr scaled by 1/K approximate one step on the full batch.
const GRAD_ACCUM_STEPS = 10
const GRAD_BATCH_MICRO = GRAD_BATCH ÷ GRAD_ACCUM_STEPS

# Nuisance samples: ODE-free (only affect observation model), so can be large
const M_NUISANCE = min(4096, max(512, 32 * (L_CONTRASTIVE + 1)))

# Prior bounds
const μ_max_lo, μ_max_hi = 0.3f0, 0.5f0
const K_s_lo, K_s_hi = 0.3f0, 0.6f0
const σ_lo, σ_hi = 0.05f0, 0.15f0  # Nuisance: measurement noise std

# ASCII aliases (useful for scripts / CLI usage)
const mu_max_lo, mu_max_hi = μ_max_lo, μ_max_hi
const sigma_lo, sigma_hi = σ_lo, σ_hi

function targeted_spce_loss(model, ps, st, data)
    # Batch layout (episode minibatch size B ≤ GRAD_BATCH)
    # θ_full: (3, L+1, B) - full parameter samples [μ_max, K_s, σ]
    # θ_N_numer: (M, B) - nuisance samples for numerator
    θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer = data

    # Infer batch size from data (supports micro-batching)
    B = size(ε, 2)

    # Ensure accumulators start at zero (keeps allocation device-safe)
    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    # True parameters per episode (first contrastive sample)
    θ_T_true = θ_full[1:2, 1:1, :]   # (2, 1, B)
    σ_true = θ_full[3, 1, :]         # (B,)

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
#  Training Diagnostics
# ============================================================================

function _push_diag!(diag::Dict, key::String, val::Float32)
    v = get!(diag, key, Float32[])
    push!(v, val)
end

function _collect_array_stats!(diag::Dict, path::String, x_cpu::AbstractArray{<:Real})
    nrm = Float32(sqrt(sum(abs2, x_cpu)))
    mx  = Float32(maximum(abs, x_cpu))
    has_nan = Float32(any(isnan, x_cpu) || any(isinf, x_cpu))
    _push_diag!(diag, path * ".norm", nrm)
    _push_diag!(diag, path * ".max_abs", mx)
    _push_diag!(diag, path * ".has_nan", has_nan)
end

function _collect_norm_only!(diag::Dict, path::String, x_cpu::AbstractArray{<:Real})
    nrm = Float32(sqrt(sum(abs2, x_cpu)))
    _push_diag!(diag, path * ".norm", nrm)
end

# Extract Adam (mt, vt) from optimizer state, handling both bare Adam and OptimiserChain
function _collect_adam_moments!(diag::Dict, prefix::String, state::Tuple)
    # Bare Adam state: (mt::AbstractArray, vt::AbstractArray, βt)
    if length(state) >= 2 && state[1] isa AbstractArray && state[2] isa AbstractArray
        _collect_norm_only!(diag, prefix * ".mt", Array(state[1]))
        _collect_norm_only!(diag, prefix * ".vt", Array(state[2]))
        return
    end
    # OptimiserChain state: (rule1_state, rule2_state, ...) — recurse into sub-tuples
    for s in state
        if s isa Tuple
            _collect_adam_moments!(diag, prefix, s)
            return
        end
    end
end

"""
    _walk_trees!(diag, prefix, ps, grads, opt_state)

Recursively walk Lux parameter / gradient / optimizer-state trees in parallel,
collecting diagnostics for each leaf array.
"""
function _walk_trees!(diag::Dict, prefix::String, ps, grads, opt_state)
    if ps isa AbstractArray
        ps_cpu = Array(ps)
        _collect_array_stats!(diag, prefix * ".param", ps_cpu)
        if grads isa AbstractArray
            _collect_array_stats!(diag, prefix * ".grad", Array(grads))
        end
        if opt_state isa Optimisers.Leaf
            _collect_adam_moments!(diag, prefix, opt_state.state)
        end
        return
    end
    # Recurse into NamedTuples / nested structures
    for k in keys(ps)
        child_ps = ps[k]
        child_g  = grads isa NamedTuple ? get(grads, k, nothing) : nothing
        child_o  = opt_state isa NamedTuple ? get(opt_state, k, nothing) : nothing
        _walk_trees!(diag, prefix == "" ? string(k) : prefix * "." * string(k),
                     child_ps, child_g, child_o)
    end
end

function collect_diagnostics!(diag::Dict, train_state, grads)
    _walk_trees!(diag, "", train_state.parameters, grads, train_state.optimizer_state)
end

# ============================================================================
#  Training Utilities
# ============================================================================

function cosine_lr(iter, n_iters, lr_max, lr_min, warmup)
    if iter <= warmup
        return lr_min + (lr_max - lr_min) * (iter / warmup)
    end
    progress = (iter - warmup) / (n_iters - warmup)
    return lr_min + 0.5f0 * (lr_max - lr_min) * (1 + cospi(progress))
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
            θ_N_numer = sample_θ_N(rng, M_NUISANCE, B_micro) |> xdev

            input_buffer = zeros(Float32, 2, N_STEPS, B_micro) |> xdev
            observations = zeros(Float32, N_STEPS, B_micro) |> xdev
            designs = zeros(Float32, N_STEPS, B_micro) |> xdev
            ε = randn(rng, Float32, N_STEPS, B_micro) |> xdev
            ll_denom = zeros(Float32, n_denom, B_micro) |> xdev
            ll_numer = zeros(Float32, M_NUISANCE, B_micro) |> xdev

            data = (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)

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

    # Save diagnostics + loss to disk
    serialize("diagnostics.jls", Dict("diagnostics" => diagnostics, "loss_history" => loss_history))

    # Save trained parameters (CPU arrays, no Reactant dependency to load)
    _to_cpu(x) = x
    _to_cpu(x::AbstractArray) = collect(x)
    _to_cpu(x::NamedTuple) = map(_to_cpu, x)
    _to_cpu(x::Tuple) = map(_to_cpu, x)
    serialize("checkpoint.jls", Dict(
        "parameters" => _to_cpu(train_state.parameters),
        "states"     => _to_cpu(train_state.states),
        "loss_history" => loss_history,
    ))
    println("Saved checkpoint.jls")

    return train_state, loss_history, diagnostics
end
