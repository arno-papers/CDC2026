# ============================================================================
# Core definitions for DADS experiments (CPU-safe, no Reactant).
#
# This file is meant to be `include`d by scripts. It contains everything
# except Reactant-dependent code (integrate with @trace, loss, training).
# CPU-only scripts (plot_trajectories.jl, compare_static_bim.jl) include
# this file directly; training.jl includes common.jl which adds Reactant.
# ============================================================================

using Lux, Random
using Optimisers
using Printf, Dates
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
    @return 10.0f0 .* sigmoid.(output_head(x[:, end, :]))
end

# ============================================================================
#  Targeted sPCE Loss with Nuisance Parameters
#
#  θ = (μ_max, K_s, σ, Cx0) - full parameters (4D)
#       θ_T = (μ_max, K_s)   - target parameters
#       θ_N = (σ, Cx0)       - nuisance parameters
#
#  Joint sampling: each of M_NUISANCE samples is a (σ, Cx0) pair with its
#  own ODE trajectory, giving a clean 2D (M_NUISANCE, B) logsumexp.
#  Denominator: average over L+1 full θ samples.
# ============================================================================

const N_STEPS = 14
const DT = 1.0f0            # Total time per control interval
const N_SUBSTEPS = 500      # Integration substeps per control interval

# -----------------------------------------------------------------------------
# Training budget allocation
#
# Budget: C_traj = B * (L + 2 + M_NUISANCE) trajectory rollouts per update.
#   Breakdown: 1 generation + (L+1) denominator + M_NUISANCE numerator.
#   Each joint (σ, Cx0) sample needs its own ODE trajectory.
# Optimal (L, B) minimizes MSE proxy: 1/B + λ/(L+1)² subject to budget.
# -----------------------------------------------------------------------------
const ODE_BUDGET_TRAJ = 2121728
const M_NUISANCE = 128   # Joint (σ, Cx0) samples; each needs one ODE trajectory

const (L_CONTRASTIVE, GRAD_BATCH) = let
    C = ODE_BUDGET_TRAJ
    λ = 1.0  # equal weight on variance (1/B) vs squared bias (1/(L+1)²)
    best_L, best_B, best_obj = 1, fld(C, 3 + M_NUISANCE), Inf
    for L in 1:(C - 2 - M_NUISANCE)
        B = fld(C, L + 2 + M_NUISANCE)
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
const GRAD_ACCUM_STEPS = 16
const GRAD_BATCH_MICRO = GRAD_BATCH ÷ GRAD_ACCUM_STEPS

# Prior bounds
const μ_max_lo, μ_max_hi = 0.3f0, 0.5f0
const K_s_lo, K_s_hi = 0.3f0, 0.6f0
const σ_lo, σ_hi = 0.05f0, 0.15f0  # Nuisance: measurement noise std
const Cx0_lo, Cx0_hi = 0.10f0, 0.50f0   # Nuisance: initial biomass

# ASCII aliases (useful for scripts / CLI usage)
const mu_max_lo, mu_max_hi = μ_max_lo, μ_max_hi
const sigma_lo, sigma_hi = σ_lo, σ_hi

# ============================================================================
#  Sampling
# ============================================================================

function sample_θ_full(rng, n_samples)
    # Full parameters: (μ_max, K_s, σ, Cx0)
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
    σ = rand(rng, Float32, M, B)
    σ .= σ_lo .+ (σ_hi - σ_lo) .* σ
    Cx0 = rand(rng, Float32, M, B)
    Cx0 .= Cx0_lo .+ (Cx0_hi - Cx0_lo) .* Cx0
    return σ, Cx0
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
#  Results I/O
# ============================================================================

_to_cpu(x) = x
_to_cpu(x::AbstractArray) = collect(x)
_to_cpu(x::NamedTuple) = map(_to_cpu, x)
_to_cpu(x::Tuple) = map(_to_cpu, x)

"""
    save_results(dir, train_state, loss_history, diagnostics)

Save training results to `dir/`:
- `checkpoint.jls` — trained parameters (CPU), model states, and loss history
- `diagnostics.jls` — per-layer gradient/optimizer diagnostics + loss history
"""
function save_results(dir::AbstractString, train_state, loss_history, diagnostics)
    mkpath(dir)
    serialize(joinpath(dir, "diagnostics.jls"),
        Dict("diagnostics" => diagnostics, "loss_history" => loss_history))
    serialize(joinpath(dir, "checkpoint.jls"), Dict(
        "parameters"   => _to_cpu(train_state.parameters),
        "states"       => _to_cpu(train_state.states),
        "loss_history" => loss_history,
    ))
    println("Saved results to $dir/")
end

"""
    load_results(dir) -> (; parameters, states, loss_history, diagnostics)

Load training results from `dir/`. Returns a named tuple with:
- `parameters` — trained model weights (CPU `Float32` arrays)
- `states` — Lux model states
- `loss_history` — `Vector{Float32}` of per-iteration losses
- `diagnostics` — `Dict{String, Vector{Float32}}` of per-layer metrics

To resume training on GPU:

    r = load_results("results/2x_budget_cosine_lr")
    xdev = reactant_device()
    ps_ra = r.parameters |> xdev
    st_ra = r.states |> xdev
"""
function load_results(dir::AbstractString)
    ckpt_path = joinpath(dir, "checkpoint.jls")
    diag_path = joinpath(dir, "diagnostics.jls")

    ckpt = open(deserialize, ckpt_path)
    ps = ckpt["parameters"]
    st = ckpt["states"]
    loss_history = ckpt["loss_history"]

    diagnostics = if isfile(diag_path)
        d = open(deserialize, diag_path)
        d["diagnostics"]
    else
        Dict{String, Vector{Float32}}()
    end

    return (; parameters=ps, states=st, loss_history, diagnostics)
end
