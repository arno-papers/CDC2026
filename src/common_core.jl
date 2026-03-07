# ============================================================================
# Core definitions for DADS experiments (CPU-safe, no Reactant).
#
# This file is meant to be `include`d AFTER model.jl (which defines dynamics(),
# constants, sampling, and the policy network). It contains generic
# infrastructure: RK4, CPU integration, positional encoding, training
# utilities, diagnostics, and I/O.
#
# CPU-only scripts include this file directly; GPU scripts include
# src/common.jl which adds Reactant-dependent pieces.
# ============================================================================

using Lux, Random
using Optimisers
using Printf, Dates
using Serialization

# ============================================================================
#  ODE integrator (CPU, calls dynamics() defined in model.jl)
# ============================================================================

function rk4_step(u, θ, d, dt)
    k1 = dynamics(u, θ, d)
    k2 = dynamics(u .+ 0.5f0 * dt .* k1, θ, d)
    k3 = dynamics(u .+ 0.5f0 * dt .* k2, θ, d)
    k4 = dynamics(u .+ dt .* k3, θ, d)
    return u .+ (dt / 6.0f0) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
end

function integrate_cpu(u, θ, d, dt, n_substeps)
    dt_sub = dt / n_substeps
    for _ in 1:n_substeps
        u = rk4_step(u, θ, d, dt_sub)
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

function _collect_adam_moments!(diag::Dict, prefix::String, state::Tuple)
    if length(state) >= 2 && state[1] isa AbstractArray && state[2] isa AbstractArray
        _collect_norm_only!(diag, prefix * ".mt", Array(state[1]))
        _collect_norm_only!(diag, prefix * ".vt", Array(state[2]))
        return
    end
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

Load training results from `dir/`.
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

# ============================================================================
#  Checkpoint loading
# ============================================================================

function load_checkpoint_cpu(path::AbstractString)
    if isdir(path)
        r = load_results(path)
        return r.parameters, r.states, path
    end
    @assert isfile(path) "Checkpoint not found: $path"
    ckpt = deserialize(path)
    return ckpt["parameters"], ckpt["states"], dirname(path)
end
