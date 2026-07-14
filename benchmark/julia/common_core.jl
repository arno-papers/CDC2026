# ============================================================================
# Core definitions for DADS experiments (CPU-safe, no Reactant).
#
# This file is meant to be `include`d AFTER model.jl (which defines dynamics(),
# constants, sampling, and the policy network). It contains generic
# infrastructure: RK4, CPU integration, positional encoding, training
# utilities, and I/O.
#
# CPU-only scripts include this file directly; GPU scripts include
# src/common.jl which adds Reactant-dependent pieces.
# ============================================================================

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
#  Cosine Learning Rate
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
_to_cpu(x::AbstractArray) = Array(x)
_to_cpu(x::NamedTuple) = map(_to_cpu, x)
_to_cpu(x::Tuple) = map(_to_cpu, x)

"""
    save_results(dir, train_state, loss_history)

Save training checkpoint to `dir/checkpoint.jls` (parameters, states, loss history).
"""
function save_results(dir::AbstractString, train_state, loss_history)
    mkpath(dir)
    serialize(joinpath(dir, "checkpoint.jls"), Dict(
        "parameters"   => _to_cpu(train_state.parameters),
        "states"       => _to_cpu(train_state.states),
        "loss_history" => loss_history,
    ))
    println("Saved results to $dir/")
end

# ============================================================================
#  Checkpoint loading
# ============================================================================

function load_checkpoint_cpu(path::AbstractString)
    dir = isdir(path) ? path : ((@assert isfile(path) "Checkpoint not found: $path"); dirname(path))
    ckpt_path = isdir(path) ? joinpath(path, "checkpoint.jls") : path
    ckpt = open(deserialize, ckpt_path)
    return ckpt["parameters"], ckpt["states"], dir
end

# ============================================================================
#  Static design loading (shared across eval/plot scripts)
# ============================================================================

"""
    load_static_designs(results_dir; T=Float32)

Load static designs from `results_dir`, returning `Vector{Pair{String, Vector{T}}}`.
Handles both new (`design_bim.jls`) and legacy (`bim_std_design.jls`) filenames,
and both new (`"design"`) and legacy (`"static_design"`) dict keys.
"""
function load_static_designs(results_dir::AbstractString; T::Type=Float32)
    designs = Pair{String, Vector{T}}[]
    for (name, filename, legacy_filename) in [
        ("static_std",  "design_bim.jls",  "bim_std_design.jls"),
        ("static_spce", "design_spce.jls", "spce_static_design.jls"),
    ]
        path = joinpath(results_dir, filename)
        if !isfile(path)
            path = joinpath(results_dir, legacy_filename)
        end
        isfile(path) || continue
        d = deserialize(path)
        key = haskey(d, "design") ? "design" : "static_design"
        push!(designs, name => T.(d[key]))
    end
    return designs
end
