# ============================================================================
# Argument parsing helpers for CLI scripts.
#
# Usage:  include(joinpath(@__DIR__, "..", "src", "args.jl"))
# ============================================================================

function parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        if startswith(a, prefix)
            return split(a, "=", limit = 2)[2]
        end
    end
    return default
end

function parse_bool(args, key; default=false)
    v = parse_kwarg(args, key; default=nothing)
    v === nothing && return default
    return lowercase(v) in ("1", "true", "t", "yes", "y")
end

function parse_int(args, key; default::Int)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Int, v)
end

function parse_float(args, key; default::Float32)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Float32, v)
end

function parse_float64(args, key; default::Float64)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Float64, v)
end
