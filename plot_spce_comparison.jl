#!/usr/bin/env julia
# ============================================================================
# Standardized sPCE comparison: score histograms + design trajectory plots.
#
# Standalone usage:
#   julia --project plot_spce_comparison.jl scores=path/to/spce_scores.jls [output=dir]
#
# Reusable via include("plot_spce_comparison.jl") from eval scripts.
# ============================================================================

using Distributions: TDist, ccdf
using Plots
using Printf
using Serialization
using Statistics

# ============================================================================
#  Canonical design styles
# ============================================================================

const DESIGN_STYLES = Dict(
    "adaptive"     => (label = "Adaptive policy",       color = :gray20),
    "static_avg"   => (label = "Static (mean adaptive)", color = :darkorange),
    "static_std"   => (label = "Static (std BIM)",       color = :dodgerblue),
    "static_cheat" => (label = "Static (cheat BIM)",     color = :crimson),
    "static_spce"  => (label = "Static (sPCE-opt)",      color = :forestgreen),
)

const DESIGN_ORDER = ["adaptive", "static_avg", "static_std", "static_cheat", "static_spce"]

# ============================================================================
#  Extract designs from scores dict → ordered Vector{Pair{String, Vector{Float64}}}
# ============================================================================

function extract_designs(d::Dict)
    designs = Pair{String, Vector{Float64}}[]
    for key in DESIGN_ORDER
        scores_key = key == "adaptive" ? "adaptive_scores" : "$(key)_scores"
        if haskey(d, scores_key)
            push!(designs, key => Float64.(d[scores_key]))
        end
    end
    return designs
end

# ============================================================================
#  Compute summary stats: mean ± std per design, paired deltas vs adaptive
# ============================================================================

function compute_summary_stats(designs::Vector{Pair{String, Vector{Float64}}})
    lines = String[]

    # Find adaptive scores (baseline)
    adaptive_idx = findfirst(p -> p.first == "adaptive", designs)
    adaptive_scores = adaptive_idx !== nothing ? designs[adaptive_idx].second : nothing

    # Per-design stats
    push!(lines, "=== Results (targeted sPCE, higher = more informative) ===")
    push!(lines, "")
    for (key, scores) in designs
        style = get(DESIGN_STYLES, key, (label = key, color = :black))
        n = length(scores)
        suffix = key == "adaptive" ? "  (baseline)" : ""
        push!(lines, @sprintf("  %-30s  mean = %8.4f  std = %8.4f  (n=%d)%s",
                               style.label, mean(scores), std(scores), n, suffix))
    end

    # Paired deltas (adaptive - each static)
    if adaptive_scores !== nothing
        push!(lines, "")
        for (key, scores) in designs
            key == "adaptive" && continue
            style = get(DESIGN_STYLES, key, (label = key, color = :black))
            delta = adaptive_scores .- scores
            n = length(delta)
            m = mean(delta)
            sem = std(delta) / sqrt(n)
            t_stat = m / sem
            p_val = 2 * ccdf(TDist(n - 1), abs(t_stat))
            push!(lines, @sprintf("  Paired: Adaptive - %-17s  delta = %+.4f ± %.4f (SEM)  t=%6.2f  p=%.2e",
                                   style.label, m, sem, t_stat, p_val))
        end
    end

    return join(lines, "\n")
end

# ============================================================================
#  Plot: sPCE score distribution histogram (single panel)
# ============================================================================

function plot_spce_histograms(designs::Vector{Pair{String, Vector{Float64}}};
                               output_path::String = "plot_spce_histograms.png",
                               title_suffix::String = "",
                               nbins::Union{Int, Nothing} = nothing)

    adaptive_idx = findfirst(p -> p.first == "adaptive", designs)
    adaptive_scores = adaptive_idx !== nothing ? designs[adaptive_idx].second : nothing

    # Auto-scale bins: clamped to [10, 50]
    n_samples = length(designs[1].second)
    bins = nbins !== nothing ? nbins : clamp(round(Int, sqrt(n_samples)), 10, 50)

    p = plot(; xlabel = "sPCE score", ylabel = "density",
               title = "sPCE score distributions" * title_suffix,
               legend = :topleft, size = (800, 450))

    # Draw adaptive mean first as prominent baseline reference
    if adaptive_scores !== nothing
        adaptive_mean = mean(adaptive_scores)
        vline!(p, [adaptive_mean]; color = :gray20, lw = 3, ls = :dash,
               label = @sprintf("Adaptive mean = %.3f", adaptive_mean))
    end

    # Histograms for each design
    for (key, scores) in designs
        style = get(DESIGN_STYLES, key, (label = key, color = :black))
        histogram!(p, scores;
                   normalize = :pdf, bins = bins,
                   fillalpha = 0.25, linewidth = 0.5,
                   color = style.color, label = style.label)
        if key != "adaptive"
            m = mean(scores)
            vline!(p, [m]; color = style.color, lw = 1.5, ls = :solid,
                   label = @sprintf("mean = %.3f", m))
        end
    end

    savefig(p, output_path)
    println("Saved: $output_path")
    return p
end

# ============================================================================
#  Plot: design trajectories (Q_in over time steps)
#
#  adaptive_designs: N_STEPS × n_trials matrix of adaptive rollouts
#  static_designs:   Vector{Pair{String, Vector}} of static designs
# ============================================================================

function plot_design_trajectories(adaptive_designs::AbstractMatrix,
                                  static_designs::Vector{<:Pair};
                                  output_path::String = "plot_design_trajectories.png",
                                  n_show::Int = 50)
    n_steps = size(adaptive_designs, 1)
    n_trials = size(adaptive_designs, 2)
    steps = collect(1:n_steps)

    p = plot(; xlabel = "step", ylabel = "Q_in (L/h)",
               title = "Design comparison ($n_trials adaptive rollouts)",
               legend = :topleft, ylims = (-0.5, 10.5), size = (800, 450))

    # Adaptive rollouts (translucent)
    n_draw = min(n_show, n_trials)
    for j in 1:n_draw
        plot!(p, steps, adaptive_designs[:, j];
              color = :gray60, alpha = 0.15, lw = 0.8,
              label = (j == 1 ? "Adaptive rollouts" : ""))
    end

    # Mean adaptive as static baseline
    avg_adaptive = vec(mean(adaptive_designs; dims = 2))
    plot!(p, steps, avg_adaptive;
          color = :gray20, lw = 3, ls = :dash,
          label = @sprintf("Mean adaptive"))

    # Static designs
    for (key, design) in static_designs
        style = get(DESIGN_STYLES, key, (label = key, color = :black))
        plot!(p, steps, design;
              color = style.color, lw = 2.5, label = style.label)
    end

    savefig(p, output_path)
    println("Saved: $output_path")
    return p
end

# ============================================================================
#  Standalone CLI
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    function _parse_kwarg(args, key; default=nothing)
        prefix = key * "="
        for a in args
            startswith(a, prefix) && return split(a, "=", limit=2)[2]
        end
        return default
    end

    scores_path = _parse_kwarg(ARGS, "scores"; default=nothing)
    if scores_path === nothing
        println("Usage: julia --project plot_spce_comparison.jl scores=path/to/spce_scores.jls [output=dir]")
        exit(1)
    end

    output_dir = _parse_kwarg(ARGS, "output"; default=dirname(scores_path))
    mkpath(output_dir)

    d = deserialize(scores_path)
    designs = extract_designs(d)

    if isempty(designs)
        println("No score arrays found in $scores_path")
        exit(1)
    end

    # Print summary
    summary = compute_summary_stats(designs)
    println()
    println(summary)
    println()

    # Histogram
    out_png = joinpath(output_dir, "plot_spce_histograms.png")
    n_trials = get(d, "n_trials", length(designs[1].second))
    L = get(d, "L", "?")
    M = get(d, "M", "?")
    plot_spce_histograms(designs; output_path=out_png,
                          title_suffix=" (L=$L, M=$M, $n_trials trials)")

    # Save summary text
    summary_path = joinpath(output_dir, "spce_summary.txt")
    open(summary_path, "w") do io
        println(io, summary)
    end
    println("Saved: $summary_path")
end
