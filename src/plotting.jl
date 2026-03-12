# ============================================================================
# Standardized sPCE comparison: score histograms + design trajectory plots.
#
# Reusable via include(joinpath(@__DIR__, "..", "src", "plotting.jl"))
# ============================================================================

using Plots
using Printf
using Statistics

# ============================================================================
#  Utility: save figure + print path
# ============================================================================

function save_plot(plt, path)
    mkpath(dirname(path))
    savefig(plt, path)
    println("Saved: $path")
end

# ============================================================================
#  Callback: live loss plot during training
# ============================================================================

function loss_plot_callback(; title="Training Loss", output_path="plot_loss_live.png",
                              save_every=10, n_iters=0)
    return (iter, _loss, loss_history, _train_state) -> begin
        if iter % save_every == 0 || iter == 1 || (n_iters > 0 && iter == n_iters)
            p = Plots.plot(loss_history;
                xlabel="Iteration", ylabel="Targeted sPCE Loss",
                title=title, label="loss", linewidth=2)
            save_plot(p, output_path)
        end
    end
end

# ============================================================================
#  Canonical design styles
# ============================================================================

const DESIGN_STYLES = Dict(
    "adaptive"     => (label = "Adaptive policy",       color = :gray20),
    "static_std"   => (label = "Static (BIM)",           color = :dodgerblue),
    "static_spce"  => (label = "Static (sPCE-opt)",      color = :forestgreen),
)

const DESIGN_ORDER = ["adaptive", "static_std", "static_spce"]

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
#  Plot: sPCE score distribution histogram
# ============================================================================

function plot_spce_histograms(designs::Vector{Pair{String, Vector{Float64}}};
                               output_path::String = "plot_spce_histograms.png",
                               title_suffix::String = "",
                               nbins::Union{Int, Nothing} = nothing)

    adaptive_idx = findfirst(p -> p.first == "adaptive", designs)
    adaptive_scores = adaptive_idx !== nothing ? designs[adaptive_idx].second : nothing

    n_samples = length(designs[1].second)
    bins = nbins !== nothing ? nbins : clamp(round(Int, sqrt(n_samples)), 10, 50)

    p = plot(; xlabel = "sPCE score", ylabel = "density",
               title = "sPCE score distributions" * title_suffix,
               legend = :topleft, size = (800, 450))

    if adaptive_scores !== nothing
        adaptive_mean = mean(adaptive_scores)
        vline!(p, [adaptive_mean]; color = :gray20, lw = 3, ls = :dash,
               label = @sprintf("Adaptive mean = %.3f", adaptive_mean))
    end

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

    save_plot(p, output_path)
    return p
end

# ============================================================================
#  Plot: design trajectories (d over time steps)
# ============================================================================

function plot_design_trajectories(adaptive_designs::AbstractMatrix,
                                  static_designs::Vector{<:Pair};
                                  output_path::String = "plot_design_trajectories.png",
                                  n_show::Int = 50,
                                  design_ylabel::String = "design")
    n_steps = size(adaptive_designs, 1)
    n_trials = size(adaptive_designs, 2)
    steps = collect(1:n_steps)

    p = plot(; xlabel = "step", ylabel = design_ylabel,
               title = "Design comparison ($n_trials adaptive rollouts)",
               legend = :topleft, ylims = (-0.5, 10.5), size = (800, 450))

    n_draw = min(n_show, n_trials)
    for j in 1:n_draw
        plot!(p, steps, adaptive_designs[:, j];
              color = :gray60, alpha = 0.15, lw = 0.8,
              label = (j == 1 ? "Adaptive rollouts" : ""))
    end

    for (key, design) in static_designs
        style = get(DESIGN_STYLES, key, (label = key, color = :black))
        plot!(p, steps, design;
              color = style.color, lw = 2.5, label = style.label)
    end

    save_plot(p, output_path)
    return p
end
