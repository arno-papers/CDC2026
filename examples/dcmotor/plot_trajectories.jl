using Plots, Random, Serialization, Statistics
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

# ============================================================================
#  Plot: per-step timing comparison (sPCE policy vs adaptive BIM)
#
#  Both timing matrices come from comparison_results.jls so that the
#  same trials (same true parameters, same seeds) are compared.
# ============================================================================

function plot_policy_timing(comparison_results;
        outfile=joinpath(@__DIR__, "results", "plot_policy_timing.png"))

    t_obs_ms = Float64.((1:N_STEPS) .* DT .* 1000)
    dt_ms = Float64(DT) * 1000
    dt_us = dt_ms * 1000

    spce_matrix_us = comparison_results["spce_step_times"] .* 1e6  # (N_STEPS, n_trials)
    bim_matrix_us  = comparison_results["bim_step_times"]  .* 1e6
    n_trials = size(spce_matrix_us, 2)

    p = plot(xlabel="Experiment step", ylabel="Time per step (μs)",
             title=@sprintf("Per-step computation time (%d trials)", n_trials),
             yscale=:log10, legend=:topright, size=(700, 400),
             bottom_margin=5Plots.mm, left_margin=5Plots.mm)

    # Helper: draw one boxplot at position t
    function draw_boxplot!(p, t, vals; color, fillcolor, w, label="")
        q1, q2, q3 = quantile(vals, [0.25, 0.5, 0.75])
        p001 = quantile(vals, 0.001)
        p999 = quantile(vals, 0.999)
        # Whiskers
        plot!(p, [t, t], [p001, q1]; color=color, lw=1, label="")
        plot!(p, [t, t], [q3, p999]; color=color, lw=1, label="")
        plot!(p, [t-w/2, t+w/2], [p001, p001]; color=color, lw=1, label="")
        plot!(p, [t-w/2, t+w/2], [p999, p999]; color=color, lw=1, label="")
        # Box (IQR)
        plot!(p, Shape([t-w, t+w, t+w, t-w], [q1, q1, q3, q3]);
              fillcolor=fillcolor, fillalpha=0.3, linecolor=color, lw=1,
              label=label)
        # Median
        plot!(p, [t-w, t+w], [q2, q2]; color=color, lw=2, label="")
    end

    w = dt_ms * 0.2  # box half-width

    # BIM boxplots (drawn first so sPCE is on top)
    for (i, t) in enumerate(t_obs_ms)
        draw_boxplot!(p, t + w * 1.1, bim_matrix_us[i, :];
                      color=:dodgerblue, fillcolor=:dodgerblue, w=w,
                      label=(i == 1 ? "Adaptive (BIM)" : ""))
    end

    # sPCE policy boxplots
    for (i, t) in enumerate(t_obs_ms)
        draw_boxplot!(p, t - w * 1.1, spce_matrix_us[i, :];
                      color=:purple, fillcolor=:mediumpurple, w=w,
                      label=(i == 1 ? "Adaptive (sPCE)" : ""))
    end

    # Sampling interval reference
    hline!(p, [dt_us]; color=:red, lw=2, ls=:dash,
           label=@sprintf("Δt = %.0f ms", dt_ms))

    save_plot(p, outfile)
    return p
end

# ============================================================================
#  Standalone
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")
    comparison_file = joinpath(results_dir, "comparison_results.jls")
    @assert isfile(comparison_file) "comparison_results.jls not found. Run eval_comparison.jl first."

    cr = deserialize(comparison_file)
    plot_policy_timing(cr)
end
