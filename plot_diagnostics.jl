# ============================================================================
# Post-training diagnostics plotter.
#
# Usage:  julia plot_diagnostics.jl [diagnostics.jls]
# ============================================================================

using Serialization, Plots

file = length(ARGS) >= 1 ? ARGS[1] : "diagnostics.jls"
@assert isfile(file) "File not found: $file"

data = deserialize(file)
diag = data["diagnostics"]
loss_history = data["loss_history"]

# Helper: collect keys matching a suffix, sorted for deterministic legend order
matching_keys(suffix) = sort(filter(k -> endswith(k, suffix), collect(keys(diag))))

# Short display name: strip the trailing stat suffix
display_name(k, suffix) = replace(k, suffix => "")

function plot_series(suffix; title, ylabel, yscale=:identity)
    ks = matching_keys(suffix)
    isempty(ks) && return plot(; title, ylabel, legend=false)
    p = plot(; title, ylabel, xlabel="Iteration", yscale, legend=:outertopright, size=(700, 400))
    for k in ks
        plot!(p, diag[k]; label=display_name(k, suffix), linewidth=1.5)
    end
    return p
end

# --- Main plots ---
p1 = plot_series(".grad.norm";  title="Gradient L2 Norms",  ylabel="‖∇‖₂", yscale=:log10)
p2 = plot_series(".param.norm"; title="Parameter L2 Norms", ylabel="‖θ‖₂")
p3 = plot_series(".mt.norm";    title="Adam 1st Moment (mₜ) Norms", ylabel="‖mₜ‖₂", yscale=:log10)
p4 = plot_series(".vt.norm";    title="Adam 2nd Moment (vₜ) Norms", ylabel="‖vₜ‖₂", yscale=:log10)

# Loss curve
p_loss = plot(loss_history; xlabel="Iteration", ylabel="Loss", title="Training Loss",
              label="loss", linewidth=2)

# NaN / Inf flags
nan_keys = matching_keys(".has_nan")
nan_iters = Int[]
for k in nan_keys
    for (i, v) in enumerate(diag[k])
        v > 0 && push!(nan_iters, i)
    end
end
unique!(nan_iters)
if !isempty(nan_iters)
    @warn "NaN/Inf detected at iterations: $(nan_iters)"
end

# Compose and save
layout = @layout [a b; c d; e{0.3h}]
p_all = plot(p1, p2, p3, p4, p_loss; layout, size=(1400, 1100), margin=5Plots.mm)
savefig(p_all, "plot_diagnostics.png")
println("Saved plot_diagnostics.png")
