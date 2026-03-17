# Small standalone utilities (no model or Reactant dependencies).
# Included by model.jl before common_core.jl / common.jl.

"""
    allocate_budget(C; lambda_L=1.0, lambda_M=1.0, B_multiplier=1) -> (L, M, B)

Jointly optimize L (contrastive), M (nuisance), B (batch) for budget C.
Minimizes `1/(B*B_multiplier) + λ_L/(L+1)² + λ_M/M²` subject to `B = C÷(L+2+M)`.
Uses analytical C^{1/3} scaling to seed a fast local search.

When `B_multiplier > 1` (gradient accumulation), the effective batch size is
`B * B_multiplier`, so the returned B is the per-microbatch size and the
optimizer shifts budget toward larger L, M.
"""
function allocate_budget(C::Int; lambda_L::Float64=1.0, lambda_M::Float64=1.0,
                         B_multiplier::Int=1)
    # Analytical seeds: L*+1 ≈ (2λ_L C B_multiplier)^{1/3}, M* ≈ (2λ_M C B_multiplier)^{1/3}
    L_seed = max(1, round(Int, (2*lambda_L*C*B_multiplier)^(1/3)) - 1)
    M_seed = max(1, round(Int, (2*lambda_M*C*B_multiplier)^(1/3)))

    # Local grid search around analytical solution
    w = max(20, L_seed ÷ 3)
    best_L, best_M, best_B, best_obj = L_seed, M_seed, fld(C, L_seed+2+M_seed), Inf
    for L in max(1, L_seed-w):(L_seed+w)
        for M in max(1, M_seed-w):(M_seed+w)
            B = fld(C, L + 2 + M)
            B < 1 && break
            obj = 1.0/(B * B_multiplier) + lambda_L/(L+1)^2 + lambda_M/M^2
            if obj < best_obj
                best_obj, best_L, best_M, best_B = obj, L, M, B
            end
        end
    end
    return (best_L, best_M, best_B)
end

# ============================================================================
#  Utility: save figure + print path
# ============================================================================

using Plots
using Printf
using Statistics

function save_plot(plt, path)
    mkpath(dirname(path))
    savefig(plt, path)
    println("Saved: $path")
end

# ============================================================================
#  Callback: live loss plot during training
# ============================================================================

function loss_plot_callback(; title="Training Loss", output_path="plot_training_loss.png",
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
