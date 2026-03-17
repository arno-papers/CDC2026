include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))

using Dates, Random
plotting = false
n_iters = 1000
seed = 0
Random.seed!(seed)
loss_png_every = 10
grad_accum = GRAD_ACCUM_STEPS
ode_budget = ODE_BUDGET_TRAJ
lr_max = 0.003f0
lr_min = 1f-5
warmup = 50
results_dir = joinpath(@__DIR__, "results")

L, M_nuis, B_micro = allocate_budget(ode_budget; B_multiplier=grad_accum)
B_total = B_micro * grad_accum
loss_png_every = loss_png_every < 1 ? 10 : loss_png_every

using Plots

Reactant.set_default_backend("gpu")

mkpath(results_dir)
open(joinpath(results_dir, "config.txt"), "w") do io
    println(io, "date: $(Dates.format(Dates.now(), "yyyy-mm-dd"))")
    println(io, "experiment: ODE_BUDGET=$(ode_budget), cosine LR")
    println(io)
    println(io, "# Hyperparameters")
    println(io, "n_iters = $n_iters")
    println(io, "lr_max = $lr_max")
    println(io, "lr_min = $lr_min")
    println(io, "warmup = $warmup")
    println(io, "optimizer = Adam")
    println(io, "grad_accum = $grad_accum")
    println(io, "B_micro = $B_micro")
    println(io, "B_total = $B_total")
    println(io, "L_contrastive = $L")
    println(io, "M_nuisance = $M_nuis")
    println(io, "N_STEPS = $N_STEPS")
    println(io, "N_SUBSTEPS = $N_SUBSTEPS")
    println(io, "DT = $DT")
    println(io, "seed = $seed")
    println(io, "ode_budget = $ode_budget")
end

println("\n=== Targeted DADS Training (Reactant + Enzyme) ===")
println("Target params: (mu_max, K_s), Nuisance: (sigma, Cx0) jointly sampled")
println("L = $L contrastive, M = $M_nuis nuisance, B = $B_total total ($(grad_accum)x$(B_micro) micro)")
println("n_iters = $n_iters, lr_max = $lr_max, lr_min = $lr_min, warmup = $warmup")
println("grad_accum = $grad_accum, plotting = $plotting")
println("results_dir = $results_dir")
println("loss_png_every = $loss_png_every\n")

rng = Random.MersenneTwister(seed)
ps, st = Lux.setup(rng, policy)

xdev = reactant_device()
println("Using device: ", xdev)

ps_ra = ps |> xdev
st_ra = st |> xdev

on_iteration = loss_plot_callback(;
    title="Training Loss",
    output_path=joinpath(results_dir, "plot_training_loss.png"),
    save_every=loss_png_every, n_iters)

println("Starting training...")
t_start = time()
train_state, loss_history = train_policy(
    policy, ps_ra, st_ra, rng;
    xdev = xdev,
    n_iters = n_iters,
    on_iteration = on_iteration,
    lr_max = lr_max,
    lr_min = lr_min,
    warmup = warmup,
    grad_accum = grad_accum,
    grad_batch = B_total,
    L = L,
    M = M_nuis,
    save_dir = results_dir,
)
t_train = time() - t_start
best_loss, best_iter = findmin(loss_history)
per_iter = t_train / n_iters
println("\nTraining complete.")
@printf("Training time: %.1fs (%.1fs/iter)\n", t_train, per_iter)
@printf("Best loss: %.7f @ iter %d\n", best_loss, best_iter)

if plotting
    include(joinpath(@__DIR__, "plot_trajectories.jl"))
    plot_trajectories(policy, train_state; rng=rng, outfile=joinpath(results_dir, "plot_trajectories.png"))
end

t_total = time() - t_start
@printf("Total wall time (incl. compilation): %.1fs\n", t_total)

open(joinpath(results_dir, "config.txt"), "a") do io
    println(io)
    println(io, "# Results")
    @printf(io, "best_loss = %.7f\n", best_loss)
    println(io, "best_iter = $best_iter")
    @printf(io, "training_time_s = %.1f\n", t_train)
    @printf(io, "per_iter_s = %.1f\n", per_iter)
    @printf(io, "total_wall_time_s = %.1f\n", t_total)
end
