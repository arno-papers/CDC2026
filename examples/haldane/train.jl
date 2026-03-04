include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "args.jl"))

function git_info_or_unknown(args...)
    try
        return strip(read(`git $(args...)`, String))
    catch
        return "unknown"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    plotting = parse_bool(ARGS, "plotting"; default=false)
    n_iters = parse_int(ARGS, "n_iters"; default=250)
    seed = parse_int(ARGS, "seed"; default=0)
    loss_png_every = parse_int(ARGS, "loss_png_every"; default=10)
    grad_accum = parse_int(ARGS, "grad_accum"; default=GRAD_ACCUM_STEPS)
    lr_max = parse_float(ARGS, "lr_max"; default=0.003f0)
    lr_min = parse_float(ARGS, "lr_min"; default=1f-5)
    warmup = parse_int(ARGS, "warmup"; default=50)
    clip_norm = parse_float(ARGS, "clip_norm"; default=1.0f0)
    results_dir = parse_kwarg(ARGS, "results_dir"; default=joinpath(@__DIR__, "results"))

    loss_png_every = loss_png_every < 1 ? 10 : loss_png_every
    B_micro = GRAD_BATCH ÷ grad_accum

    using Plots

    Reactant.set_default_backend("gpu")

    mkpath(results_dir)
    branch = git_info_or_unknown("rev-parse", "--abbrev-ref", "HEAD")
    commit = git_info_or_unknown("rev-parse", "--short", "HEAD")
    open(joinpath(results_dir, "config.txt"), "w") do io
        println(io, "branch: $branch")
        println(io, "commit: $commit")
        println(io, "date: $(Dates.format(Dates.now(), "yyyy-mm-dd"))")
        println(io, "experiment: Haldane ODE_BUDGET=$(ODE_BUDGET_TRAJ), cosine LR")
        println(io)
        println(io, "# Hyperparameters")
        println(io, "n_iters = $n_iters")
        println(io, "lr_max = $lr_max")
        println(io, "lr_min = $lr_min")
        println(io, "warmup = $warmup")
        println(io, "optimizer = Adam")
        println(io, "grad_accum = $grad_accum")
        println(io, "B_micro = $B_micro")
        println(io, "B_total = $GRAD_BATCH")
        println(io, "L_contrastive = $L_CONTRASTIVE")
        println(io, "M_NUISANCE = $M_NUISANCE  # joint (sigma, Cx0) samples")
        println(io, "N_STEPS = $N_STEPS")
        println(io, "N_SUBSTEPS = $N_SUBSTEPS")
        println(io, "DT = $DT")
        println(io, "seed = $seed")
        println(io, "ODE_BUDGET_TRAJ = $ODE_BUDGET_TRAJ")
        println(io, "SPIKE_PROB = $SPIKE_PROB")
        println(io, "SPIKE_STD = $SPIKE_STD")
        println(io, "SLAB_MEAN = $SLAB_MEAN")
        println(io, "SLAB_STD = $SLAB_STD")
    end

    println("\n=== Targeted DADS Training — Haldane (Reactant + Enzyme) ===")
    println("Target params: (alpha), Nuisance: (mu_max, K_s, sigma, Cx0)")
    println("L = $L_CONTRASTIVE contrastive, M_NUISANCE = $M_NUISANCE (joint sigma,Cx0), B = $GRAD_BATCH total ($(grad_accum)x$(B_micro) micro)")
    println("n_iters = $n_iters, lr_max = $lr_max, lr_min = $lr_min, warmup = $warmup")
    println("grad_accum = $grad_accum, clip_norm = $clip_norm, plotting = $plotting")
    println("results_dir = $results_dir")
    println("loss_png_every = $loss_png_every\n")

    rng = Random.MersenneTwister(seed)
    ps, st = Lux.setup(rng, policy)

    xdev = reactant_device()
    println("Using device: ", xdev)

    ps_ra = ps |> xdev
    st_ra = st |> xdev

    on_iteration = (iter, _loss, loss_history, _) -> begin
        if iter % loss_png_every == 0 || iter == 1 || iter == n_iters
            p = Plots.plot(loss_history;
                xlabel = "Iteration",
                ylabel = "Targeted sPCE Loss",
                title = "Training Loss (Haldane)",
                label = "loss",
                linewidth = 2,
            )
            Plots.savefig(p, joinpath(results_dir, "plot_loss_live.png"))
        end
    end

    println("Starting training...")
    t_start = time()
    train_state, loss_history, diagnostics = train_policy(
        policy, ps_ra, st_ra, rng;
        xdev = xdev,
        n_iters = n_iters,
        on_iteration = on_iteration,
        lr_max = lr_max,
        lr_min = lr_min,
        warmup = warmup,
        grad_accum = grad_accum,
        clip_norm = clip_norm,
        save_dir = results_dir,
    )
    t_train = time() - t_start
    best_loss, best_iter = findmin(loss_history)
    per_iter = t_train / n_iters
    println("\nTraining complete.")
    @printf("Training time: %.1fs (%.1fs/iter)\n", t_train, per_iter)
    @printf("Best loss: %.7f @ iter %d\n", best_loss, best_iter)

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
end
