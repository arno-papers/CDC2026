include("common.jl")

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
    if v === nothing
        return default
    end
    v = lowercase(v)
    return v in ("1", "true", "t", "yes", "y")
end

function parse_int(args, key; default::Int)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Int, v)
end

function parse_float(args, key; default::Float32)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Float32, v)
end

if abspath(PROGRAM_FILE) == @__FILE__
    plotting = parse_bool(ARGS, "plotting"; default=false)
    n_iters = parse_int(ARGS, "n_iters"; default=1000)
    seed = parse_int(ARGS, "seed"; default=0)
    loss_png_every = parse_int(ARGS, "loss_png_every"; default=10)
    grad_accum = parse_int(ARGS, "grad_accum"; default=GRAD_ACCUM_STEPS)
    lr_max = parse_float(ARGS, "lr_max"; default=0.003f0)
    lr_min = parse_float(ARGS, "lr_min"; default=1f-5)
    warmup = parse_int(ARGS, "warmup"; default=50)
    clip_norm = parse_float(ARGS, "clip_norm"; default=1.0f0)

    loss_png_every = loss_png_every < 1 ? 10 : loss_png_every
    B_micro = GRAD_BATCH ÷ grad_accum

    using Plots

    Reactant.set_default_backend("gpu")

    println("\n=== Targeted DADS Training (Reactant + Enzyme) ===")
    println("Target params: (mu_max, K_s), Nuisance: sigma_measure")
    println("L = $L_CONTRASTIVE contrastive, M = $M_NUISANCE nuisance samples, B = $GRAD_BATCH total ($(grad_accum)×$(B_micro) micro)")
    println("n_iters = $n_iters, lr_max = $lr_max, lr_min = $lr_min, warmup = $warmup")
    println("grad_accum = $grad_accum, clip_norm = $clip_norm, plotting = $plotting")
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
                title = "Training Loss",
                label = "loss",
                linewidth = 2,
            )
            Plots.savefig(p, "plot_loss_live.png")
        end
    end

    println("Starting training...")
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
    )
    println("\nTraining complete.")

    if plotting
        include("plot_trajectories.jl")
        plot_trajectories(policy, train_state; rng=rng)
    end
end
