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

if abspath(PROGRAM_FILE) == @__FILE__
    plotting = parse_bool(ARGS, "plotting"; default=false)
    n_iters = parse_int(ARGS, "n_iters"; default=1000)
    seed = parse_int(ARGS, "seed"; default=0)

    Reactant.set_default_backend("gpu")

    println("\n=== Targeted DADS Training (Reactant + Enzyme) ===")
    println("Target params: (mu_max, K_s), Nuisance: sigma_measure")
    println("L = $L_CONTRASTIVE contrastive, M = $M_NUISANCE nuisance samples")
    println("n_iters = $n_iters, plotting = $plotting\n")

    rng = Random.MersenneTwister(seed)
    ps, st = Lux.setup(rng, policy)

    xdev = reactant_device()
    println("Using device: ", xdev)

    ps_ra = ps |> xdev
    st_ra = st |> xdev

    println("Starting training...")
    train_state, loss_history = train_policy(policy, ps_ra, st_ra, rng; xdev=xdev, n_iters=n_iters)
    println("\nTraining complete.")

    if plotting
        include("plot_trajectories.jl")
        plot_trajectories(policy, train_state; rng=rng)
    end
end
