# Find the maximum ODE budget that fits on GPU for a given example.
#
# Dual-mode operation:
#   Search mode:  julia --project=. scripts/profile_budget.jl example=monod
#   Probe mode:   julia --project=. scripts/profile_budget.jl example=monod budget=2000000
#
# Search mode starts from ODE_BUDGET_TRAJ (defined in model.jl) and doubles/halves
# until the boundary is found. Each probe runs in a child process for OOM isolation.
# grad_accum always comes from GRAD_ACCUM_STEPS in model.jl.

using Printf

function _parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        startswith(a, prefix) && return split(a, "=", limit=2)[2]
    end
    return default
end

example = _parse_kwarg(ARGS, "example")
example === nothing && error("Required: example=<name> (e.g. example=monod)")
budget_str = _parse_kwarg(ARGS, "budget")

example_dir = joinpath(@__DIR__, "..", "examples", example)
@assert isdir(example_dir) "Unknown example: $example (no directory $example_dir)"

# Both modes need model.jl for constants
include(joinpath(example_dir, "model.jl"))

if budget_str !== nothing
    # ---- Probe mode: load Reactant, run one forward+backward pass ----
    budget = parse(Int, budget_str)
    include(joinpath(@__DIR__, "..", "src", "common.jl"))

    L, M, B = allocate_budget(budget)
    B_micro = B ÷ GRAD_ACCUM_STEPS
    n_denom = L + 1

    @printf("probe C=%d L=%d M=%d B=%d B_micro=%d\n", budget, L, M, B, B_micro)
    flush(stdout)

    if B_micro < 1
        println("FAIL: B_micro < 1")
        exit(1)
    end

    loss_fn = @isdefined(targeted_spce_loss_pk) ? targeted_spce_loss_pk : targeted_spce_loss
    batch_fn = @isdefined(prepare_batch_pk) ? prepare_batch_pk : _prepare_batch_default

    Reactant.set_default_backend("gpu")
    xdev = reactant_device()

    rng = Random.MersenneTwister(42)
    ps, st = Lux.setup(rng, policy)
    ps_ra = ps |> xdev
    st_ra = st |> xdev
    u0 = make_u0() |> xdev

    data = batch_fn(rng, n_denom, M, B_micro, u0, xdev)

    opt = Adam(1f-4)
    train_state = Lux.Training.TrainState(policy, ps_ra, st_ra, opt)

    t0 = time()
    _, loss_val, _, train_state_out = Lux.Training.single_train_step!(
        AutoEnzyme(), loss_fn, data, train_state
    )
    t_elapsed = time() - t0

    # Force copy-back to CPU to verify no deferred OOM
    loss_f64 = Float64(loss_val)
    ps_cpu = Lux.cpu_device()(train_state_out.parameters)
    ps_norm = sum(x -> sum(abs2, x), Lux.Functors.fleaves(ps_cpu))

    if loss_f64 == 0.0 || isnan(loss_f64) || isinf(loss_f64) || isnan(ps_norm)
        @printf("FAIL C=%d loss=%.6f ps_norm=%.6f (corrupt result)\n", budget, loss_f64, ps_norm)
        exit(1)
    end

    @printf("OK C=%d loss=%.6f ps_norm=%.6f time=%.1fs\n", budget, loss_f64, ps_norm, t_elapsed)

else
    # ---- Search mode: double/halve from ODE_BUDGET_TRAJ ----
    function run_search()
        results_dir = joinpath(example_dir, "results")
        mkpath(results_dir)
        log_path = joinpath(results_dir, "profile_budget.txt")
        write(log_path, "")

        julia_cmd = Base.julia_cmd()
        script = @__FILE__

        function probe(C::Int)
            cmd = `$julia_cmd --project=. $script example=$example budget=$C`
            @printf("  C=%d ... ", C)
            flush(stdout)
            try
                out = read(cmd, String)
                println("OK")
                open(log_path, "a") do io; println(io, out, "\n---"); end
                return true
            catch
                println("FAIL")
                open(log_path, "a") do io; println(io, "FAIL C=$C\n---"); end
                return false
            end
        end

        C = ODE_BUDGET_TRAJ
        println("=== Budget Profiling: $example ===")
        println("Starting from ODE_BUDGET_TRAJ = $C")
        println("GRAD_ACCUM_STEPS = $GRAD_ACCUM_STEPS")
        println()

        if probe(C)
            last_ok = C
            while true
                C *= 2
                if probe(C)
                    last_ok = C
                else
                    break
                end
            end
            best = last_ok
        else
            best = 0
            while C >= 1000
                C ÷= 2
                if probe(C)
                    best = C
                    break
                end
            end
            if best == 0
                println("\nNo budget fits. The example may be too large for this GPU.")
                exit(1)
            end
        end

        L, M, B = allocate_budget(best)
        B_micro = B ÷ GRAD_ACCUM_STEPS
        println()
        println("============================================================")
        println("RESULT: Max feasible budget for $example")
        println("============================================================")
        @printf("  C_max        = %d\n", best)
        @printf("  L            = %d\n", L)
        @printf("  M            = %d\n", M)
        @printf("  B            = %d\n", B)
        @printf("  B_micro      = %d\n", B_micro)
        @printf("  grad_accum   = %d\n", GRAD_ACCUM_STEPS)
        println("============================================================")
        println()
        println("Results log: $log_path")
    end

    run_search()
end
