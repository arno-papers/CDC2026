# Find the maximum ODE budget that fits on GPU for a given example.
#
# Dual-mode operation:
#   Search mode:  julia --project=. scripts/profile_budget.jl example=monod [mode=train|eval]
#   Probe mode:   julia --project=. scripts/profile_budget.jl example=monod budget=2000000 [mode=train|eval]
#
# mode=train (default): Enzyme forward+backward pass; uses allocate_budget for L/M/B split.
# mode=eval:            Forward-only @jit; uses B=2 with large L=M for tight sPCE bound.
#
# Search mode starts from ODE_BUDGET_TRAJ (train) or 11_000 (eval) and doubles/halves
# until the boundary is found. Each probe runs in a child process for OOM isolation.

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
mode = _parse_kwarg(ARGS, "mode"; default="train")
@assert mode in ("train", "eval") "mode must be 'train' or 'eval', got: $mode"

example_dir = joinpath(@__DIR__, "..", "examples", example)
@assert isdir(example_dir) "Unknown example: $example (no directory $example_dir)"

# Both modes need model.jl for constants
include(joinpath(example_dir, "model.jl"))

if budget_str !== nothing
    # ---- Probe mode: load Reactant, run one forward+backward pass ----
    budget = parse(Int, budget_str)
    include(joinpath(@__DIR__, "..", "src", "common.jl"))

    if mode == "eval"
        # Eval allocation: small B, large L for tight sPCE bound
        B = 2
        n_per_ep = budget ÷ B
        L = (n_per_ep - 2) ÷ 2
        M = L
        B_micro = B
    else
        L, M, B_micro = allocate_budget(budget; B_multiplier=GRAD_ACCUM_STEPS)
        B = B_micro * GRAD_ACCUM_STEPS
    end
    n_denom = L + 1

    @printf("probe C=%d L=%d M=%d B=%d B_micro=%d mode=%s\n", budget, L, M, B, B_micro, mode)
    flush(stdout)

    if B_micro < 1
        println("FAIL: B_micro < 1")
        exit(1)
    end

    loss_fn = targeted_spce_loss
    batch_fn = _prepare_batch_default

    Reactant.set_default_backend("gpu")
    xdev = reactant_device()

    rng = Random.MersenneTwister(42)
    ps, st = Lux.setup(rng, policy)
    ps_ra = ps |> xdev
    st_ra = st |> xdev
    u0 = make_u0() |> xdev

    data = batch_fn(rng, n_denom, M, B_micro, u0, xdev)

    if mode == "eval"
        # Forward-only probe: @jit (no Enzyme backward tape)
        # @eval needed because @jit macro is only available after common.jl loads Reactant
        @eval _jit_forward(f, m, p, s, d) = Reactant.@jit f(m, p, s, d)
        t0 = time()
        loss_val, _, _ = Base.invokelatest(_jit_forward, loss_fn, policy, ps_ra, st_ra, data)
        t_elapsed = time() - t0

        loss_f64 = Float64(loss_val)
        if loss_f64 == 0.0 || isnan(loss_f64) || isinf(loss_f64)
            @printf("FAIL C=%d loss=%.6f (corrupt result, mode=eval)\n", budget, loss_f64)
            exit(1)
        end
        @printf("OK C=%d loss=%.6f time=%.1fs (mode=eval)\n", budget, loss_f64, t_elapsed)
    else
        # Training probe: Enzyme forward + backward
        opt = Adam(1f-4)
        train_state = Lux.Training.TrainState(policy, ps_ra, st_ra, opt)

        t0 = time()
        local train_state_out = train_state
        local loss_val = 0.0
        for _step in 1:GRAD_ACCUM_STEPS
            local data = batch_fn(rng, n_denom, M, B_micro, u0, xdev)
            _, loss_val, _, train_state_out = Lux.Training.single_train_step!(
                AutoEnzyme(), loss_fn, data, train_state_out
            )
        end
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
    end

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
            cmd = `$julia_cmd --project=. $script example=$example budget=$C mode=$mode`
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

        C = mode == "eval" ? 11_000 : ODE_BUDGET_TRAJ
        println("=== Budget Profiling: $example (mode=$mode) ===")
        println("Starting budget   = $C")
        println("GRAD_ACCUM_STEPS  = $GRAD_ACCUM_STEPS")
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
            first_fail = C
            # Phase 2: Binary search refinement (2 steps)
            for _ in 1:2
                mid = (last_ok + first_fail) ÷ 2
                mid == last_ok && break
                if probe(mid)
                    last_ok = mid
                else
                    first_fail = mid
                end
            end
            best = last_ok
        else
            first_fail = C
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
            # Phase 2: Binary search refinement (2 steps)
            last_ok = best
            for _ in 1:2
                mid = (last_ok + first_fail) ÷ 2
                mid == last_ok && break
                if probe(mid)
                    last_ok = mid
                else
                    first_fail = mid
                end
            end
            best = last_ok
        end

        if mode == "eval"
            B = 2
            n_per_ep = best ÷ B
            L = (n_per_ep - 2) ÷ 2
            M = L
            B_micro = B
        else
            L, M, B_micro = allocate_budget(best; B_multiplier=GRAD_ACCUM_STEPS)
            B = B_micro * GRAD_ACCUM_STEPS
        end
        println()
        println("============================================================")
        println("RESULT: Max feasible budget for $example (mode=$mode)")
        println("============================================================")
        @printf("  C_max        = %d\n", best)
        @printf("  L            = %d\n", L)
        @printf("  M            = %d\n", M)
        @printf("  B            = %d\n", B)
        @printf("  B_micro      = %d\n", B_micro)
        if mode == "train"
            @printf("  grad_accum   = %d\n", GRAD_ACCUM_STEPS)
        end
        println("============================================================")
        println()
        println("Results log: $log_path")
    end

    run_search()
end
