#=
Profiled version of DADS training with Reactant

Uses Reactant.with_profiler to generate XLA traces.

Default viewer: Perfetto (prints a ui.perfetto.dev link).
Optionally: TensorBoard/XProf (set `viewer=tensorboard`).

Run with Julia 1.11 (recommended for Enzyme/Reactant):
  julia +1.11 --project=. profiler.jl
=#

using Lux, Reactant, Random
using Optimisers
using Printf

# Include the main definitions
include("common.jl")

function parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        if startswith(a, prefix)
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_viewer(args; default::Symbol=:perfetto)
    v = parse_kwarg(args, "viewer"; default=nothing)
    v === nothing && return default
    v = lowercase(v)
    v in ("perfetto", "parfetto") && return :perfetto
    v in ("tensorboard", "tb", "xprof") && return :tensorboard
    error("Unknown viewer=$v (expected perfetto|tensorboard)")
end

function print_profile_howto(profile_dir::AbstractString; viewer::Symbol)
    println("\nProfiling complete. Traces saved to: $profile_dir")
    if viewer == :perfetto
        println("Perfetto link should have been printed (ui.perfetto.dev).")
        println("TensorBoard/XProf also works: tensorboard --logdir $profile_dir")
    else
        println("Open in TensorBoard/XProf: tensorboard --logdir $profile_dir")
        println("Perfetto also works (rerun with viewer=perfetto to print a link).")
    end
end

# ============================================================================
#  Profiled Training
# ============================================================================

function train_policy_profiled(
    model, ps, st, rng;
    xdev,
    n_iters=3,
    profile_dir="./traces",
    viewer::Symbol=:perfetto,
)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
    
    Reactant.with_profiler(profile_dir; create_perfetto_link=(viewer == :perfetto)) do
        for iteration in 1:n_iters
            n_denom = L_CONTRASTIVE + 1
            θ_full = sample_θ_full(rng, n_denom, GRAD_BATCH) |> xdev
            θ_N_numer = sample_θ_N(rng, M_NUISANCE, GRAD_BATCH) |> xdev
            u0 = zeros(Float32, 3, 1, 1)
            u0[1, 1, 1] = 3.0f0
            u0[2, 1, 1] = 0.25f0
            u0[3, 1, 1] = 7.0f0
            u0 = u0 |> xdev
            input_buffer = zeros(Float32, 2, N_STEPS, GRAD_BATCH) |> xdev
            observations = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
            designs = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
            ε = randn(rng, Float32, N_STEPS, GRAD_BATCH) |> xdev
            ll_denom = zeros(Float32, n_denom, GRAD_BATCH) |> xdev

            ll_numer = zeros(Float32, M_NUISANCE, GRAD_BATCH) |> xdev
            data = (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)

            _, loss, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(), targeted_spce_loss, data, train_state; return_gradients=Val(false)
            )
            
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, n_iters, loss)
        end
    end

    print_profile_howto(profile_dir; viewer=viewer)
    
    return train_state
end

function make_step_data(rng, xdev)
    n_denom = L_CONTRASTIVE + 1
    θ_full = sample_θ_full(rng, n_denom, GRAD_BATCH) |> xdev
    θ_N_numer = sample_θ_N(rng, M_NUISANCE, GRAD_BATCH) |> xdev
    u0 = zeros(Float32, 3, 1, 1)
    u0[1, 1, 1] = 3.0f0
    u0[2, 1, 1] = 0.25f0
    u0[3, 1, 1] = 7.0f0
    u0 = u0 |> xdev
    input_buffer = zeros(Float32, 2, N_STEPS, GRAD_BATCH) |> xdev
    observations = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
    designs = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
    ε = randn(rng, Float32, N_STEPS, GRAD_BATCH) |> xdev
    ll_denom = zeros(Float32, n_denom, GRAD_BATCH) |> xdev
    ll_numer = zeros(Float32, M_NUISANCE, GRAD_BATCH) |> xdev
    return (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)
end

# ============================================================================
#  Run profiled training
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== Profiled DADS Training ===\n")

    viewer = parse_viewer(ARGS; default=:perfetto)

    Reactant.set_default_backend("gpu")
    
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, policy)
    
    xdev = reactant_device()
    ps_ra = ps |> xdev
    st_ra = st |> xdev
    
    profile_dir = joinpath(tempdir(), "dads_training_trace")
    println("Profile output: $profile_dir\n")

    # Warm up outside the profiler so traces reflect steady-state execution.
    # This should compile XLA/Enzyme once, which is amortized during real training.
    println("Warming up (compilation)...")
    train_state = Lux.Training.TrainState(policy, ps_ra, st_ra, Adam(0.001f0))
    data = make_step_data(rng, xdev)
    Lux.Training.single_train_step!(
        AutoEnzyme(), targeted_spce_loss, data, train_state; return_gradients=Val(false)
    )
    println("Warmup done.")

    # Some backends recompile when profiling is enabled. Do a 1-step profiled run
    # into a throwaway directory to compile the instrumented executable.
    compile_profile_dir = joinpath(tempdir(), "dads_compile_trace")
    println("Warming up profiler instrumentation (discarding traces): $compile_profile_dir")
    Reactant.with_profiler(compile_profile_dir; create_perfetto_link=false) do
        train_state = Lux.Training.TrainState(policy, ps_ra, st_ra, Adam(0.001f0))
        data = make_step_data(rng, xdev)
        Lux.Training.single_train_step!(
            AutoEnzyme(), targeted_spce_loss, data, train_state; return_gradients=Val(false)
        )
    end

    println("Starting profiler...\n")

    train_state = train_policy_profiled(policy, ps_ra, st_ra, rng;
                                         xdev=xdev, n_iters=3, profile_dir=profile_dir, viewer=viewer)
end
