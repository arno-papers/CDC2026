#=
Profiled version of DADS training with Reactant

Uses Reactant.with_profiler to generate XLA traces.
The trace shows peak memory allocation, kernel timings, and memory timeline —
useful for checking if training fits on a smaller GPU (e.g. 8GB VRAM).

Default viewer: Perfetto (prints a ui.perfetto.dev link).
Optionally: TensorBoard/XProf (set `viewer=tensorboard`).

Run:
  julia --project=. profiler.jl
  julia --project=. profiler.jl grad_accum=20 n_iters=3 viewer=perfetto
=#

using Lux, Reactant, Random
using Optimisers
using Printf

# Include the main definitions
include("common.jl")

# ============================================================================
#  CLI Parsing
# ============================================================================

function parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        if startswith(a, prefix)
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_int(args, key; default::Int)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Int, v)
end

function parse_viewer(args; default::Symbol=:perfetto)
    v = parse_kwarg(args, "viewer"; default=nothing)
    v === nothing && return default
    v = lowercase(v)
    v in ("perfetto", "parfetto") && return :perfetto
    v in ("tensorboard", "tb", "xprof") && return :tensorboard
    error("Unknown viewer=$v (expected perfetto|tensorboard)")
end

# ============================================================================
#  Profiled Training
# ============================================================================

function train_policy_profiled(
    model, ps, st, rng;
    xdev,
    n_iters=5,
    grad_accum=GRAD_ACCUM_STEPS,
    profile_dir="./traces",
    viewer::Symbol=:perfetto,
)
    B_micro = GRAD_BATCH ÷ grad_accum
    n_denom = L_CONTRASTIVE + 1

    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))

    # Pre-allocate u0 once (same across all micro-batches)
    u0 = zeros(Float32, 3, 1, 1)
    u0[1, 1, 1] = 3.0f0
    u0[2, 1, 1] = 0.25f0
    u0[3, 1, 1] = 7.0f0
    u0 = u0 |> xdev

    println("Profiling config:")
    println("  L = $L_CONTRASTIVE, M = $M_NUISANCE")
    println("  B = $GRAD_BATCH total ($(grad_accum)×$(B_micro) micro)")
    println("  n_iters = $n_iters, viewer = $viewer")
    println("  profile_dir = $profile_dir\n")

    Reactant.with_profiler(profile_dir; create_perfetto_link=(viewer == :perfetto)) do
        for iteration in 1:n_iters
            total_loss = 0.0f0

            for _k in 1:grad_accum
                θ_full = sample_θ_full(rng, n_denom, B_micro) |> xdev
                θ_N_numer = sample_θ_N(rng, M_NUISANCE, B_micro) |> xdev

                input_buffer = zeros(Float32, 2, N_STEPS, B_micro) |> xdev
                observations = zeros(Float32, N_STEPS, B_micro) |> xdev
                designs = zeros(Float32, N_STEPS, B_micro) |> xdev
                ε = randn(rng, Float32, N_STEPS, B_micro) |> xdev
                ll_denom = zeros(Float32, n_denom, B_micro) |> xdev
                ll_numer = zeros(Float32, M_NUISANCE, B_micro) |> xdev

                data = (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer)

                _, loss_k, _, train_state = Lux.Training.single_train_step!(
                    AutoEnzyme(), targeted_spce_loss, data, train_state;
                    return_gradients=Val(false),
                )
                total_loss += loss_k
            end

            avg_loss = total_loss / Float32(grad_accum)
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, n_iters, avg_loss)
        end
    end
    return train_state
end

# ============================================================================
#  Run profiled training (script)
# ============================================================================

println("\n=== Profiled DADS Training ===\n")

viewer = parse_viewer(ARGS; default=:perfetto)
n_iters = parse_int(ARGS, "n_iters"; default=5)
grad_accum = parse_int(ARGS, "grad_accum"; default=GRAD_ACCUM_STEPS)

Reactant.set_default_backend("gpu")

rng = Random.default_rng()
ps, st = Lux.setup(rng, policy)

xdev = reactant_device()
ps_ra = ps |> xdev
st_ra = st |> xdev

profile_dir = joinpath(tempdir(), "dads_training_trace")
println("Profile output: $profile_dir\n")

println("Starting profiler...\n")

train_state = train_policy_profiled(
    policy,
    ps_ra,
    st_ra,
    rng;
    xdev=xdev,
    n_iters=n_iters,
    grad_accum=grad_accum,
    profile_dir=profile_dir,
    viewer=viewer,
)
