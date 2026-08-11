# ============================================================================
# Julia / Reactant benchmark: per-train-step time of the targeted sPCE loss.
#
# Times a single steady-state `single_train_step!` (loss + reverse-mode gradient
# via Enzyme, plus a µs-scale Adam update on the ~8.5k policy params). Data prep
# is OUTSIDE the timed region; the loss scalar is materialized INSIDE to force an
# XLA/GPU sync so we time the whole compiled program, not an async dispatch.
#
# Two integrators, selected with INTEG:
#   hand : production hand-written RK4 in a `@trace for` (src/common.jl)
#   sd   : SimpleDiffEq `SimpleRK4` `step!` driven inside a `@trace for`
# Same model / loss / policy / config for both — the delta is purely the
# integrator lowering.
#
# Env knobs:
#   INTEG=hand|sd   BACKEND=gpu|cpu   L=255  M=256  BMICRO=512
#   NWARM=3  NTIME=20  SEED=0
# Memory (share a contended GPU): set from the shell before launching, e.g.
#   XLA_REACTANT_GPU_PREALLOCATE=false XLA_REACTANT_GPU_MEM_FRACTION=0.5
# ============================================================================

include(joinpath(@__DIR__, "monod_model.jl"))
include(joinpath(@__DIR__, "common.jl"))

using SimpleDiffEq
using SciMLBase: init, step!
using Statistics

# --- SimpleDiffEq integrator: SimpleRK4 step! inside a (non-unrolling) @trace for
_sd_rhs(u, p, t) = dynamics(u, p[1], p[2])
function integrate_sd(u, θ, d, dt, n_substeps)
    dt_sub = dt / n_substeps
    prob  = ODEProblem(_sd_rhs, u, (0.0f0, dt), (θ, d))
    integ = init(prob, SimpleRK4(); dt = dt_sub)
    @trace mincut=true track_numbers=false for _ in 1:n_substeps
        step!(integ)
    end
    return integ.u
end

const INTEG = get(ENV, "INTEG", "hand")
INTEG == "sd" && (integrate(u, θ, d, dt, n) = integrate_sd(u, θ, d, dt, n))

backend = get(ENV, "BACKEND", "gpu")
L       = parse(Int, get(ENV, "L", "255"))
M       = parse(Int, get(ENV, "M", "256"))
B_micro = parse(Int, get(ENV, "BMICRO", "512"))
NWARM   = parse(Int, get(ENV, "NWARM", "3"))
NTIME   = parse(Int, get(ENV, "NTIME", "20"))
SEED    = parse(Int, get(ENV, "SEED", "0"))
Reactant.set_default_backend(backend)
say(a...) = (println(a...); flush(stdout))

rng = Random.MersenneTwister(SEED)
ps, st = Lux.setup(rng, policy)
n_params = Lux.parameterlength(policy)
xdev = reactant_device()
ps_ra = ps |> xdev; st_ra = st |> xdev
u0 = make_u0() |> xdev
ts = Lux.Training.TrainState(policy, ps_ra, st_ra, Optimisers.Adam(1f-3))
n_denom = L + 1
prep() = _prepare_batch_default(rng, n_denom, M, B_micro, u0, xdev)

say("="^68)
say("Julia/Reactant | integrator=$INTEG | backend=$backend")
say("L=$L  M=$M  B_micro=$B_micro  n_denom=$n_denom  N_STEPS=$N_STEPS  N_SUBSTEPS=$N_SUBSTEPS")
say("trajectories/step = ", B_micro * (L + 2 + M), "   policy params = ", n_params)
say("Reactant ", pkgversion(Reactant))
say("="^68)

# Wrap in a function so `ts` reassignment uses normal local scope.
function run_bench(ts, prep, NWARM, NTIME)
    local last_loss = 0.0f0
    t_compile = @elapsed for i in 1:NWARM
        data = prep()
        _, l, _, ts = Lux.Training.single_train_step!(AutoEnzyme(), targeted_spce_loss, data, ts)
        last_loss = Float32(l)   # force sync
    end
    say("warmup ($NWARM steps, incl compile): ", round(t_compile; digits=1), "s  (loss=", last_loss, ")")
    times = Float64[]
    for i in 1:NTIME
        data = prep()                       # data prep OUTSIDE the timed region
        t = @elapsed begin
            _, l, _, ts = Lux.Training.single_train_step!(AutoEnzyme(), targeted_spce_loss, data, ts)
            last_loss = Float32(l)          # materialize → force GPU sync
        end
        push!(times, t)
    end
    return times, last_loss, t_compile
end

times, loss, t_compile = run_bench(ts, prep, NWARM, NTIME)
med = median(times) * 1000
say("per-step over $NTIME timed iters:")
say("  median = ", round(med; digits=2), " ms")
say("  mean   = ", round(mean(times)*1000; digits=2), " ms")
say("  min    = ", round(minimum(times)*1000; digits=2), " ms")
say("  max    = ", round(maximum(times)*1000; digits=2), " ms")
# Machine-parseable summary line for run_all.sh to collect.
say("RESULT julia integ=$INTEG L=$L M=$M B=$B_micro median_ms=",
    round(med; digits=2), " compile_s=", round(t_compile; digits=1), " loss=", loss)
say("DONE")
