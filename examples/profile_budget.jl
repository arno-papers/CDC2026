#!/usr/bin/env julia
# Probe a single ODE budget C: compile + run one forward+backward pass.
# Exits 0 on success, nonzero on OOM/crash.
#
# Called by profile_budget.sh — not meant to be run directly.
#
# Usage:
#   julia --project=. examples/profile_budget.jl example=monod budget=2000000 grad_accum=1

using Printf

function _parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        startswith(a, prefix) && return split(a, "=", limit=2)[2]
    end
    return default
end
_parse_int(args, key; default) = let v = _parse_kwarg(args, key); v === nothing ? default : parse(Int, v) end

example    = _parse_kwarg(ARGS, "example"; default="monod")
budget     = _parse_int(ARGS, "budget"; default=2_000_000)
grad_accum = _parse_int(ARGS, "grad_accum"; default=1)

example_dir = joinpath(@__DIR__, example)
@assert isdir(example_dir) "Unknown example: $example"

include(joinpath(example_dir, "model.jl"))
include(joinpath(@__DIR__, "..", "src", "common.jl"))

L, M, B = allocate_budget(budget)
B_micro = B ÷ grad_accum
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
