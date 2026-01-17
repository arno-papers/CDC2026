#=
Profiled version of DADS training with Reactant

Uses Reactant.with_profiler to generate XLA traces for analysis with xprof.
=#

using Lux, Reactant, Random
using Optimisers
using Printf
using LogExpFunctions: logsumexp

Reactant.set_default_backend("gpu")

# Include the main definitions
include("mainGPU.jl")

# ============================================================================
#  Profiled Training
# ============================================================================

function train_policy_profiled(model, ps, st, rng; n_iters=20, profile_dir="./traces")
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
    
    Reactant.with_profiler(profile_dir) do
        for iteration in 1:n_iters
            θ_full = sample_θ_full(rng, L_CONTRASTIVE + 1) |> xdev
            θ_N_numer = sample_θ_N(rng, M_NUISANCE) |> xdev
            u0 = Float32[3.0, 0.25, 7.0] |> xdev
            input_buffer = zeros(Float32, 2, N_STEPS, 1) |> xdev
            observations = zeros(Float32, N_STEPS) |> xdev
            designs = zeros(Float32, N_STEPS) |> xdev
            ε = randn(rng, Float32, N_STEPS) |> xdev
            ll_denom = zeros(Float32, L_CONTRASTIVE + 1) |> xdev

            data = (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom)

            _, loss, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(), targeted_spce_loss, data, train_state; return_gradients=Val(false)
            )
            
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, n_iters, loss)
        end
    end
    
    println("\n✓ Profiling complete! Traces saved to: $profile_dir")
    println("Analyze with xprof: https://github.com/openxla/xprof")
    
    return train_state
end

# ============================================================================
#  Run profiled training
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== Profiled DADS Training ===\n")
    
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, policy)
    
    xdev = reactant_device()
    ps_ra = ps |> xdev
    st_ra = st |> xdev
    
    profile_dir = joinpath(tempdir(), "dads_training_trace")
    println("Profile output: $profile_dir\n")
    
    train_state = train_policy_profiled(policy, ps_ra, st_ra, rng; 
                                         n_iters=20, profile_dir=profile_dir)
end
