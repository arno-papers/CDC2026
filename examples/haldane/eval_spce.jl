# CPU-based targeted sPCE evaluation for Haldane.
# Compares two checkpoints with identical L, M, seeds.
#
# Usage:
#   julia --project=. examples/haldane/eval_spce.jl checkpoint1.jls checkpoint2.jls [L=500] [M=500] [n_trials=500]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU forward simulation (returns substrate trajectory)
# ============================================================================

function substrate_trajectory_cpu(theta_dyn::Vector{Float32}, design::Vector{Float32};
        Cx0::Float32=0.25f0, n_substeps::Int=N_SUBSTEPS)
    u = reshape(Float32[3.0f0, Cx0, 7.0f0], 3, 1)
    theta_mat = reshape(theta_dyn, N_PARAMS_DYN, 1)
    cs = Vector{Float32}(undef, N_STEPS)
    @inbounds for step in 1:N_STEPS
        u = integrate_cpu(u, theta_mat, design[step], DT, n_substeps)
        cs[step] = u[1, 1]
    end
    return cs
end

# ============================================================================
#  Log-likelihood (Gaussian observation model)
# ============================================================================

function log_likelihood_cpu(observations::Vector{Float32}, theta_dyn::Vector{Float32},
        sigma::Float32, design::Vector{Float32}; Cx0::Float32=0.25f0)
    cs = substrate_trajectory_cpu(theta_dyn, design; Cx0=Cx0)
    ll = 0.0
    σ² = Float64(sigma)^2
    @inbounds for k in 1:N_STEPS
        r = Float64(observations[k] - cs[k])
        ll -= 0.5 * (r^2 / σ² + log(σ²))
    end
    return ll
end

# ============================================================================
#  Targeted sPCE for one trial
# ============================================================================

function logsumexp(x::Vector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function eval_spce_trial(model, ps_cpu, st_cpu, rng; L::Int, M::Int)
    # 1. Sample true parameters
    θ_true = sample_θ_full(rng, 1)  # (5, 1)
    μ_max_true = θ_true[1, 1]
    K_s_true   = θ_true[2, 1]
    α_true     = θ_true[3, 1]
    σ_true     = θ_true[4, 1]
    Cx0_true   = θ_true[5, 1]
    theta_dyn_true = Float32[μ_max_true, K_s_true, α_true]

    # 2. Adaptive rollout → designs + observations
    u = reshape(Float32[3.0f0, Cx0_true, 7.0f0], 3, 1)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = zeros(Float32, N_STEPS)
    observations = zeros(Float32, N_STEPS)
    st_local = st_cpu
    theta_mat = reshape(theta_dyn_true, N_PARAMS_DYN, 1)
    @inbounds for step in 1:N_STEPS
        action, st_local = model(input_buffer, ps_cpu, st_local)
        d = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = d
        u = integrate_cpu(u, theta_mat, d, DT, N_SUBSTEPS)
        y_obs = u[1, 1] + σ_true * randn(rng, Float32)
        observations[step] = y_obs
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = d
    end

    # 3. Denominator: L+1 joint samples (all params vary)
    #    First sample = true params, rest are contrastive
    ll_denom = Vector{Float64}(undef, L + 1)
    ll_denom[1] = log_likelihood_cpu(observations, theta_dyn_true, σ_true, designs; Cx0=Cx0_true)
    θ_denom = sample_θ_full(rng, L)  # (5, L)
    for ℓ in 1:L
        θ_d = Float32[θ_denom[1, ℓ], θ_denom[2, ℓ], θ_denom[3, ℓ]]
        ll_denom[ℓ + 1] = log_likelihood_cpu(observations, θ_d, θ_denom[4, ℓ], designs; Cx0=θ_denom[5, ℓ])
    end

    # 4. Numerator: M samples with TRUE α, resample nuisance (μ_max, K_s, σ, Cx0)
    ll_numer = Vector{Float64}(undef, M)
    for m in 1:M
        μ_m = μ_max_lo + (μ_max_hi - μ_max_lo) * rand(rng, Float32)
        K_m = K_s_lo + (K_s_hi - K_s_lo) * rand(rng, Float32)
        σ_m = σ_lo + (σ_hi - σ_lo) * rand(rng, Float32)
        Cx0_m = Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng, Float32)
        θ_n = Float32[μ_m, K_m, α_true]  # α fixed!
        ll_numer[m] = log_likelihood_cpu(observations, θ_n, σ_m, designs; Cx0=Cx0_m)
    end

    # 5. sPCE = log E_numer[p(y|θ)] - log E_denom[p(y|θ)]
    return logsumexp(ll_numer) - log(Float64(M)) - (logsumexp(ll_denom) - log(Float64(L + 1)))
end

# ============================================================================
#  Evaluate a single checkpoint
# ============================================================================

function evaluate_checkpoint(ckpt_file::String; L::Int, M::Int, n_trials::Int, seed::Int)
    ckpt = deserialize(ckpt_file)
    ps_cpu = ckpt["parameters"]
    st_cpu = ckpt["states"]

    rng = Random.MersenneTwister(seed)
    scores = Vector{Float64}(undef, n_trials)

    for i in 1:n_trials
        scores[i] = eval_spce_trial(policy, ps_cpu, st_cpu, rng; L=L, M=M)
        if i % 100 == 0 || i == 1
            @printf("  trial %d/%d  running mean = %.4f\n", i, n_trials, mean(scores[1:i]))
            flush(stdout)
        end
    end
    return scores
end

# ============================================================================
#  Main
# ============================================================================

if length(ARGS) < 2
    println("Usage: julia --project=. eval_spce.jl checkpoint1.jls checkpoint2.jls [L=500] [M=500] [n_trials=500]")
    exit(1)
end

ckpt1 = ARGS[1]
ckpt2 = ARGS[2]
L = 500
M = 500
n_trials = 500
seed = 42

for arg in ARGS[3:end]
    key, val = split(arg, '='; limit=2)
    if     key == "L";        L        = parse(Int, val)
    elseif key == "M";        M        = parse(Int, val)
    elseif key == "n_trials"; n_trials = parse(Int, val)
    elseif key == "seed";     seed     = parse(Int, val)
    end
end

@assert isfile(ckpt1) "Checkpoint not found: $ckpt1"
@assert isfile(ckpt2) "Checkpoint not found: $ckpt2"

println("=== Haldane Targeted sPCE Comparison ===")
println("L = $L, M = $M, n_trials = $n_trials, seed = $seed")
println()

println("--- Checkpoint 1: $ckpt1 ---")
t1 = time()
scores1 = evaluate_checkpoint(ckpt1; L=L, M=M, n_trials=n_trials, seed=seed)
t1 = time() - t1

println("\n--- Checkpoint 2: $ckpt2 ---")
t2 = time()
scores2 = evaluate_checkpoint(ckpt2; L=L, M=M, n_trials=n_trials, seed=seed)
t2 = time() - t2

# Results
m1, s1 = mean(scores1), std(scores1)
m2, s2 = mean(scores2), std(scores2)
sem1, sem2 = s1 / sqrt(n_trials), s2 / sqrt(n_trials)

println("\n=== Results (targeted sPCE, higher = more informative) ===\n")
@printf("  ckpt1: mean = %8.4f ± %.4f (SEM)  std = %.4f  [%.1fs]\n", m1, sem1, s1, t1)
@printf("  ckpt2: mean = %8.4f ± %.4f (SEM)  std = %.4f  [%.1fs]\n", m2, sem2, s2, t2)

# Paired t-test
delta = scores1 .- scores2
t_stat = mean(delta) / (std(delta) / sqrt(n_trials))
@printf("\n  Paired: ckpt1 - ckpt2 = %+.4f ± %.4f (SEM)  t = %.2f\n", mean(delta), std(delta)/sqrt(n_trials), t_stat)
println()
if abs(t_stat) > 1.96
    winner = t_stat > 0 ? "ckpt1" : "ckpt2"
    println("  => $winner is significantly better (p < 0.05)")
else
    println("  => No significant difference (|t| < 1.96)")
end
