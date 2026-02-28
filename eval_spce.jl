#!/usr/bin/env julia
# Evaluate adaptive vs static designs on the targeted sPCE criterion (CPU).
#
# Usage:
#   julia --project eval_spce.jl [checkpoint=...] [n_trials=500] [L=1000] [M=128]

include("common_core.jl")
include("compare_static_bim.jl")  # for load_checkpoint_cpu, rollout, parse helpers
include("plot_spce_comparison.jl") # histogram plots + summary stats

using Dates
using Distributions: TDist, ccdf
using Printf
using Random
using Statistics
using Serialization
using Plots

# ============================================================================
#  Targeted sPCE score for a single trial (CPU, scalar)
#
#  Given a design ξ (vector of N_STEPS feed rates), true params, and
#  observations y, compute:
#     sPCE = log[ (1/M) Σ_m p(y | θ_T, σ_m, Cx0_m, ξ) ]
#          - log[ (1/(L+1)) Σ_ℓ p(y | θ_ℓ, ξ) ]
#
#  where p(y|θ,ξ) = Π_k N(y_k; C_s(k;θ,ξ), σ²)
# ============================================================================

function log_likelihood(observations::Vector{Float64}, theta_T, sigma, Cx0,
                         design::AbstractVector; n_substeps::Int=N_SUBSTEPS)
    # Simulate trajectory
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    # Accumulate log-likelihood
    σ² = Float64(sigma)^2
    ll = 0.0
    for k in 1:N_STEPS
        residual = observations[k] - cs[k]
        ll -= 0.5 * (residual^2 / σ² + log(σ²))
    end
    return ll
end

function logsumexp(x::Vector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function spce_score(observations::Vector{Float64}, design::AbstractVector,
                     theta_T_true, sigma_true,
                     denom_samples, numer_sigma, numer_Cx0;
                     n_substeps::Int=N_SUBSTEPS)
    L_plus_1 = length(denom_samples)
    M = length(numer_sigma)

    # Denominator: (L+1) full-parameter contrastive samples
    ll_denom = Vector{Float64}(undef, L_plus_1)
    for ℓ in 1:L_plus_1
        θT_ℓ, σ_ℓ, Cx0_ℓ = denom_samples[ℓ]
        ll_denom[ℓ] = log_likelihood(observations, θT_ℓ, σ_ℓ, Cx0_ℓ, design;
                                      n_substeps=n_substeps)
    end
    log_denominator = logsumexp(ll_denom) - log(Float64(L_plus_1))

    # Numerator: M joint (σ, Cx0) samples at true θ_T
    ll_numer = Vector{Float64}(undef, M)
    for m in 1:M
        ll_numer[m] = log_likelihood(observations, theta_T_true, numer_sigma[m],
                                      numer_Cx0[m], design; n_substeps=n_substeps)
    end
    log_numerator = logsumexp(ll_numer) - log(Float64(M))

    return log_numerator - log_denominator
end

# ============================================================================
#  Generate observations from a rollout (true trajectory + noise)
# ============================================================================

function generate_observations(rng, theta_T, sigma, Cx0, design;
                                n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    observations = [cs[k] + Float64(sigma) * randn(rng) for k in 1:N_STEPS]
    return observations
end

# ============================================================================
#  Main
# ============================================================================

checkpoint   = parse_kwarg(ARGS, "checkpoint"; default="results/joint-nuisance-initfix")
spce_design_path = parse_kwarg(ARGS, "spce_design"; default=nothing)
n_trials     = parse_int(ARGS, "n_trials"; default=500)
n_substeps   = parse_int(ARGS, "n_substeps"; default=N_SUBSTEPS)
L            = parse_int(ARGS, "L"; default=L_CONTRASTIVE)
M            = parse_int(ARGS, "M"; default=M_NUISANCE)
seed         = parse_int(ARGS, "seed"; default=0)

output_dir = joinpath(checkpoint, "spce_evaluation")
mkpath(output_dir)

# Load model and static designs
ps_cpu, st_cpu, _ = load_checkpoint_cpu(checkpoint)

cheating_comp = deserialize(joinpath(checkpoint, "bim_comparison_cheating", "bim_comparison.jls"))
static_cheating = cheating_comp["static_design"]

standard_comp = deserialize(joinpath(checkpoint, "bim_comparison", "bim_comparison.jls"))
static_standard = standard_comp["static_design"]

# Optionally load sPCE-optimized static design
has_spce_opt = false
local static_spce_opt
if spce_design_path !== nothing
    spce_file = spce_design_path
else
    # Default: look for it alongside the checkpoint
    spce_file = joinpath(checkpoint, "spce_static_opt", "spce_static_design.jls")
end
if isfile(spce_file)
    spce_data = deserialize(spce_file)
    static_spce_opt = spce_data["design"]
    has_spce_opt = true
    println("Loaded sPCE-optimized design from: $spce_file")
else
    println("No sPCE-optimized design found (looked at: $spce_file)")
    static_spce_opt = zeros(Float32, N_STEPS)  # unused placeholder
end

println("=== Targeted sPCE Evaluation ===")
println("n_trials = $n_trials, L = $L, M = $M, n_substeps = $n_substeps")
println()
println("Static (standard BIM): [", join(round.(static_standard; digits=3), ", "), "]")
println("Static (cheating BIM): [", join(round.(static_cheating; digits=3), ", "), "]")
if has_spce_opt
    println("Static (sPCE-opt):     [", join(round.(static_spce_opt; digits=3), ", "), "]")
end
println()
flush(stdout)

rng = MersenneTwister(seed)
t_start = time()

adaptive_scores     = Vector{Float64}(undef, n_trials)
static_std_scores   = Vector{Float64}(undef, n_trials)
static_cheat_scores = Vector{Float64}(undef, n_trials)
static_spce_scores  = has_spce_opt ? Vector{Float64}(undef, n_trials) : Float64[]
all_adaptive_designs = Matrix{Float32}(undef, N_STEPS, n_trials)

for i in 1:n_trials
    # Draw true parameters
    θ_draw = sample_θ_full(rng, 1)
    θT = Float32[θ_draw[1, 1], θ_draw[2, 1]]
    σ  = Float32(θ_draw[3, 1])
    Cx0 = Float32(θ_draw[4, 1])

    # Draw contrastive samples (shared across all designs for fair comparison)
    denom_samples = Vector{Tuple{Vector{Float32}, Float32, Float32}}(undef, L + 1)
    # First contrastive sample = true parameters
    denom_samples[1] = (copy(θT), σ, Cx0)
    for ℓ in 2:(L + 1)
        θ_c = sample_θ_full(rng, 1)
        denom_samples[ℓ] = (Float32[θ_c[1, 1], θ_c[2, 1]], Float32(θ_c[3, 1]), Float32(θ_c[4, 1]))
    end

    numer_σ = [σ_lo + (σ_hi - σ_lo) * rand(rng, Float32) for _ in 1:M]
    numer_Cx0 = [Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng, Float32) for _ in 1:M]

    # Adaptive rollout — generates observations from the adaptive design
    rng_rollout = MersenneTwister(seed + i)  # separate rng for rollout noise
    d_adapt = rollout_adaptive_design_cpu(policy, ps_cpu, st_cpu, rng_rollout, θT, σ;
                                           Cx0=Cx0, n_substeps=n_substeps)
    all_adaptive_designs[:, i] .= d_adapt
    obs_adapt = generate_observations(MersenneTwister(seed + i), θT, σ, Cx0, d_adapt;
                                       n_substeps=n_substeps)

    # Static designs — generate their own observations with same true params
    obs_std   = generate_observations(MersenneTwister(seed + n_trials + i), θT, σ, Cx0,
                                       static_standard; n_substeps=n_substeps)
    obs_cheat = generate_observations(MersenneTwister(seed + 2*n_trials + i), θT, σ, Cx0,
                                       static_cheating; n_substeps=n_substeps)

    # Score each on sPCE
    adaptive_scores[i] = spce_score(obs_adapt, d_adapt, θT, σ,
                                     denom_samples, numer_σ, numer_Cx0;
                                     n_substeps=n_substeps)
    static_std_scores[i] = spce_score(obs_std, static_standard, θT, σ,
                                       denom_samples, numer_σ, numer_Cx0;
                                       n_substeps=n_substeps)
    static_cheat_scores[i] = spce_score(obs_cheat, static_cheating, θT, σ,
                                         denom_samples, numer_σ, numer_Cx0;
                                         n_substeps=n_substeps)

    # sPCE-optimized static design
    if has_spce_opt
        obs_spce = generate_observations(MersenneTwister(seed + 3*n_trials + i), θT, σ, Cx0,
                                          static_spce_opt; n_substeps=n_substeps)
        static_spce_scores[i] = spce_score(obs_spce, static_spce_opt, θT, σ,
                                            denom_samples, numer_σ, numer_Cx0;
                                            n_substeps=n_substeps)
    end

    if i % 50 == 0
        @printf("  trial %d/%d\n", i, n_trials)
        flush(stdout)
    end
end

# ---- Mean adaptive as static baseline ----
avg_adaptive = Float32.(vec(mean(all_adaptive_designs; dims=2)))
println("\nMean adaptive design: [", join(round.(avg_adaptive; digits=3), ", "), "]")
println("Evaluating mean adaptive design as static baseline...")
flush(stdout)

static_avg_scores = Vector{Float64}(undef, n_trials)
rng_avg = MersenneTwister(seed + 999)
for i in 1:n_trials
    θ_draw = sample_θ_full(rng_avg, 1)
    θT = Float32[θ_draw[1, 1], θ_draw[2, 1]]
    σ  = Float32(θ_draw[3, 1])
    Cx0 = Float32(θ_draw[4, 1])

    denom_samples = Vector{Tuple{Vector{Float32}, Float32, Float32}}(undef, L + 1)
    denom_samples[1] = (copy(θT), σ, Cx0)
    for ℓ in 2:(L + 1)
        θ_c = sample_θ_full(rng_avg, 1)
        denom_samples[ℓ] = (Float32[θ_c[1, 1], θ_c[2, 1]], Float32(θ_c[3, 1]), Float32(θ_c[4, 1]))
    end
    numer_σ = [σ_lo + (σ_hi - σ_lo) * rand(rng_avg, Float32) for _ in 1:M]
    numer_Cx0 = [Cx0_lo + (Cx0_hi - Cx0_lo) * rand(rng_avg, Float32) for _ in 1:M]

    obs_avg = generate_observations(MersenneTwister(seed + 4*n_trials + i), θT, σ, Cx0,
                                     avg_adaptive; n_substeps=n_substeps)
    static_avg_scores[i] = spce_score(obs_avg, avg_adaptive, θT, σ,
                                       denom_samples, numer_σ, numer_Cx0;
                                       n_substeps=n_substeps)
    if i % 50 == 0
        @printf("  trial %d/%d\n", i, n_trials)
        flush(stdout)
    end
end

t_total = time() - t_start
@printf("\nTotal evaluation time: %.1fs\n", t_total)

# ---- Save serialized scores ----
scores_dict = Dict{String, Any}(
    "adaptive_scores"      => adaptive_scores,
    "static_avg_scores"    => static_avg_scores,
    "static_std_scores"    => static_std_scores,
    "static_cheat_scores"  => static_cheat_scores,
    "adaptive_designs"     => all_adaptive_designs,
    "n_trials"             => n_trials,
    "L"                    => L,
    "M"                    => M,
    "B"                    => 1,
    "n_substeps"           => n_substeps,
    "seed"                 => seed,
    "wall_time_s"          => t_total,
)
if has_spce_opt
    scores_dict["static_spce_scores"] = static_spce_scores
end
serialize(joinpath(output_dir, "spce_scores.jls"), scores_dict)

# ---- Summary stats (standardized) ----
score_designs = extract_designs(scores_dict)
summary = compute_summary_stats(score_designs)
println()
println(summary)
flush(stdout)

# ---- Histogram plot ----
plot_spce_histograms(score_designs;
    output_path = joinpath(output_dir, "plot_spce_histograms.png"),
    title_suffix = " (L=$L, M=$M, $n_trials trials)")

# ---- Design trajectory plot ----
static_designs_vec = Pair{String, Vector{Float32}}[
    "static_avg"   => avg_adaptive,
    "static_std"   => Float32.(static_standard),
    "static_cheat" => Float32.(static_cheating),
]
if has_spce_opt
    push!(static_designs_vec, "static_spce" => Float32.(static_spce_opt))
end
plot_design_trajectories(all_adaptive_designs, static_designs_vec;
    output_path = joinpath(output_dir, "plot_design_trajectories.png"))

# ---- Save summary text ----
open(joinpath(output_dir, "spce_summary.txt"), "w") do io
    println(io, "# Targeted sPCE Evaluation")
    println(io, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io)
    println(io, "n_trials = $n_trials")
    println(io, "L = $L")
    println(io, "M = $M")
    println(io, "n_substeps = $n_substeps")
    println(io, "seed = $seed")
    @printf(io, "wall_time_s = %.1f\n", t_total)
    println(io)
    println(io, "static_avg_design   = [", join(round.(avg_adaptive; digits=4), ", "), "]")
    println(io, "static_std_design   = [", join(round.(static_standard; digits=4), ", "), "]")
    println(io, "static_cheat_design = [", join(round.(static_cheating; digits=4), ", "), "]")
    if has_spce_opt
        println(io, "static_spce_design  = [", join(round.(static_spce_opt; digits=4), ", "), "]")
    end
    println(io)
    println(io, summary)
end
println("Saved: $(joinpath(output_dir, "spce_summary.txt"))")
println("Saved: $(joinpath(output_dir, "spce_scores.jls"))")
println("\nDone. Outputs in: $output_dir")
