#!/usr/bin/env julia
# GPU-accelerated sPCE evaluation using Reactant (@jit, forward-only).
#
# Usage:
#   julia --project=. examples/monod/eval_spce.jl [n_trials=1000] [L=5000] [M=5000] [B=32]

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

using Dates
using Printf
using Random
using Serialization
using Statistics

# ============================================================================
#  CPU helpers
# ============================================================================

function log_likelihood(observations::Vector{Float64}, theta_T, sigma, Cx0,
                         design::AbstractVector; n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    σ² = Float64(sigma)^2
    ll = 0.0
    for k in 1:N_STEPS
        residual = observations[k] - cs[k]
        ll -= 0.5 * (residual^2 / σ² + log(σ²))
    end
    return ll
end

function _logsumexp(x::Vector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

function spce_score(observations::Vector{Float64}, design::AbstractVector,
                     theta_T_true, sigma_true,
                     denom_samples, numer_sigma, numer_Cx0;
                     n_substeps::Int=N_SUBSTEPS)
    L_plus_1 = length(denom_samples)
    M = length(numer_sigma)

    ll_denom = Vector{Float64}(undef, L_plus_1)
    for ℓ in 1:L_plus_1
        θT_ℓ, σ_ℓ, Cx0_ℓ = denom_samples[ℓ]
        ll_denom[ℓ] = log_likelihood(observations, θT_ℓ, σ_ℓ, Cx0_ℓ, design;
                                      n_substeps=n_substeps)
    end
    log_denominator = _logsumexp(ll_denom) - log(Float64(L_plus_1))

    ll_numer = Vector{Float64}(undef, M)
    for m in 1:M
        ll_numer[m] = log_likelihood(observations, theta_T_true, numer_sigma[m],
                                      numer_Cx0[m], design; n_substeps=n_substeps)
    end
    log_numerator = _logsumexp(ll_numer) - log(Float64(M))

    return log_numerator - log_denominator
end

function generate_observations(rng, theta_T, sigma, Cx0, design;
                                n_substeps::Int=N_SUBSTEPS)
    params = Float64[Float64(theta_T[1]), Float64(theta_T[2]), Float64(Cx0)]
    cs = substrate_trajectory_diff(params, Float64.(design); n_substeps=n_substeps)
    observations = [cs[k] + Float64(sigma) * randn(rng) for k in 1:N_STEPS]
    return observations
end

# ============================================================================
#  GPU kernels
# ============================================================================

function adaptive_spce_eval(model, ps, st, data)
    θ_full, θ_obs_numer, u0, input_buffer, observations, designs, ε, ll_denom, ll_numer, n_substeps_val = data

    B = size(ε, 3)

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_dyn_true = θ_full[1:N_PARAMS_DYN, 1:1, :]
    θ_obs_true_3d = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1:1, :]
    θ_obs_true = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1, :]

    u = make_initial_state(u0, θ_dyn_true, θ_obs_true_3d, B)

    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        d = action
        designs[step, :] .= d[1, :]

        u = integrate(u, θ_dyn_true, d, DT, n_substeps_val)

        y_noisy = observe_noisy(u, θ_obs_true, ε, step)

        observations[step, :] .= y_noisy
        input_buffer[1, step, :] .= y_noisy
        input_buffer[2, step, :] .= d[1, :]
    end

    n_denom = size(θ_full, 2)
    θ_dyn_denom = θ_full[1:N_PARAMS_DYN, :, :]
    θ_obs_denom = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, :, :]

    u_denom = make_initial_state(u0, θ_dyn_denom, θ_obs_denom, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_denom, observations[step:step, :], u_denom, θ_obs_denom)
    end

    M_N = size(ll_numer, 1)

    u_numer = make_initial_state(u0, θ_dyn_true, θ_obs_numer, B)

    for step in 1:N_STEPS
        d_step = designs[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_true, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_numer, observations[step:step, :], u_numer, θ_obs_numer)
    end

    ll_max_num = maximum(ll_numer; dims=1)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))
    log_numerator = lse_num .- log(Float32(M_N))

    ll_max_den = maximum(ll_denom; dims=1)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))
    log_denominator = lse_den .- log(Float32(n_denom))

    spce_per_episode = log_numerator .- log_denominator
    return spce_per_episode, st, (;)
end

function static_spce_eval(model, ps, st, data)
    θ_full, θ_obs_numer, u0, observations, ε, ll_denom, ll_numer, n_substeps_val = data

    B = size(ε, 3)
    design = ps.layer_1.weight

    ll_denom .= 0.0f0
    ll_numer .= 0.0f0

    θ_dyn_true = θ_full[1:N_PARAMS_DYN, 1:1, :]
    θ_obs_true_3d = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1:1, :]
    θ_obs_true = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, 1, :]

    u = make_initial_state(u0, θ_dyn_true, θ_obs_true_3d, B)

    for step in 1:N_STEPS
        d_step = design[step:step, :]
        u = integrate(u, θ_dyn_true, d_step, DT, n_substeps_val)
        y_noisy = observe_noisy(u, θ_obs_true, ε, step)
        observations[step, :] .= y_noisy
    end

    n_denom = size(θ_full, 2)
    θ_dyn_denom = θ_full[1:N_PARAMS_DYN, :, :]
    θ_obs_denom = θ_full[N_PARAMS_DYN+1:N_PARAMS_DYN+N_PARAMS_OBS, :, :]

    u_denom = make_initial_state(u0, θ_dyn_denom, θ_obs_denom, B)

    for step in 1:N_STEPS
        d_step = design[step:step, :]
        u_denom = integrate(u_denom, θ_dyn_denom, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_denom, observations[step:step, :], u_denom, θ_obs_denom)
    end

    M_N = size(ll_numer, 1)

    u_numer = make_initial_state(u0, θ_dyn_true, θ_obs_numer, B)

    for step in 1:N_STEPS
        d_step = design[step:step, :]
        u_numer = integrate(u_numer, θ_dyn_true, d_step, DT, n_substeps_val)
        log_likelihood_step!(ll_numer, observations[step:step, :], u_numer, θ_obs_numer)
    end

    ll_max_num = maximum(ll_numer; dims=1)
    lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims=1))
    log_numerator = lse_num .- log(Float32(M_N))

    ll_max_den = maximum(ll_denom; dims=1)
    lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims=1))
    log_denominator = lse_den .- log(Float32(n_denom))

    spce_per_episode = log_numerator .- log_denominator
    return spce_per_episode, st, (;)
end

# ============================================================================
#  Main
# ============================================================================

checkpoint       = joinpath(@__DIR__, "results")
n_trials         = 1000
n_substeps       = N_SUBSTEPS
seed             = 0
Random.seed!(seed)

# Paper-quality defaults: 5× training L,M to reduce NMC bias to O(1e-4).
# The paired t-test cancels common bias, but absolute sPCE values benefit
# from larger samples.  B is pure parallelism (no statistical effect).
L                = 5000
M                = 5000
B                = 32

# Parse CLI args (key=value), e.g.: julia ... eval_spce.jl n_trials=500 L=1000
for arg in ARGS
    key, val = split(arg, '='; limit=2)
    if     key == "n_trials";  global n_trials  = parse(Int, val)
    elseif key == "L";        global L         = parse(Int, val)
    elseif key == "M";        global M         = parse(Int, val)
    elseif key == "B";        global B         = parse(Int, val)
    elseif key == "seed";     global seed      = parse(Int, val)
    else   @warn "Unknown argument: $arg"
    end
end

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

# ---- Load model and static designs ----
ps_cpu, st_cpu, _ = load_checkpoint_cpu(checkpoint)

static_designs = load_static_designs(results_dir)
has_spce_opt = any(p -> p.first == "static_spce", static_designs)

println("\n=== GPU-Accelerated Targeted sPCE Evaluation ===")
println("n_trials   = $n_trials")
println("L          = $L")
println("M          = $M")
println("B          = $B")
println("n_substeps = $n_substeps")
println("seed       = $seed")
println()
for (name, d) in static_designs
    println("$name: [", join(round.(d; digits=3), ", "), "]")
end
println()
flush(stdout)

# ---- Set up GPU ----
Reactant.set_default_backend("gpu")
xdev = reactant_device()
println("Using device: ", xdev)

n_denom = L + 1

u0 = make_u0()
u0_ra = u0 |> xdev

rng_setup = MersenneTwister(seed)
_, st_init = Lux.setup(rng_setup, policy)
ps_ra = ps_cpu |> xdev
st_ra = st_init |> xdev

make_design_model() = Dense(1 => N_STEPS; use_bias=true)
static_model = make_design_model()
ps_static_init, st_static_init = Lux.setup(rng_setup, static_model)
ps_static_cpu = (layer_1 = (weight = reshape(copy(static_designs[1].second), N_STEPS, 1),
                            bias = zeros(Float32, N_STEPS)),)
ps_static_ra = ps_static_cpu |> xdev
st_static_ra = st_static_init |> xdev

# ---- Batch loop ----
n_batches = cld(n_trials, B)
rng = MersenneTwister(seed)

all_scores = Dict{String, Vector{Float64}}()
all_scores["adaptive"] = Float64[]
for (name, _) in static_designs
    all_scores[name] = Float64[]
end
t_start = time()
println("\nStarting evaluation: $n_batches batches of $B episodes")
println("First batch includes compilation time (~5-15 min)...")
flush(stdout)

for batch_idx in 1:n_batches
    actual_B = min(B, n_trials - (batch_idx - 1) * B)

    θ_full = sample_θ_full(rng, n_denom, B)
    θ_obs_numer = sample_θ_N_joint(rng, M, B)
    ε_shared = randn(rng, Float32, N_NOISE_CHANNELS, N_STEPS, B)

    θ_full_ra = θ_full |> xdev
    θ_obs_numer_ra = θ_obs_numer |> xdev

    input_buffer = zeros(Float32, 2, N_STEPS, B) |> xdev
    observations = zeros(Float32, N_STEPS, B) |> xdev
    designs_buf = zeros(Float32, N_STEPS, B) |> xdev
    ε_ra = ε_shared |> xdev
    ll_denom_buf = zeros(Float32, n_denom, B) |> xdev
    ll_numer_buf = zeros(Float32, M, B) |> xdev

    data_adaptive = (θ_full_ra, θ_obs_numer_ra, u0_ra,
                     input_buffer, observations, designs_buf, ε_ra,
                     ll_denom_buf, ll_numer_buf, n_substeps)

    scores_ra, _, _ = @jit adaptive_spce_eval(policy, ps_ra, st_ra, data_adaptive)
    scores_cpu = Array(scores_ra)
    append!(all_scores["adaptive"], Float64.(scores_cpu[1, 1:actual_B]))

    for (name, design) in static_designs
        ε_static = randn(rng, Float32, N_NOISE_CHANNELS, N_STEPS, B)
        ε_static_ra = ε_static |> xdev
        observations_s = zeros(Float32, N_STEPS, B) |> xdev
        ll_denom_s = zeros(Float32, n_denom, B) |> xdev
        ll_numer_s = zeros(Float32, M, B) |> xdev

        copyto!(ps_static_ra.layer_1.weight, reshape(design, N_STEPS, 1))

        data_static = (θ_full_ra, θ_obs_numer_ra, u0_ra,
                       observations_s, ε_static_ra, ll_denom_s, ll_numer_s, n_substeps)

        scores_s_ra, _, _ = @jit static_spce_eval(static_model, ps_static_ra, st_static_ra, data_static)
        scores_s_cpu = Array(scores_s_ra)
        append!(all_scores[name], Float64.(scores_s_cpu[1, 1:actual_B]))
    end

    t_elapsed = time() - t_start
    if batch_idx == 1
        @printf("  batch %d/%d done (%.1fs, includes compilation)\n", batch_idx, n_batches, t_elapsed)
    elseif batch_idx % 5 == 0 || batch_idx == n_batches
        @printf("  batch %d/%d done (%.1fs total)\n", batch_idx, n_batches, t_elapsed)
    end
    flush(stdout)
end

t_total = time() - t_start
@printf("\nTotal evaluation time: %.1fs\n", t_total)

# ---- Save ----
scores_dict = Dict{String, Any}(
    "adaptive_scores"      => all_scores["adaptive"],
    "static_std_scores"    => all_scores["static_std"],
    "n_trials"             => n_trials,
    "L"                    => L,
    "M"                    => M,
    "B"                    => B,
    "n_substeps"           => n_substeps,
    "seed"                 => seed,
    "wall_time_s"          => t_total,
)
if has_spce_opt
    scores_dict["static_spce_scores"] = all_scores["static_spce"]
end
serialize(joinpath(results_dir, "spce_scores.jls"), scores_dict)

# ---- Print and save score statistics ----
adaptive_scores = all_scores["adaptive"]
println("\n=== Results (targeted sPCE, higher = more informative) ===\n")
summary_lines = String[]

for name in DESIGN_ORDER
    haskey(all_scores, name) || continue
    scores = all_scores[name]
    style = get(DESIGN_STYLES, name, (label = name, color = :black))
    m = mean(scores)
    s = std(scores)
    sem = s / sqrt(length(scores))
    line = @sprintf("  %-30s  mean = %8.4f  std = %8.4f  SEM = %6.4f  (n=%d)",
                     style.label, m, s, sem, length(scores))
    println(line)
    push!(summary_lines, line)
end
println()

for name in DESIGN_ORDER
    name == "adaptive" && continue
    haskey(all_scores, name) || continue
    scores = all_scores[name]
    style = get(DESIGN_STYLES, name, (label = name, color = :black))
    delta = adaptive_scores .- scores
    t_stat = mean(delta) / (std(delta) / sqrt(length(delta)))
    line = @sprintf("  Paired: Adaptive - %-20s  delta = %+.4f ± %.4f (SEM)  t=%6.2f",
                     style.label, mean(delta), std(delta) / sqrt(length(delta)), t_stat)
    println(line)
    push!(summary_lines, line)
end
flush(stdout)

open(joinpath(results_dir, "spce_summary.txt"), "w") do io
    println(io, "# Targeted sPCE Evaluation (GPU)")
    println(io, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io)
    println(io, "n_trials = $n_trials")
    println(io, "L = $L")
    println(io, "M = $M")
    println(io, "B = $B")
    println(io, "n_substeps = $n_substeps")
    println(io, "seed = $seed")
    @printf(io, "wall_time_s = %.1f\n", t_total)
    println(io)
    for (name, d) in static_designs
        println(io, "$(name)_design = [", join(round.(d; digits=4), ", "), "]")
    end
    println(io)
    println(io, "=== Results (targeted sPCE, higher = more informative) ===")
    println(io)
    for line in summary_lines
        println(io, line)
    end
end

println("\nDone. Outputs in: $results_dir")
