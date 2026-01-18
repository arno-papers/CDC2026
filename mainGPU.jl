#=
DADS - Deep Adaptive Design with Targeted sPCE (GPU version with Reactant + Enzyme)

Fed-batch bioreactor with uncertain Monod kinetics.
Supports nuisance parameters (measurement noise σ) with targeted inference on (μ_max, K_s).

Uses Reactant for XLA compilation and Enzyme for automatic differentiation on GPU.

See:
- https://ae-foster.github.io/posts/2022/05/20/brl.html
- https://arnostrouwen.com/posts/dynamic-experimental-design/

Does not run on Julia 1.12
=#

using Lux, Reactant, Random
using Reactant: @trace
using Optimisers
using Printf

Reactant.set_default_backend("gpu")

# ============================================================================
#  Bioreactor Dynamics
# ============================================================================

function bioreactor_dynamics(u, θ, Q_in)
    C_s = u[:, 1]
    C_x = u[:, 2]
    V = u[:, 3]
    μ_max = θ[:, 1]
    K_s = θ[:, 2]
    μ = @. μ_max * C_s / (K_s + C_s)
    σ = @. μ / 0.777f0
    du1 = @. -σ * C_x + Q_in / V * (50.0f0 - C_s)
    du2 = @. μ * C_x - Q_in / V * C_x
    du3 = fill(Q_in, size(u, 1))
    return [du1 du2 du3]
end

function rk4_step(u, θ, Q_in, dt)
    k1 = bioreactor_dynamics(u, θ, Q_in)
    k2 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k1, θ, Q_in)
    k3 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k2, θ, Q_in)
    k4 = bioreactor_dynamics(u .+ dt .* k3, θ, Q_in)
    return u .+ (dt / 6.0f0) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
end

function integrate(u, θ, Q_in, dt, n_substeps)
    u = u .+ 0  # Materialize any lazy wrappers (ReshapedArray -> concrete array)
    dt_sub = dt / n_substeps
    @trace for _ in 1:n_substeps
        u = rk4_step(u, θ, Q_in, dt_sub)
    end
    return u
end

# ============================================================================
#  Positional Encoding
# ============================================================================

function sinusoidal_pe(seq_len::Int)
    position = reshape(Float32.(0:(seq_len - 1)), 1, seq_len)
    div_term = exp.(Float32.(0:2:31) .* -(log(1000.0f0) / 32.0f0))
    angles = div_term * position
    pe = zeros(Float32, 32, seq_len)
    pe[1:2:end, :] .= sin.(angles)
    pe[2:2:end, :] .= cos.(angles[1:16, :])
    return pe
end

# ============================================================================
#  Policy
# ============================================================================

const policy = @compact(
    input_proj = Dense(2 => 32),
    rms1 = RMSNorm((32,)),
    mha = MultiHeadAttention(32; nheads = 4),
    rms2 = RMSNorm((32,)),
    ff = Chain(Dense(32 => 64, gelu), Dense(64 => 32)),
    output_head = Dense(32 => 1),
) do x
    seq_len = size(x, 2)
    x = input_proj(x)
    x = x .+ reshape(sinusoidal_pe(seq_len), 32, seq_len, 1)
    h = rms1(x)
    attn, _ = mha(h)
    x = x + attn
    x = x + ff(rms2(x))
    @return 10.0f0 .* sigmoid.(output_head(x[:, end, :]))
end

# ============================================================================
#  Targeted sPCE Loss with Nuisance Parameters
#
#  θ = (μ_max, K_s, σ) - full parameters (3D)
#       θ_T = (μ_max, K_s) - target parameters
#       θ_N = σ             - nuisance parameter
#
#  Numerator: average over M nuisance samples given θ_{T,0}
#  Denominator: average over L+1 full θ samples
# ============================================================================

const N_STEPS = 10
const DT = 1.0f0            # Total time per control interval
const N_SUBSTEPS = 500      # Integration substeps per control interval
const L_CONTRASTIVE = 256   # Contrastive samples for denominator
const M_NUISANCE = 128      # Nuisance samples for numerator

# Prior bounds
const μ_max_lo, μ_max_hi = 0.3f0, 0.5f0
const K_s_lo, K_s_hi = 0.3f0, 0.6f0
const σ_lo, σ_hi = 0.05f0, 0.15f0  # Nuisance: measurement noise std

function targeted_spce_loss(model, ps, st, data)
    # θ_full: (3, L+1) - full parameter samples [μ_max, K_s, σ]
    # θ_N_numer: (M,) - nuisance samples for numerator
    θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom = data

    # Extract true parameters (first sample) - keep as arrays for Reactant tracing
    # Use permutedims instead of ' to avoid Adjoint wrapper
    θ_T_true = permutedims(θ_full[1:2, 1:1], (2, 1))  # (1, 2)
    σ_true = θ_full[3:3, 1:1]                         # (1, 1) array - NO scalar indexing

    u = reshape(u0, 1, 3)                # (1, 3) - batch of 1

    # Rollout with true θ
    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        Q_in = @allowscalar action[1]

        designs = @allowscalar begin
            designs[step] = Q_in
            designs
        end

        u = integrate(u, θ_T_true, Q_in, DT, N_SUBSTEPS)

        # Noisy observation: y = C_s + σ_true * ε
        obs = u[1:1, 1:1]                # (1, 1) array
        noise = ε[step:step]             # (1,) array
        y_noisy = obs .+ σ_true .* noise # broadcast - result is (1, 1)

        y_scalar = @allowscalar y_noisy[1]
        observations = @allowscalar begin
            observations[step] = y_scalar
            observations
        end

        input_buffer = @allowscalar begin
            input_buffer[1, step, 1] = y_scalar
            input_buffer[2, step, 1] = Q_in
            input_buffer
        end
    end

    # ========================================================================
    # DENOMINATOR: (1/(L+1)) * Σ_ℓ p(h_K | θ_ℓ, π)
    # ========================================================================
    n_denom = L_CONTRASTIVE + 1
    θ_T_denom = permutedims(θ_full[1:2, :], (2, 1))  # (n_denom, 2)
    σ²_denom = vec(θ_full[3:3, :]) .^ 2              # (n_denom,)

    u_denom = repeat(reshape(u0, 1, 3), n_denom, 1)  # (n_denom, 3)

    for step in 1:N_STEPS
        d = @allowscalar designs[step]
        u_denom = integrate(u_denom, θ_T_denom, d, DT, N_SUBSTEPS)
        pred_obs = u_denom[:, 1]                              # (n_denom,)
        actual_obs = @allowscalar observations[step]          # scalar
        residual = actual_obs .- pred_obs
        ll_denom = ll_denom .- 0.5f0 .* residual.^2 ./ σ²_denom .- 0.5f0 .* log.(σ²_denom)
    end

    # ========================================================================
    # NUMERATOR: (1/M) * Σ_m p(h_K | θ_{T,0}, θ_N^{(m)}, π)
    # All M samples share θ_T_true, only σ differs - simulate once, broadcast likelihood
    # ========================================================================
    σ²_numer = θ_N_numer .^ 2                        # (M,)
    u_numer = reshape(u0, 1, 3)                      # (1, 3) - single simulation
    ll_numer = zeros(Float32, M_NUISANCE)            # (M,)

    for step in 1:N_STEPS
        d = @allowscalar designs[step]
        u_numer = integrate(u_numer, θ_T_true, d, DT, N_SUBSTEPS)
        pred_obs = u_numer[1:1, 1:1]                 # (1, 1) array
        actual_obs = observations[step:step]         # (1,) array
        residual² = (actual_obs .- pred_obs) .^ 2    # (1, 1) array
        # Broadcast residual² over all M sigma values
        ll_numer = ll_numer .- 0.5f0 .* vec(residual²) ./ σ²_numer .- 0.5f0 .* log.(σ²_numer)
    end

    # ========================================================================
    # Targeted sPCE loss
    # ========================================================================
    ll_max_num = maximum(ll_numer)
    lse_num = ll_max_num + log(sum(exp.(ll_numer .- ll_max_num)))
    log_numerator = lse_num - log(Float32(M_NUISANCE))

    ll_max_den = maximum(ll_denom)
    lse_den = ll_max_den + log(sum(exp.(ll_denom .- ll_max_den)))
    log_denominator = lse_den - log(Float32(n_denom))

    loss = -(log_numerator - log_denominator)

    return loss, st, (;)
end

# ============================================================================
#  Sampling
# ============================================================================

function sample_θ_full(rng, n_samples)
    # Full parameters: (μ_max, K_s, σ)
    θ = rand(rng, Float32, 3, n_samples)
    θ[1, :] .= μ_max_lo .+ (μ_max_hi - μ_max_lo) .* θ[1, :]
    θ[2, :] .= K_s_lo .+ (K_s_hi - K_s_lo) .* θ[2, :]
    θ[3, :] .= σ_lo .+ (σ_hi - σ_lo) .* θ[3, :]
    return θ
end

function sample_θ_N(rng, n_samples)
    θ = rand(rng, Float32, n_samples)
    θ .= σ_lo .+ (σ_hi - σ_lo) .* θ
    return θ
end

# ============================================================================
#  Training
# ============================================================================

function train_policy(model, ps, st, rng; n_iters = 50)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
    loss_history = Float32[]

    for iteration in 1:n_iters
        # θ_full: L+1 full samples (first generates data, all used in denominator)
        θ_full = sample_θ_full(rng, L_CONTRASTIVE + 1) |> xdev

        # θ_N_numer: M nuisance samples for numerator
        θ_N_numer = sample_θ_N(rng, M_NUISANCE) |> xdev

        # Buffers
        u0 = Float32[3.0, 0.25, 7.0] |> xdev
        input_buffer = zeros(Float32, 2, N_STEPS, 1) |> xdev
        observations = zeros(Float32, N_STEPS) |> xdev
        designs = zeros(Float32, N_STEPS) |> xdev
        ε = randn(rng, Float32, N_STEPS) |> xdev
        n_denom = L_CONTRASTIVE + 1
        ll_denom = zeros(Float32, n_denom) |> xdev   # (n_denom,) - accumulate log-likelihoods

        data = (
            θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε, ll_denom
        )

        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoEnzyme(), targeted_spce_loss, data, train_state
        )
        push!(loss_history, loss)

        if iteration % 10 == 0 || iteration == 1
            @printf("Iter: [%4d/%4d]\tTargeted sPCE Loss: %.8f\n", iteration, n_iters, loss)
        end
    end

    return train_state, loss_history
end

# ============================================================================
#  Setup and run
# ============================================================================

println("\n=== Targeted DADS Training (GPU with Reactant + Enzyme) ===")
println("Target params: (μ_max, K_s), Nuisance: σ_measure")
println("L = $L_CONTRASTIVE contrastive, M = $M_NUISANCE nuisance samples\n")

const rng = Random.default_rng()
ps, st = Lux.setup(rng, policy)

const xdev = reactant_device()
println("Using device: ", xdev)

ps_ra = ps |> xdev
st_ra = st |> xdev

println("Starting training...")
train_state, loss_history = train_policy(policy, ps_ra, st_ra, rng; n_iters=1000)

println("\n✓ Targeted DADS training complete!")
