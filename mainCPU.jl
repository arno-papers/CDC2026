#=
DADS - Deep Adaptive Design with Targeted sPCE (CPU version with Mooncake)

Fed-batch bioreactor with uncertain Monod kinetics.
Supports nuisance parameters (measurement noise σ) with targeted inference on (μ_max, K_s).

Uses Mooncake for automatic differentiation on CPU.

See:
- https://ae-foster.github.io/posts/2022/05/20/brl.html
- https://arnostrouwen.com/posts/dynamic-experimental-design/
=#

using StableRNGs
using Lux, ADTypes, Optimisers, Mooncake
using LogExpFunctions: logsumexp
using Printf

const RNG = StableRNGs.StableRNG(42)

# ============================================================================
#  Bioreactor Dynamics
# ============================================================================

function bioreactor_dynamics(u, θ, Q_in)
    C_s, C_x, V = u
    μ_max, K_s = θ
    μ = μ_max * C_s / (K_s + C_s)
    σ = μ / 0.777f0
    return Float32[-σ * C_x + Q_in / V * (50f0 - C_s),
                   μ * C_x - Q_in / V * C_x,
                   Q_in]
end

function rk4_step(u, θ, Q_in, dt)
    k1 = bioreactor_dynamics(u, θ, Q_in)
    k2 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k1, θ, Q_in)
    k3 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k2, θ, Q_in)
    k4 = bioreactor_dynamics(u .+ dt .* k3, θ, Q_in)
    return u .+ (dt / 6f0) .* (k1 .+ 2f0 .* k2 .+ 2f0 .* k3 .+ k4)
end

# ============================================================================
#  Positional Encoding
# ============================================================================

function sinusoidal_pe(seq_len::Int)
    position = reshape(Float32.(0:(seq_len-1)), 1, seq_len)
    div_term = exp.(Float32.(0:2:31) .* -(log(1000f0) / 32f0))
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
    mha = MultiHeadAttention(32; nheads=4),
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
    @return 10f0 .* sigmoid.(output_head(x[:, end, :]))
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

const N_STEPS = 5
const L_CONTRASTIVE = 10    # Contrastive samples for denominator
const M_NUISANCE = 5        # Nuisance samples for numerator

# Prior bounds
const μ_max_lo, μ_max_hi = 0.3f0, 0.5f0
const K_s_lo, K_s_hi = 0.3f0, 0.6f0
const σ_lo, σ_hi = 0.05f0, 0.15f0  # Nuisance: measurement noise std

function targeted_spce_loss(model, ps, st, data)
    # θ_full: (3, L+1) - full parameter samples [μ_max, K_s, σ]
    # θ_N_numer: (M,) - nuisance samples for numerator
    θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε = data
    
    # Extract true parameters (first sample)
    θ_T_true = θ_full[1:2, 1]
    σ_true = θ_full[3, 1]
    
    u = copy(u0)
    
    # Rollout with true θ
    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        Q_in = action[1]
        designs[step] = Q_in
        
        u = rk4_step(u, θ_T_true, Q_in, 1f0)
        
        # Noisy observation: y = C_s + σ_true * ε
        y_noisy = u[1] + σ_true * ε[step]
        observations[step] = y_noisy
        
        input_buffer[1, step, 1] = y_noisy
        input_buffer[2, step, 1] = Q_in
    end
    
    # ========================================================================
    # DENOMINATOR: (1/(L+1)) * Σ_ℓ p(h_K | θ_ℓ, π)
    # ========================================================================
    n_denom = L_CONTRASTIVE + 1
    log_likes_denom = zeros(Float32, n_denom)
    
    for i in 1:n_denom
        θ_T_i = θ_full[1:2, i]
        σ_i = θ_full[3, i]
        σ²_i = σ_i^2
        
        u_i = copy(u0)
        ll = 0f0
        for step in 1:N_STEPS
            u_i = rk4_step(u_i, θ_T_i, designs[step], 1f0)
            ll -= 0.5f0 * (observations[step] - u_i[1])^2 / σ²_i + 0.5f0 * log(σ²_i)
        end
        log_likes_denom[i] = ll
    end
    
    # ========================================================================
    # NUMERATOR: (1/M) * Σ_m p(h_K | θ_{T,0}, θ_N^{(m)}, π)  
    # ========================================================================
    log_likes_numer = zeros(Float32, M_NUISANCE)
    
    for m in 1:M_NUISANCE
        σ_m = θ_N_numer[m]
        σ²_m = σ_m^2
        
        u_m = copy(u0)
        ll = 0f0
        for step in 1:N_STEPS
            u_m = rk4_step(u_m, θ_T_true, designs[step], 1f0)
            ll -= 0.5f0 * (observations[step] - u_m[1])^2 / σ²_m + 0.5f0 * log(σ²_m)
        end
        log_likes_numer[m] = ll
    end
    
    # ========================================================================
    # Targeted sPCE loss
    # ========================================================================
    log_numerator = logsumexp(log_likes_numer) - log(Float32(M_NUISANCE))
    log_denominator = logsumexp(log_likes_denom) - log(Float32(n_denom))
    
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

function train_policy(model, ps, st, rng; n_iters=50)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
    
    for iteration in 1:n_iters
        # θ_full: L+1 full samples (first generates data, all used in denominator)
        θ_full = sample_θ_full(rng, L_CONTRASTIVE + 1)
        
        # θ_N_numer: M nuisance samples for numerator
        θ_N_numer = sample_θ_N(rng, M_NUISANCE)
        
        # Buffers
        u0 = Float32[3.0, 0.25, 7.0]
        input_buffer = zeros(Float32, 2, N_STEPS, 1)
        observations = zeros(Float32, N_STEPS)
        designs = zeros(Float32, N_STEPS)
        ε = randn(rng, Float32, N_STEPS)
        
        data = (θ_full, θ_N_numer, u0, input_buffer, observations, designs, ε)
        
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoMooncake(; config=nothing), targeted_spce_loss, data, train_state
        )
        
        if iteration % 10 == 0 || iteration == 1
            @printf("Iter: [%4d/%4d]\tTargeted sPCE Loss: %.8f\n", iteration, n_iters, loss)
        end
    end
    
    return train_state
end

# ============================================================================
#  Setup and run
# ============================================================================

println("\n=== Targeted DADS Training (CPU with Mooncake) ===")
println("Target params: (μ_max, K_s), Nuisance: σ_measure")
println("L = $L_CONTRASTIVE contrastive, M = $M_NUISANCE nuisance samples\n")

ps, st = Lux.setup(RNG, policy)

println("Starting training...")
train_state = train_policy(policy, ps, st, RNG)

println("\n✓ Targeted DADS training complete!")
