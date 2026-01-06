#=
DADS - Deep Adaptive Design (GPU version with Reactant + Enzyme)

Fed-batch bioreactor with uncertain Monod kinetics.
Uses Reactant for XLA compilation and Enzyme for automatic differentiation on GPU.

See:
- https://ae-foster.github.io/posts/2022/05/20/brl.html
- https://arnostrouwen.com/posts/dynamic-experimental-design/
=#

using Lux, Reactant, Random
using Optimisers
using Printf

Reactant.set_default_backend("gpu")

# ============================================================================
#  Bioreactor Dynamics
# ============================================================================

function bioreactor_dynamics(u, θ, Q_in)
    C_s = @allowscalar u[1]
    C_x = @allowscalar u[2]
    V = @allowscalar u[3]
    μ_max = @allowscalar θ[1]
    K_s = @allowscalar θ[2]
    μ = μ_max * C_s / (K_s + C_s)
    σ = μ / 0.777f0
    du1 = -σ * C_x + Q_in / V * (50f0 - C_s)
    du2 = μ * C_x - Q_in / V * C_x
    du3 = Q_in
    return [du1, du2, du3]
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
#  SPCE Loss (all buffers pre-allocated on device)
# ============================================================================

const N_STEPS = 5
const N_SAMPLES = 11
const σ_measure = 0.1f0

function spce_loss(model, ps, st, data)
    θ_samples, u0, input_buffer, observations, designs, ε, log_likes = data
    
    # Extract true θ
    θ_true_1 = @allowscalar θ_samples[1, 1]
    θ_true_2 = @allowscalar θ_samples[2, 1]
    θ_true = [θ_true_1, θ_true_2]
    
    u = u0
    σ² = σ_measure^2
    
    # Rollout with true θ
    for step in 1:N_STEPS
        action, st = model(input_buffer, ps, st)
        Q_in = @allowscalar action[1]
        
        # Store design
        designs = @allowscalar begin
            designs[step] = Q_in
            designs
        end
        
        u = rk4_step(u, θ_true, Q_in, 1f0)
        
        # Noisy observation using reparameterization: y = C_s + σ * ε
        obs = @allowscalar u[1]
        noise = @allowscalar ε[step]
        y_noisy = obs + σ_measure * noise
        
        # Store observation
        observations = @allowscalar begin
            observations[step] = y_noisy
            observations
        end
        
        # Policy sees noisy observations
        input_buffer = @allowscalar begin
            input_buffer[1, step, 1] = y_noisy
            input_buffer[2, step, 1] = Q_in
            input_buffer
        end
    end
    
    # Compute log-likelihoods for all θ samples
    for i in 1:N_SAMPLES
        θ_i_1 = @allowscalar θ_samples[1, i]
        θ_i_2 = @allowscalar θ_samples[2, i]
        θ_i = [θ_i_1, θ_i_2]
        
        u_i = u0
        ll = 0f0
        for step in 1:N_STEPS
            d = @allowscalar designs[step]
            u_i = rk4_step(u_i, θ_i, d, 1f0)
            pred_obs = @allowscalar u_i[1]
            actual_obs = @allowscalar observations[step]
            ll = ll - 0.5f0 * (actual_obs - pred_obs)^2 / σ²
        end
        log_likes = @allowscalar begin
            log_likes[i] = ll
            log_likes
        end
    end
    
    # SPCE loss with stable logsumexp
    ll_max = @allowscalar maximum(log_likes)
    lse = ll_max + log(sum(exp.(@allowscalar(log_likes) .- ll_max)))
    ll_1 = @allowscalar log_likes[1]
    loss = -(ll_1 - lse + log(Float32(N_SAMPLES)))
    
    return loss, st, (;)
end

# ============================================================================
#  Training
# ============================================================================

function sample_θ(rng, n_samples)
    θ = rand(rng, Float32, 2, n_samples)
    θ[1, :] .= 0.3f0 .+ 0.2f0 .* θ[1, :]
    θ[2, :] .= 0.3f0 .+ 0.3f0 .* θ[2, :]
    return θ
end

function train_policy(model, ps, st, rng; n_iters=50)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(0.001f0))
    
    for iteration in 1:n_iters
        # All buffers pre-allocated on device
        θ_samples = sample_θ(rng, N_SAMPLES) |> xdev
        u0 = Float32[3.0, 0.25, 7.0] |> xdev
        input_buffer = zeros(Float32, 2, N_STEPS, 1) |> xdev
        observations = zeros(Float32, N_STEPS) |> xdev
        designs = zeros(Float32, N_STEPS) |> xdev
        ε = randn(rng, Float32, N_STEPS) |> xdev
        log_likes = zeros(Float32, N_SAMPLES) |> xdev
        
        data = (θ_samples, u0, input_buffer, observations, designs, ε, log_likes)
        
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoEnzyme(), spce_loss, data, train_state
        )
        
        if iteration % 10 == 0 || iteration == 1
            @printf("Iter: [%4d/%4d]\tLoss: %.8f\n", iteration, n_iters, loss)
        end
    end
    
    return train_state
end

# ============================================================================
#  Setup and run
# ============================================================================

println("\n=== DADS Training (GPU with Reactant + Enzyme) ===\n")

const rng = Random.default_rng()
ps, st = Lux.setup(rng, policy)

const xdev = reactant_device()
println("Using device: ", xdev)

ps_ra = ps |> xdev
st_ra = st |> xdev

println("Starting training...")
train_state = train_policy(policy, ps_ra, st_ra, rng)

println("\n✓ DADS training complete!")
