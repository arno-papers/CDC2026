#=
Run training and generate diagnostic plots.
=#

include("mainGPU.jl")

using Plots
using Statistics: mean, std

# ============================================================================
#  Transfer trained policy to CPU
# ============================================================================

const cdev = cpu_device()
const ps_cpu = train_state.parameters |> cdev
const st_cpu = train_state.states |> cdev

# ============================================================================
#  Run policy on CPU (for plotting)
# ============================================================================

function run_policy_cpu(θ, u0, ps, st)
    u = reshape(u0, 1, 3)
    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    designs = Float32[]
    trajectory = [copy(vec(u))]

    st_local = st
    for step in 1:N_STEPS
        action, st_local = policy(input_buffer, ps, st_local)
        Q_in = action[1]
        push!(designs, Q_in)

        u = integrate_cpu(u, reshape(θ, 1, 2), Q_in, DT, N_SUBSTEPS)
        push!(trajectory, copy(vec(u)))

        # Update input buffer with observation and design
        y_obs = u[1, 1]  # C_s observation (no noise for visualization)
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = Q_in
    end

    return hcat(trajectory...)', designs
end

# ============================================================================
#  Plot 1: Loss curve
# ============================================================================

function plot_loss(loss_history)
    p = plot(loss_history,
        xlabel = "Iteration",
        ylabel = "Targeted sPCE Loss",
        title = "Training Loss",
        label = "Loss",
        linewidth = 2,
        marker = :circle,
        markersize = 3
    )

    # Add smoothed version
    if length(loss_history) > 10
        window = min(10, length(loss_history) ÷ 5)
        smoothed = [mean(loss_history[max(1,i-window):i]) for i in 1:length(loss_history)]
        plot!(p, smoothed, label = "Smoothed", linewidth = 2, linestyle = :dash)
    end

    return p
end

# ============================================================================
#  Plot 2: State trajectories with trained policy
# ============================================================================

function plot_trajectories(; n_samples = 10)
    u0 = Float32[3.0, 0.25, 7.0]
    t = 0:N_STEPS

    p1 = plot(title = "Substrate Cₛ", xlabel = "Step", ylabel = "Concentration")
    p2 = plot(title = "Biomass Cₓ", xlabel = "Step", ylabel = "Concentration")
    p3 = plot(title = "Volume V", xlabel = "Step", ylabel = "Volume")
    p4 = plot(title = "Design Qᵢₙ", xlabel = "Step", ylabel = "Flow rate")

    for i in 1:n_samples
        θ = Float32[
            μ_max_lo + (μ_max_hi - μ_max_lo) * rand(rng),
            K_s_lo + (K_s_hi - K_s_lo) * rand(rng)
        ]

        traj, designs = run_policy_cpu(θ, u0, ps_cpu, st_cpu)

        label = i == 1 ? "Samples" : ""
        plot!(p1, t, traj[:, 1], alpha = 0.5, color = :blue, label = label)
        plot!(p2, t, traj[:, 2], alpha = 0.5, color = :red, label = label)
        plot!(p3, t, traj[:, 3], alpha = 0.5, color = :green, label = label)
        plot!(p4, 1:N_STEPS, designs, alpha = 0.5, color = :purple, label = label, marker = :circle)
    end

    hline!(p1, [0], color = :black, linestyle = :dash, label = "Zero", linewidth = 1)
    hline!(p2, [0], color = :black, linestyle = :dash, label = "", linewidth = 1)

    return plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 700))
end

# ============================================================================
#  Plot 3: Observation spread with trained policy
# ============================================================================

function plot_observation_spread(; n_samples = 100)
    u0 = Float32[3.0, 0.25, 7.0]
    observations = zeros(Float32, n_samples, N_STEPS + 1)

    for i in 1:n_samples
        θ = Float32[
            μ_max_lo + (μ_max_hi - μ_max_lo) * rand(rng),
            K_s_lo + (K_s_hi - K_s_lo) * rand(rng)
        ]
        traj, _ = run_policy_cpu(θ, u0, ps_cpu, st_cpu)
        observations[i, :] = traj[:, 1]
    end

    t = 0:N_STEPS
    obs_mean = vec(mean(observations, dims = 1))
    obs_std = vec(std(observations, dims = 1))

    p = plot(t, obs_mean, ribbon = 2 * obs_std,
             fillalpha = 0.3,
             xlabel = "Step",
             ylabel = "Cₛ",
             title = "Observation Spread (Mean ± 2σ)",
             label = "With trained policy",
             linewidth = 2)

    return p
end

# ============================================================================
#  CPU-safe integration (without @trace)
# ============================================================================

function integrate_cpu(u, θ, Q_in, dt, n_substeps)
    dt_sub = dt / n_substeps
    for _ in 1:n_substeps
        u = rk4_step(u, θ, Q_in, dt_sub)
    end
    return u
end

# ============================================================================
#  Plot 4: Posterior visualization
# ============================================================================

function simulate_with_designs(θ, designs, u0)
    u = reshape(u0, 1, 3)
    observations = Float32[]

    for Q_in in designs
        u = integrate_cpu(u, reshape(θ, 1, 2), Q_in, DT, N_SUBSTEPS)
        push!(observations, u[1, 1])  # C_s observation
    end

    return observations
end

function compute_log_likelihood(obs_true, obs_pred, σ)
    ll = 0.0f0
    for (y_true, y_pred) in zip(obs_true, obs_pred)
        ll -= 0.5f0 * ((y_true - y_pred) / σ)^2
        ll -= 0.5f0 * log(σ^2)
    end
    return ll
end

function plot_posterior(; θ_true = nothing, σ_obs = 1.0f0, grid_size = 100)
    u0 = Float32[3.0, 0.25, 7.0]

    # Use provided true params or sample random
    if isnothing(θ_true)
        θ_true = Float32[
            μ_max_lo + (μ_max_hi - μ_max_lo) * 0.5,  # Middle of prior
            K_s_lo + (K_s_hi - K_s_lo) * 0.5
        ]
    end

    # Run policy with true parameters to get observations and designs
    traj, designs = run_policy_cpu(θ_true, u0, ps_cpu, st_cpu)
    obs_true = traj[2:end, 1]  # C_s at each step (skip initial)

    # Sanity check: simulate_with_designs should reproduce obs_true
    obs_check = simulate_with_designs(θ_true, designs, u0)
    max_err = maximum(abs.(obs_true .- obs_check))
    if max_err > 1e-5
        @warn "Simulation mismatch!" max_err obs_true obs_check
    end

    # Add noise to observations
    obs_noisy = obs_true .+ σ_obs .* randn(rng, Float32, length(obs_true))

    # Compute log-likelihood over parameter grid
    μ_max_grid = range(μ_max_lo, μ_max_hi, length = grid_size)
    K_s_grid = range(K_s_lo, K_s_hi, length = grid_size)

    log_lik = zeros(Float32, grid_size, grid_size)

    for (i, μ_max) in enumerate(μ_max_grid)
        for (j, K_s) in enumerate(K_s_grid)
            θ_test = Float32[μ_max, K_s]
            obs_pred = simulate_with_designs(θ_test, designs, u0)
            log_lik[i, j] = compute_log_likelihood(obs_noisy, obs_pred, σ_obs)
        end
    end

    # Convert to posterior (proportional, with uniform prior)
    log_lik_shifted = log_lik .- maximum(log_lik)
    posterior = exp.(log_lik_shifted)
    posterior ./= sum(posterior)

    p = heatmap(μ_max_grid, K_s_grid, posterior',
                xlabel = "μ_max",
                ylabel = "K_s",
                title = "Posterior (θ_true = $(round.(θ_true, digits=2)))",
                color = :viridis,
                clims = (0, maximum(posterior)))

    # Mark true parameter
    scatter!(p, [θ_true[1]], [θ_true[2]],
             marker = :star, markersize = 12, color = :red,
             label = "True θ")

    return p
end

# ============================================================================
#  Generate plots
# ============================================================================

println("\n=== Generating Plots ===")

p_loss = plot_loss(loss_history)
savefig(p_loss, "plot_loss.png")
println("  Saved plot_loss.png")

p_traj = plot_trajectories(n_samples = 20)
savefig(p_traj, "plot_trajectories.png")
println("  Saved plot_trajectories.png")

p_spread = plot_observation_spread(n_samples = 200)
savefig(p_spread, "plot_spread.png")
println("  Saved plot_spread.png")

# Posterior plots for different true parameters
println("  Generating posterior plots...")
p_post1 = plot_posterior(θ_true = Float32[0.35, 0.4])
savefig(p_post1, "plot_posterior_1.png")
println("  Saved plot_posterior_1.png")

p_post2 = plot_posterior(θ_true = Float32[0.45, 0.5])
savefig(p_post2, "plot_posterior_2.png")
println("  Saved plot_posterior_2.png")

p_post3 = plot_posterior(θ_true = Float32[0.4, 0.45])
savefig(p_post3, "plot_posterior_3.png")
println("  Saved plot_posterior_3.png")

# Combined posterior plot
p_posteriors = plot(p_post1, p_post2, p_post3, layout = (1, 3), size = (1200, 400))
savefig(p_posteriors, "plot_posteriors.png")
println("  Saved plot_posteriors.png")

# Summary
p_summary = plot(p_loss, p_spread, layout = (1, 2), size = (1000, 400))
savefig(p_summary, "plot_summary.png")
println("  Saved plot_summary.png")

println("\n✓ All plots saved!")
