using Plots, Random, Serialization, Statistics
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "plotting.jl"))

# ============================================================================
#  Timed rollout: returns trajectory, observations, designs, and per-step
#  policy evaluation times.
# ============================================================================

function rollout_timed(model, ps_cpu, st_cpu, rng, theta_dyn, sigma;
                       n_substeps::Int=N_SUBSTEPS)
    k, J, f_val = theta_dyn
    denom = Float32(R_CONST) * f_val + k^2
    u = reshape(Float32[f_val * Float32(V_PRE) / denom,
                         k * Float32(V_PRE) / denom, 0.0f0], 3, 1)
    theta_mat = reshape(Float32[k, J, f_val], 3, 1)

    traj = zeros(Float32, N_STEPS + 1, 2)      # [i, ω] per step
    traj[1, :] .= u[1:2]
    observations = zeros(Float32, N_STEPS)
    designs = zeros(Float32, N_STEPS)
    nn_times_s = zeros(Float64, N_STEPS)
    gc_times_s = zeros(Float64, N_STEPS)
    compile_times_s = zeros(Float64, N_STEPS)

    input_buffer = zeros(Float32, 2, N_STEPS, 1)
    st_local = st_cpu

    GC.gc()
    for step in 1:N_STEPS
        timed = @timed model(input_buffer, ps_cpu, st_local)
        action, st_local = timed.value
        nn_times_s[step] = timed.time
        gc_times_s[step] = timed.gctime
        compile_times_s[step] = timed.compile_time

        v_in = clamp(Float32(action[1]), ACTION_LO, ACTION_HI)
        designs[step] = v_in
        u = integrate_cpu(u, theta_mat, v_in, DT, n_substeps)
        traj[step + 1, :] .= u[1:2]

        y_obs = u[2, 1] + sigma * randn(rng, Float32)
        observations[step] = y_obs
        input_buffer[1, step, 1] = y_obs
        input_buffer[2, step, 1] = v_in
    end

    return (; traj, observations, designs,
              nn_times_us=nn_times_s .* 1e6,
              gc_times_us=gc_times_s .* 1e6,
              compile_times_us=compile_times_s .* 1e6)
end

# ============================================================================
#  Plot: rollout with real-time timing annotation
# ============================================================================

function plot_realtime_rollout(model, ps_cpu, st_cpu;
        rng, n_k=5, n_noise_replicates=3,
        outfile=joinpath(@__DIR__, "results", "plot_trajectories.png"))

    t_phys_ms = Float64.((0:N_STEPS) .* DT .* 1000)   # physical time in ms
    t_obs_ms = Float64.((1:N_STEPS) .* DT .* 1000)     # observation times
    dt_ms = Float64(DT) * 1000                          # sampling interval in ms

    # Fixed nuisance parameters (midpoints)
    J_fix = (J_lo + J_hi) / 2
    f_fix = (f_lo + f_hi) / 2
    σ_fix = (sigma_lo + sigma_hi) / 2

    # Sweep k across its prior range
    k_vals = range(k_lo, k_hi; length=n_k)
    colors = cgrad(:viridis, n_k; categorical=true)

    # Warm up the model so timing is representative
    warmup_rng = MersenneTwister(999)
    rollout_timed(model, ps_cpu, st_cpu, warmup_rng,
                  Float32[0.5, J_fix, f_fix], σ_fix)

    # Collect rollouts: n_k values × n_noise_replicates each
    results = []  # (k_idx, k_val, result)
    for (ki, k) in enumerate(k_vals)
        theta = Float32[k, J_fix, f_fix]
        for _ in 1:n_noise_replicates
            GC.gc()
            r = rollout_timed(model, ps_cpu, st_cpu, rng, theta, σ_fix)
            push!(results, (ki, k, r))
        end
    end

    # --- Panel 1: Angular velocity + observations ---
    p1 = plot(xlabel="Time (ms)", ylabel="ω (rad/s)",
              title="Angular velocity")
    labeled = Set{Int}()
    for (ki, k, r) in results
        first = ki ∉ labeled
        lab = first ? @sprintf("k = %.2f", k) : ""
        first && push!(labeled, ki)
        plot!(p1, t_phys_ms, r.traj[:, 2]; color=colors[ki], lw=1.2, alpha=0.7, label=lab)
        scatter!(p1, t_obs_ms, r.observations; color=colors[ki],
                 markersize=2.5, markerstrokewidth=0, alpha=0.5, label="")
    end

    # --- Panel 2: Voltage designs ---
    p2 = plot(xlabel="Time (ms)", ylabel="V (V)",
              title="Applied voltage (design)", ylims=(-0.5, ACTION_HI + 0.5))
    labeled = Set{Int}()
    for (ki, k, r) in results
        first = ki ∉ labeled
        lab = first ? @sprintf("k = %.2f", k) : ""
        first && push!(labeled, ki)
        plot!(p2, t_obs_ms, r.designs; color=colors[ki], lw=1.5, alpha=0.7,
              marker=:circle, markersize=2.5, markerstrokewidth=0, label=lab)
    end

    # --- Panel 3: NN eval time vs sampling interval ---
    all_times = vcat([r.nn_times_us for (_, _, r) in results]...)
    med_us = median(all_times)

    p3 = plot(xlabel="Time (ms)", ylabel="Policy eval (μs)",
              title="Policy evaluation time", yscale=:log10)

    for (ki, _, r) in results
        scatter!(p3, t_obs_ms, r.nn_times_us; alpha=0.5, color=colors[ki],
                 markersize=2.5, markerstrokewidth=0, label="")
    end

    # Sampling interval reference line
    hline!(p3, [dt_ms * 1000]; color=:red, lw=2, ls=:dash,
           label=@sprintf("Δt = %.0f ms = %.0f μs", dt_ms, dt_ms * 1000))
    annotate!(p3, [(t_obs_ms[end] * 0.5, med_us * 3,
              text(@sprintf("median = %.0f μs (%.0f× margin)", med_us, dt_ms * 1000 / med_us),
                   8, :left, :gray40))])

    plt = plot(p1, p2, p3; layout=(1, 3), size=(1400, 400),
               bottom_margin=5Plots.mm, left_margin=5Plots.mm)
    save_plot(plt, outfile)
    return plt
end

# ============================================================================
#  Plot: standalone policy evaluation timing with boxplots
# ============================================================================

function plot_policy_timing(model, ps_cpu, st_cpu;
        rng, n_rollouts=500,
        outfile=joinpath(@__DIR__, "results", "plot_policy_timing.png"))

    t_obs_ms = Float64.((1:N_STEPS) .* DT .* 1000)
    dt_ms = Float64(DT) * 1000
    dt_us = dt_ms * 1000

    # Warm up
    warmup_rng = MersenneTwister(999)
    theta_warmup = Float32[0.5, (J_lo+J_hi)/2, (f_lo+f_hi)/2]
    rollout_timed(model, ps_cpu, st_cpu, warmup_rng, theta_warmup, 1.0f0)

    # Collect timing: (N_STEPS, n_rollouts) matrices
    timing_matrix = zeros(Float64, N_STEPS, n_rollouts)
    gc_matrix = zeros(Float64, N_STEPS, n_rollouts)
    compile_matrix = zeros(Float64, N_STEPS, n_rollouts)
    for j in 1:n_rollouts
        theta = Float32[
            k_lo + (k_hi - k_lo) * rand(rng),
            J_lo + (J_hi - J_lo) * rand(rng),
            f_lo + (f_hi - f_lo) * rand(rng),
        ]
        sigma = sigma_lo + (sigma_hi - sigma_lo) * rand(rng, Float32)
        r = rollout_timed(model, ps_cpu, st_cpu, rng, theta, sigma)
        timing_matrix[:, j] .= r.nn_times_us
        gc_matrix[:, j] .= r.gc_times_us
        compile_matrix[:, j] .= r.compile_times_us
    end

    # Diagnostic: investigate spikes
    all_wall = vec(timing_matrix)
    all_gc = vec(gc_matrix)
    all_compile = vec(compile_matrix)
    spike_mask = all_wall .> quantile(all_wall, 0.99)
    n_spikes = sum(spike_mask)
    if n_spikes > 0
        println("=== Timing spike analysis (top 1%, n=$n_spikes) ===")
        @printf("  wall:    median=%.0f μs, max=%.0f μs\n", median(all_wall[spike_mask]), maximum(all_wall[spike_mask]))
        @printf("  gc:      median=%.0f μs, max=%.0f μs\n", median(all_gc[spike_mask]), maximum(all_gc[spike_mask]))
        @printf("  compile: median=%.0f μs, max=%.0f μs\n", median(all_compile[spike_mask]), maximum(all_compile[spike_mask]))
        gc_fraction = sum(all_gc[spike_mask]) / sum(all_wall[spike_mask])
        compile_fraction = sum(all_compile[spike_mask]) / sum(all_wall[spike_mask])
        @printf("  gc accounts for %.1f%% of spike wall time\n", gc_fraction * 100)
        @printf("  compile accounts for %.1f%% of spike wall time\n", compile_fraction * 100)
    end

    n_total = N_STEPS * n_rollouts

    p = plot(xlabel="Experiment step", ylabel="Policy eval (μs)",
             title=@sprintf("Policy evaluation time (%d calls across %d rollouts)", n_total, n_rollouts),
             yscale=:log10, legend=:topright, size=(700, 400),
             bottom_margin=5Plots.mm, left_margin=5Plots.mm)

    # Boxplot at each step with 99.9th percentile whiskers
    for (i, t) in enumerate(t_obs_ms)
        vals = timing_matrix[i, :]
        q1, q2, q3 = quantile(vals, [0.25, 0.5, 0.75])
        p001 = quantile(vals, 0.001)
        p999 = quantile(vals, 0.999)
        w = dt_ms * 0.3  # box half-width in ms

        # Whiskers to 0.1% / 99.9%
        plot!(p, [t, t], [p001, q1]; color=:mediumpurple, lw=1, label="")
        plot!(p, [t, t], [q3, p999]; color=:mediumpurple, lw=1, label="")
        plot!(p, [t-w/2, t+w/2], [p001, p001]; color=:mediumpurple, lw=1, label="")
        plot!(p, [t-w/2, t+w/2], [p999, p999]; color=:mediumpurple, lw=1, label="")

        # Box (IQR)
        plot!(p, Shape([t-w, t+w, t+w, t-w], [q1, q1, q3, q3]);
              fillcolor=:mediumpurple, fillalpha=0.3, linecolor=:mediumpurple, lw=1,
              label=(i == 1 ? "IQR (whiskers: 0.1–99.9%)" : ""))

        # Median line
        plot!(p, [t-w, t+w], [q2, q2]; color=:purple, lw=2, label="")
    end

    # Sampling interval reference
    hline!(p, [dt_us]; color=:red, lw=2, ls=:dash,
           label=@sprintf("Δt = %.0f ms = %.0f μs", dt_ms, dt_us))

    # Summary stats across all steps
    all_times = vec(timing_matrix)
    med_all = median(all_times)
    p999_all = quantile(all_times, 0.999)
    annotate!(p, [(t_obs_ms[3], med_all * 4,
              text(@sprintf("median = %.0f μs, p99.9 = %.0f μs (%.0f× margin)",
                            med_all, p999_all, dt_us / p999_all),
                   9, :left, :gray40))])

    save_plot(p, outfile)
    return p
end

# ============================================================================
#  Standalone
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir = joinpath(@__DIR__, "results")
    file = length(ARGS) >= 1 ? ARGS[1] : joinpath(results_dir, "checkpoint.jls")
    @assert isfile(file) "Checkpoint not found: $file. Run training first."

    ckpt = deserialize(file)
    ps_cpu = ckpt["parameters"]
    st_cpu = ckpt["states"]

    rng = MersenneTwister(42)
    plot_realtime_rollout(policy, ps_cpu, st_cpu; rng)

    rng2 = MersenneTwister(42)
    plot_policy_timing(policy, ps_cpu, st_cpu; rng=rng2)
end
