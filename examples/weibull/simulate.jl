include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "common_core.jl"))

using Plots

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

# Fixed nuisance dynamics (midpoints)
CL_fix = 3.0f0
Q_d_fix = 1.5f0

# Constant infusion rate
Q_const = 5.0f0

# Grid of target parameters
k_a_vals  = [0.5f0, 1.5f0, 3.0f0]
k_tr_vals = [0.5f0, 1.5f0, 3.0f0]

t = collect(1:N_STEPS) .* Float64(DT)

p = Plots.plot(; xlabel="Time (hr)", ylabel="C_c (mg/L)",
    title="PK Profiles (Q_in=$(Q_const) mg/hr, CL=$(CL_fix), Q_d=$(Q_d_fix))",
    legend=:outerright, size=(800, 400))

println("Simulating PK concentration profiles...")
println("  V_c=$(V_C) L, V_p=$(V_P) L")
println("  Q_in=$(Q_const) mg/hr constant, $(N_STEPS) steps, dt=$(DT), substeps=$(N_SUBSTEPS)\n")

for k_a in k_a_vals
    for k_tr in k_tr_vals
        θ = reshape(Float32[k_a, k_tr, CL_fix, Q_d_fix], N_PARAMS_DYN, 1)
        u = zeros(Float32, 5, 1)
        concentrations = Float64[]

        for step in 1:N_STEPS
            u = integrate_cpu(u, θ, Q_const, DT, N_SUBSTEPS)
            push!(concentrations, Float64(u[4, 1] / V_C))
        end

        c_max = maximum(concentrations)
        t_max = argmax(concentrations)
        @printf("  k_a=%.1f, k_tr=%.1f: C_max=%.3f mg/L at t=%d hr\n", k_a, k_tr, c_max, t_max)

        Plots.plot!(p, t, concentrations;
            label="k_a=$(k_a), k_tr=$(k_tr)", lw=1.5)
    end
end

Plots.savefig(p, joinpath(results_dir, "plot_simulate.png"))
println("\nSaved: $(joinpath(results_dir, "plot_simulate.png"))")

# Verify stability: check for NaN/Inf and negative concentrations
println("\nStability check...")
all_ok = true
for k_a in [K_A_LO, K_A_HI]
    for k_tr in [K_TR_LO, K_TR_HI]
        for cl in [CL_LO, CL_HI]
            for qd in [Q_D_LO, Q_D_HI]
                θ = reshape(Float32[k_a, k_tr, cl, qd], N_PARAMS_DYN, 1)
                u = zeros(Float32, 5, 1)
                for step in 1:N_STEPS
                    u = integrate_cpu(u, θ, ACTION_HI, DT, N_SUBSTEPS)
                end
                if any(isnan, u) || any(isinf, u) || any(x -> x < -1f-6, u)
                    @printf("  UNSTABLE: k_a=%.1f k_tr=%.1f CL=%.1f Q_d=%.1f -> u=%s\n",
                            k_a, k_tr, cl, qd, u[:, 1])
                    global all_ok = false
                end
            end
        end
    end
end
println(all_ok ? "All parameter corners stable." : "WARNING: instabilities detected!")
