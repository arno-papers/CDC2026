# Eager, CUDA.jl-based reproduction of the first training loss evaluation.
#
# This file is self-contained (no `common.jl`) and avoids Reactant/XLA so you can
# step through the first forward loss pass in the REPL / debugger.
#
# CLI:
#   julia +1.11 --project=. training_eager_cuda_first_iter.jl seed=0
#
# REPL:
#   using Debugger
#   @enter include("training_eager_cuda_first_iter.jl")

using Lux
using Random
using CUDA

CUDA.allowscalar(false)

if Base.JLOptions().can_inline == 0
    error("CUDA kernels won't compile with --inline=no. Restart Julia without --inline=no (Base.JLOptions().can_inline must be 1).")
end

# -----------------------------------------------------------------------------
# Experiment constants
#
# This file is a self-contained, eager (CUDA.jl) reproduction of a single forward
# loss evaluation. Hyperparameters (L, M, B) use the same budget-based scheme
# as `common.jl`.
# -----------------------------------------------------------------------------

const N_STEPS = 10
const DT = 1.0f0            # Total time per control interval
const N_SUBSTEPS = 500      # Integration substeps per control interval

# Training budget allocation (see Appendix, Section "Computational Budget Trade-offs")
# Budget: C_traj = B * (L + 3) trajectory rollouts per optimizer update.
# Optimal (L, B) minimizes MSE proxy: 1/B + λ/(L+1)² subject to budget constraint.
const ODE_BUDGET_TRAJ = 4144

const (L_CONTRASTIVE, GRAD_BATCH) = let
    C = ODE_BUDGET_TRAJ
    λ = 1.0  # equal weight on variance (1/B) vs squared bias (1/(L+1)²)
    best_L, best_B, best_obj = 1, fld(C, 4), Inf
    for L in 1:(C - 3)
        B = fld(C, L + 3)
        B < 1 && break
        obj = 1.0/B + λ/(L+1)^2
        if obj < best_obj
            best_obj, best_L, best_B = obj, L, B
        end
    end
    (best_L, best_B)
end

# Nuisance samples: ODE-free (only affect observation model), so can be large
const M_NUISANCE = min(4096, max(512, 32 * (L_CONTRASTIVE + 1)))

const MU_MAX_LO, MU_MAX_HI = 0.3f0, 0.5f0
const K_S_LO, K_S_HI = 0.3f0, 0.6f0
const SIGMA_LO, SIGMA_HI = 0.05f0, 0.15f0

# -----------------------------------------------------------------------------
# Bioreactor dynamics (RK4 integrator)
# -----------------------------------------------------------------------------

function bioreactor_dynamics(u, theta, q_in)
    # Layout: (state_dim, ...batch_dims)
    # Keep the batch dimensions intact (e.g. (3, B) or (3, n_denom, B)).
    c_s = selectdim(u, 1, 1)
    c_x = selectdim(u, 1, 2)
    v = selectdim(u, 1, 3)

    mu_max = selectdim(theta, 1, 1)
    k_s = selectdim(theta, 1, 2)

    mu = @. mu_max * c_s / (k_s + c_s)
    sigma = @. mu / 0.777f0

    du1 = @. -sigma * c_x + (q_in ./ v) * (50.0f0 - c_s)
    du2 = @. mu * c_x - (q_in ./ v) * c_x
    du3 = @. 0.0f0 * v + q_in

    du = similar(u)
    selectdim(du, 1, 1) .= du1
    selectdim(du, 1, 2) .= du2
    selectdim(du, 1, 3) .= du3
    return du
end

function rk4_step(u, theta, q_in, dt)
    k1 = bioreactor_dynamics(u, theta, q_in)
    k2 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k1, theta, q_in)
    k3 = bioreactor_dynamics(u .+ 0.5f0 * dt .* k2, theta, q_in)
    k4 = bioreactor_dynamics(u .+ dt .* k3, theta, q_in)
    return u .+ (dt / 6.0f0) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
end

function integrate(u, theta, q_in, dt, n_substeps)
    dt_sub = dt / n_substeps
    for _ in 1:n_substeps
        u = rk4_step(u, theta, q_in, dt_sub)
    end
    return u
end

# -----------------------------------------------------------------------------
# Positional encoding (GPU-cached)
# -----------------------------------------------------------------------------

function sinusoidal_pe_cpu(seq_len::Int)
    position = reshape(Float32.(0:(seq_len - 1)), 1, seq_len)
    div_term = exp.(Float32.(0:2:31) .* -(log(1000.0f0) / 32.0f0))
    angles = div_term * position
    pe = zeros(Float32, 32, seq_len)
    pe[1:2:end, :] .= sin.(angles)
    pe[2:2:end, :] .= cos.(angles[1:16, :])
    return pe
end

const _PE_CACHE = Dict{Int, Any}()
function sinusoidal_pe(seq_len::Int)
    return get!(_PE_CACHE, seq_len) do
        cu(sinusoidal_pe_cpu(seq_len))
    end
end

# -----------------------------------------------------------------------------
# Policy (Lux)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Sampling (copied from common.jl)
# -----------------------------------------------------------------------------

function sample_theta_full(rng, n_denom::Int, b::Int)
    theta = rand(rng, Float32, 3, n_denom, b)
    @views begin
        theta[1, :, :] .= MU_MAX_LO .+ (MU_MAX_HI - MU_MAX_LO) .* theta[1, :, :]
        theta[2, :, :] .= K_S_LO .+ (K_S_HI - K_S_LO) .* theta[2, :, :]
        theta[3, :, :] .= SIGMA_LO .+ (SIGMA_HI - SIGMA_LO) .* theta[3, :, :]
    end
    return theta
end

function sample_theta_N(rng, m::Int, b::Int)
    theta = rand(rng, Float32, m, b)
    theta .= SIGMA_LO .+ (SIGMA_HI - SIGMA_LO) .* theta
    return theta
end

# -----------------------------------------------------------------------------
# Main (top-level) code: this is the "iteration 1" forward loss pass
# -----------------------------------------------------------------------------

seed = 0
rng = Random.MersenneTwister(0)

ps, st = Lux.setup(rng, policy)
ps = cu(ps)
st = cu(st)

n_denom = L_CONTRASTIVE + 1

theta_full = sample_theta_full(rng, n_denom, GRAD_BATCH) |> cu
theta_N_numer = sample_theta_N(rng, M_NUISANCE, GRAD_BATCH) |> cu

u0 = zeros(Float32, 3, 1, 1)
u0[1, 1, 1] = 3.0f0
u0[2, 1, 1] = 0.25f0
u0[3, 1, 1] = 7.0f0
u0 = cu(u0)
input_buffer = zeros(Float32, 2, N_STEPS, GRAD_BATCH) |> cu
observations = zeros(Float32, N_STEPS, GRAD_BATCH) |> cu
designs = zeros(Float32, N_STEPS, GRAD_BATCH) |> cu
eps = randn(rng, Float32, N_STEPS, GRAD_BATCH) |> cu
ll_denom = zeros(Float32, n_denom, GRAD_BATCH) |> cu
ll_numer = zeros(Float32, M_NUISANCE, GRAD_BATCH) |> cu

ll_denom .= 0.0f0
ll_numer .= 0.0f0

theta_T_true = theta_full[1:2, 1:1, :]
sigma_true = theta_full[3:3, 1, :]

b = GRAD_BATCH
u = repeat(u0, 1, 1, b)

for step in 1:N_STEPS
    global st, u

    action, st = policy(input_buffer, ps, st)
    q_in = action
    @views designs[step:step, :] .= q_in

    u = integrate(u, theta_T_true, q_in, DT, N_SUBSTEPS)

    obs = @view u[1:1, 1, :]
    noise = eps[step:step, :]
    y_noisy = obs .+ sigma_true .* noise

    @views begin
        observations[step:step, :] .= y_noisy
        input_buffer[1:1, step, :] .= y_noisy
        input_buffer[2:2, step, :] .= q_in
    end
end

theta_T_denom = theta_full[1:2, :, :]
sigma2_denom = (theta_full[3, :, :]) .^ 2
u_denom = repeat(u0, 1, n_denom, b)

for step in 1:N_STEPS
    global u_denom, ll_denom

    q_step = designs[step:step, :]
    u_denom = integrate(u_denom, theta_T_denom, q_step, DT, N_SUBSTEPS)

    pred_obs = @view u_denom[1, :, :]
    actual_obs = observations[step:step, :]
    residual = actual_obs .- pred_obs
    ll_denom .-= 0.5f0 .* (residual .^ 2 ./ sigma2_denom .+ log.(sigma2_denom))
end

sigma2_numer = theta_N_numer .^ 2
u_numer = repeat(u0, 1, 1, b)

for step in 1:N_STEPS
    global u_numer, ll_numer

    q_step = designs[step:step, :]
    u_numer = integrate(u_numer, theta_T_true, q_step, DT, N_SUBSTEPS)

    pred_obs = @view u_numer[1:1, 1, :]
    actual_obs = observations[step:step, :]
    residual2 = (actual_obs .- pred_obs) .^ 2
    ll_numer .-= 0.5f0 .* (residual2 ./ sigma2_numer .+ log.(sigma2_numer))
end

ll_max_num = maximum(ll_numer; dims = 1)
lse_num = ll_max_num .+ log.(sum(exp.(ll_numer .- ll_max_num); dims = 1))
log_numerator = lse_num .- log(Float32(M_NUISANCE))

ll_max_den = maximum(ll_denom; dims = 1)
lse_den = ll_max_den .+ log.(sum(exp.(ll_denom .- ll_max_den); dims = 1))
log_denominator = lse_den .- log(Float32(n_denom))

loss_per_episode = -(log_numerator .- log_denominator)
loss = sum(loss_per_episode) / Float32(b)

CUDA.synchronize()

result = (;
    loss,
    ps,
    st,
    theta_full,
    theta_N_numer,
    u0,
    input_buffer,
    observations,
    designs,
    eps,
    ll_denom,
    ll_numer,
)

if abspath(PROGRAM_FILE) == @__FILE__
    println("loss = ", loss)
end
