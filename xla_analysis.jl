#=
XLA / HLO analysis for the DADS training step.

This script is analogous to `profiler.jl`, but instead of collecting execution
traces it dumps the post-optimization XLA HLO and runs XLA analyses.

It uses Reactant functionality:
  - `Reactant.Compiler.compile_xla` to obtain the XLA HLO module(s)
  - `Reactant.XLA.cost_analysis` for a module-level cost estimate
  - `Reactant.XLA.GPUPerformanceModel` to estimate per-instruction runtimes (GPU)

Run with Julia 1.11:
  julia +1.11 --project=. xla_analysis.jl

Options (CLI kwargs):
  - `outdir=...`   Output directory for dumped HLO files
  - `stage=...`    `after` (default) | `before` | `both`
  - `style=...`    `default` (default) | `compact` | `canonical` | `fingerprint` | `module_fingerprint`
  - `top=...`      How many instructions to print for the GPU perf model (default: 30)
=#

using Lux, Reactant, Random
using Optimisers
using Printf

const LuxReactantExt = Base.get_extension(Lux, :LuxReactantExt)

include("common.jl")

function parse_kwarg(args, key; default=nothing)
    prefix = key * "="
    for a in args
        if startswith(a, prefix)
            return split(a, "=", limit=2)[2]
        end
    end
    return default
end

function parse_int(args, key; default::Int)
    v = parse_kwarg(args, key; default=nothing)
    return v === nothing ? default : parse(Int, v)
end

function parse_stage(args; default::Symbol=:after)
    v = parse_kwarg(args, "stage"; default=nothing)
    v === nothing && return default
    v = lowercase(v)
    v in ("after", "post", "xla") && return :after
    v in ("before", "pre") && return :before
    v == "both" && return :both
    error("Unknown stage=$v (expected after|before|both)")
end

function parse_style(args; default::Symbol=:default)
    v = parse_kwarg(args, "style"; default=nothing)
    v === nothing && return default
    v = lowercase(v)
    v in ("default", "full") && return :default
    v in ("compact", "short") && return :compact
    v in ("canonical", "canon") && return :canonical
    v in ("fingerprint",) && return :fingerprint
    v in ("module_fingerprint", "modulefingerprint", "mf") && return :module_fingerprint
    error("Unknown style=$v (expected default|compact|canonical|fingerprint|module_fingerprint)")
end

function ioctx_for_style(io::IO, style::Symbol)
    style === :default && return io
    style === :compact && return IOContext(io, :compact => true)
    style === :canonical && return IOContext(io, :canonical => true)
    style === :fingerprint && return IOContext(io, :fingerprint => true)
    style === :module_fingerprint && return IOContext(io, :module_fingerprint => true)
    return io
end

function make_step_data(rng, xdev)
    n_denom = L_CONTRASTIVE + 1
    theta_full = sample_θ_full(rng, n_denom, GRAD_BATCH) |> xdev
    theta_N_numer = sample_θ_N(rng, M_NUISANCE, GRAD_BATCH) |> xdev

    u0 = zeros(Float32, 3, 1, 1)
    u0[1, 1, 1] = 3.0f0
    u0[2, 1, 1] = 0.25f0
    u0[3, 1, 1] = 7.0f0
    u0 = u0 |> xdev

    input_buffer = zeros(Float32, 2, N_STEPS, GRAD_BATCH) |> xdev
    observations = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
    designs = zeros(Float32, N_STEPS, GRAD_BATCH) |> xdev
    eps = randn(rng, Float32, N_STEPS, GRAD_BATCH) |> xdev
    ll_denom = zeros(Float32, n_denom, GRAD_BATCH) |> xdev
    ll_numer = zeros(Float32, M_NUISANCE, GRAD_BATCH) |> xdev

    return (theta_full, theta_N_numer, u0, input_buffer, observations, designs, eps, ll_denom, ll_numer)
end

function grad_and_step(model, ps, st, opt_state, data; is_sharded::Bool=false)
    # Equivalent to the core of `Lux.Training.single_train_step!` for the Reactant backend,
    # but avoids Lux's `get_device(...)` dispatch which is not allowed inside a compile trace.
    LuxReactantExt === nothing && error("LuxReactantExt not loaded; ensure Reactant is available")
    return LuxReactantExt.compute_gradients_internal_and_step!(
        targeted_spce_loss,
        model,
        data,
        ps,
        st,
        opt_state,
        nothing,
        is_sharded,
    )
end

function as_vec(x)
    return x isa AbstractVector ? x : [x]
end

function dump_hlo_modules(outdir::AbstractString, tag::AbstractString, hlo_modules, style::Symbol)
    mkpath(outdir)
    for (i, hlo) in enumerate(as_vec(hlo_modules))
        path = joinpath(outdir, "$(tag)_$(i).hlo")
        open(path, "w") do io
            show(ioctx_for_style(io, style), hlo)
        end
    end
    return nothing
end

function print_cost_analysis(io::IO, client, hlo_modules)
    for (i, hlo) in enumerate(as_vec(hlo_modules))
        println(io, "\n--- Cost analysis (module $i) ---")
        try
            show(io, MIME("text/plain"), Reactant.XLA.cost_analysis(client, hlo))
            println(io)
        catch err
            println(io, "cost_analysis failed: ", err)
        end
    end
    return nothing
end

function instruction_line(inst)
    return sprint(io -> show(IOContext(io, :compact => true), inst))
end

function run_gpu_perf_model(io::IO, device, hlo_modules; top::Int)
    device === nothing && return nothing

    # Only supported for streaming-executor backends (CUDA at the moment).
    desc = try
        Reactant.XLA.StreamExecutorDeviceDescription(device)
    catch
        return nothing
    end

    ctx = Reactant.MLIR.IR.Context(Reactant.registry[], false)
    @ccall Reactant.MLIR.API.mlir_c.RegisterDialects(
        ctx::Reactant.MLIR.API.MlirContext
    )::Cvoid

    Reactant.MLIR.IR.activate!(ctx)
    try
        model = Reactant.XLA.GPUPerformanceModel(ctx, desc)

        for (mi, hlo) in enumerate(as_vec(hlo_modules))
            println(io, "\n--- GPU performance model (module $mi) ---")
            try
                model(hlo) # run module-wide analysis (required by some estimates)
            catch err
                println(io, "module analysis failed: ", err)
                continue
            end

            insts = hlo.entry_computation.instructions
            rows = Vector{Tuple{Int64,Any,Reactant.XLA.EstimateRunTimeData}}()

            for (ii, inst) in enumerate(insts)
                est = try
                    model(inst)
                catch
                    continue
                end
                push!(rows, (Int64(ii), inst, est))
            end

            if isempty(rows)
                println(io, "(no instruction estimates)")
                continue
            end

            sort!(rows; by=r -> r[3].execution_time_ns, rev=true)
            nshow = min(top, length(rows))

            println(io, "Top $(nshow) instructions by estimated execution time:")
            for k in 1:nshow
                ii, inst, est = rows[k]
                @printf(
                    io,
                    "[%4d] %9.3f ms  opcode=%s  flops=%d  read=%s  write=%s\n",
                    ii,
                    est.execution_time_ns / 1e6,
                    sprint(show, inst.opcode),
                    est.flops,
                    Base.format_bytes(est.bytes_read),
                    Base.format_bytes(est.bytes_written),
                )
                println(io, "       ", instruction_line(inst))
            end
        end
    finally
        Reactant.MLIR.IR.deactivate!(ctx)
    end

    return nothing
end

function analyze_once(; before_xla_optimizations::Bool, outdir::AbstractString, style::Symbol, top::Int)
    println("\nCompiling (before_xla_optimizations=$(before_xla_optimizations))...")

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, policy)

    xdev = reactant_device()
    ps_ra = ps |> xdev
    st_ra = st |> xdev

    data = make_step_data(rng, xdev)
    # Use Lux's Reactant-compatible optimizer wrapper (includes PJRT scalars for hyperparams).
    train_state = Lux.Training.TrainState(policy, ps_ra, st_ra, Adam(0.001f0))
    opt_state = train_state.optimizer_state

    mod, exec, hlo_modules, _mlir_fn_res, device, client, _module_string =
        Reactant.Compiler.compile_xla(
            grad_and_step,
            (policy, ps_ra, st_ra, opt_state, data);
            before_xla_optimizations=before_xla_optimizations,
            client=nothing,
        )

    tag = before_xla_optimizations ? "xla_hlo_before" : "xla_hlo_after"
    dump_hlo_modules(outdir, tag, hlo_modules, style)
    println("HLO dumped to: ", outdir)

    print_cost_analysis(stdout, client, hlo_modules)
    run_gpu_perf_model(stdout, device, hlo_modules; top=top)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n=== XLA / HLO Analysis (Reactant) ===\n")

    stage = parse_stage(ARGS; default=:after)
    style = parse_style(ARGS; default=:default)
    top = parse_int(ARGS, "top"; default=30)

    outdir = parse_kwarg(ARGS, "outdir"; default=joinpath(tempdir(), "dads_xla_analysis"))

    Reactant.set_default_backend("gpu")

    if stage in (:after, :both)
        analyze_once(; before_xla_optimizations=false, outdir=outdir, style=style, top=top)
    end
    if stage in (:before, :both)
        analyze_once(; before_xla_optimizations=true, outdir=outdir, style=style, top=top)
    end
end
