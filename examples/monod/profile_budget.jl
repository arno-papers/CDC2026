# Wrapper so VSC submit infrastructure can call: EXAMPLE=monod TASK=profile_budget
# Delegates to the shell-based profiler which spawns safe subprocesses per budget probe.
#
# Pass SCRIPT_ARGS="grad_accum=16 budget_lo=500000 budget_hi=40000000" to customize.

function _get_arg(args, key; default="")
    prefix = key * "="
    for a in args
        startswith(a, prefix) && return split(a, "=", limit=2)[2]
    end
    return default
end

grad_accum = _get_arg(ARGS, "grad_accum"; default="1")
budget_lo  = _get_arg(ARGS, "budget_lo"; default="500000")
budget_hi  = _get_arg(ARGS, "budget_hi"; default="20000000")

script = joinpath(@__DIR__, "..", "profile_budget.sh")
run(`bash $script monod $grad_accum $budget_lo $budget_hi`)
