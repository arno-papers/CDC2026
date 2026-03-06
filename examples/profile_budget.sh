#!/usr/bin/env bash
set -euo pipefail
# Profile maximum ODE budget C that fits on GPU.
# Spawns a separate Julia process per probe so OOM kills only the child.
#
# Usage:
#   bash examples/profile_budget.sh [example] [grad_accum] [budget_lo] [budget_hi]
#
# Example:
#   bash examples/profile_budget.sh monod 1 500000 20000000
#   bash examples/profile_budget.sh monod 16 500000 40000000

EXAMPLE="${1:-monod}"
GRAD_ACCUM="${2:-1}"
BUDGET_LO="${3:-500000}"
BUDGET_HI="${4:-20000000}"
JULIA="${JULIA_BIN:-julia}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_SCRIPT="${SCRIPT_DIR}/profile_budget.jl"
RESULTS_DIR="${SCRIPT_DIR}/${EXAMPLE}/results"

echo "=== Budget Profiling: ${EXAMPLE} ==="
echo "grad_accum = ${GRAD_ACCUM}"
echo "Search range: [${BUDGET_LO}, ${BUDGET_HI}]"
echo "Julia: ${JULIA}"
echo

mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/profile_budget.txt"
: > "${LOG}"  # truncate

probe() {
    local C="$1"
    # Run in subprocess; capture output, check exit code
    local out rc
    set +e
    out=$("${JULIA}" --project=. "${PROBE_SCRIPT}" \
        "example=${EXAMPLE}" "budget=${C}" "grad_accum=${GRAD_ACCUM}" 2>&1)
    rc=$?
    set -e
    echo "${out}" | tail -5
    echo "${out}" >> "${LOG}"
    echo "---" >> "${LOG}"
    return ${rc}
}

# ---- Coarse phase: doubling ----
echo "--- Coarse phase (doubling) ---"
last_ok=0
first_fail=${BUDGET_HI}
C=${BUDGET_LO}

while (( C <= BUDGET_HI )); do
    echo -n "Trying C=${C} ... "
    if probe "${C}"; then
        last_ok=${C}
        C=$(( C * 2 ))
    else
        echo "  -> FAILED (OOM or error)"
        first_fail=${C}
        break
    fi
done

if (( last_ok == 0 )); then
    echo
    echo "No budget fits. Try reducing budget_lo or increasing grad_accum."
    exit 1
fi

# ---- Fine phase: binary search ----
echo
echo "--- Fine phase (binary search in [${last_ok}, ${first_fail}]) ---"
lo=${last_ok}
hi=${first_fail}

# Stop when bracket is <5% of lo or <50k
while (( hi - lo > 50000 && hi - lo > lo / 20 )); do
    mid=$(( (lo + hi) / 2 ))
    echo -n "Trying C=${mid} ... "
    if probe "${mid}"; then
        lo=${mid}
    else
        echo "  -> FAILED"
        hi=${mid}
    fi
done

# ---- Report ----
echo
echo "============================================================"
echo "RESULT: Max feasible budget for ${EXAMPLE} (grad_accum=${GRAD_ACCUM})"
echo "============================================================"
# Use Julia to print the allocation for the final budget
"${JULIA}" --project=. -e "
include(\"src/budget.jl\")
L, M, B = allocate_budget(${lo})
B_micro = B ÷ ${GRAD_ACCUM}
using Printf
@printf(\"  C_max        = %d\n\", ${lo})
@printf(\"  L            = %d\n\", L)
@printf(\"  M            = %d\n\", M)
@printf(\"  B            = %d\n\", B)
@printf(\"  B_micro      = %d\n\", B_micro)
@printf(\"  grad_accum   = %d\n\", ${GRAD_ACCUM})
"
echo "============================================================"
echo
echo "Results log: ${LOG}"
