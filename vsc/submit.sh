#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Unified VSC submit script.
#
# Usage:
#   EXAMPLE=monod TASK=train ./vsc/submit.sh
#   EXAMPLE=monod TASK=eval_bim WALLTIME=2:00:00 ./vsc/submit.sh
#   EXAMPLE=monod TASK=optimize_static WALLTIME=4:00:00 ./vsc/submit.sh
#   EXAMPLE=monod TASK=eval_spce WALLTIME=4:00:00 ./vsc/submit.sh
#   DRY_RUN=1 EXAMPLE=monod TASK=train ./vsc/submit.sh  # cost estimate only
#
# SCRIPT_ARGS passes extra arguments to the Julia script, e.g.:
#   SCRIPT_ARGS="n_iters=500 seed=42" EXAMPLE=monod TASK=train ./vsc/submit.sh
# ============================================================================

EXAMPLE="${EXAMPLE:?Set EXAMPLE (e.g. monod)}"
TASK="${TASK:?Set TASK (e.g. train, eval_bim, optimize_static, eval_spce, eval_posterior)}"

REMOTE_HOST="${REMOTE_HOST:-tier2}"
SSH_SOCKET="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"
SSH_OPTS=(-S "${SSH_SOCKET}" -o BatchMode=yes)

LOCAL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-CDC2026}"
PARTITION="${PARTITION:-gpu_a100}"
ACCOUNT="${ACCOUNT:-intro_vsc32553}"
JULIA_VERSION="${JULIA_VERSION:-1.12.5}"
DRY_RUN="${DRY_RUN:-0}"
SCRIPT_ARGS="${SCRIPT_ARGS:-}"

# Task-specific defaults for walltime
case "${TASK}" in
  train)        WALLTIME="${WALLTIME:-24:00:00}" ;;
  optimize_*)   WALLTIME="${WALLTIME:-4:00:00}" ;;
  eval_*)       WALLTIME="${WALLTIME:-4:00:00}" ;;
  *)            WALLTIME="${WALLTIME:-4:00:00}" ;;
esac

RESULTS_BASENAME="${RESULTS_BASENAME:-${EXAMPLE}-${TASK}-$(date +%Y%m%d-%H%M%S)}"

source_branch="$(git -C "${LOCAL_REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
source_commit="$(git -C "${LOCAL_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Verify SSH connection
ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" >/dev/null

# Resolve remote paths
remote_vsc_data="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "source /etc/profile 2>/dev/null; printf %s \"\$VSC_DATA\"")"
remote_vsc_scratch="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "source /etc/profile 2>/dev/null; printf %s \"\$VSC_SCRATCH\"")"
if [[ -z "${remote_vsc_data}" ]]; then
  echo "[submit] ERROR: remote VSC_DATA is empty" >&2
  exit 1
fi
if [[ -z "${remote_vsc_scratch}" ]]; then
  remote_vsc_scratch="${remote_vsc_data}"
fi

RUN_ROOT="${RUN_ROOT:-${remote_vsc_scratch}/CDC2026-runs}"
PERSIST_ROOT="${PERSIST_ROOT:-${remote_vsc_data}/CDC2026-runs}"
remote_repo_dir="${remote_vsc_data}/${REMOTE_SUBDIR}"

echo "[submit] Example:     ${EXAMPLE}"
echo "[submit] Task:        ${TASK}"
echo "[submit] Local repo:  ${LOCAL_REPO_DIR}"
echo "[submit] Remote repo: ${remote_repo_dir}"
echo "[submit] Walltime:    ${WALLTIME}"
echo "[submit] Result tag:  ${RESULTS_BASENAME}"
echo "[submit] Source:      ${source_branch}@${source_commit}"
if [[ -n "${SCRIPT_ARGS}" ]]; then
  echo "[submit] Script args: ${SCRIPT_ARGS}"
fi

# Sync code
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'mkdir -p \"${remote_repo_dir}\"'"

rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.ruff_cache/' \
  --exclude='examples/*/results/' \
  --exclude='.remote_logs/' \
  -e "ssh -S ${SSH_SOCKET}" \
  "${LOCAL_REPO_DIR}/" "${REMOTE_HOST}:${remote_repo_dir}/"

# Bootstrap Julia and instantiate
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${JULIA_VERSION}" <<'EOF'
set -euo pipefail
remote_repo_dir="$1"
julia_version="$2"
cd "${remote_repo_dir}"
if [[ -z "${VSC_DATA:-}" ]]; then
  export VSC_DATA="$(dirname "${remote_repo_dir}")"
fi
./vsc/bootstrap_julia.sh
export JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026"
"$VSC_DATA/software/julia/${julia_version}/bin/julia" --project=. -e 'using Pkg; Pkg.instantiate()'
EOF

# Cost estimate
echo "[submit] Estimated credits:"
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" <<'EOF'
set -euo pipefail
cd "$1"
sam-quote sbatch --partition="$2" --time="$3" --account="$4" vsc/job.slurm || true
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[submit] DRY_RUN=1, stopping before sbatch"
  exit 0
fi

# Submit
job_submit_output="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- \
  "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" "${JULIA_VERSION}" \
  "${RUN_ROOT}" "${PERSIST_ROOT}" "${RESULTS_BASENAME}" \
  "${EXAMPLE}" "${TASK}" "${SCRIPT_ARGS}" \
  "${source_branch}" "${source_commit}" <<'EOF'
set -euo pipefail
remote_repo_dir="$1"
partition="$2"
walltime="$3"
account="$4"
julia_version="$5"
run_root="$6"
persist_root="$7"
results_basename="$8"
example="$9"
task="${10}"
script_args="${11}"
source_branch="${12}"
source_commit="${13}"

cd "${remote_repo_dir}"
if [[ -z "${VSC_DATA:-}" ]]; then
  export VSC_DATA="$(dirname "${remote_repo_dir}")"
fi
mkdir -p logs

sbatch --parsable \
  --partition="${partition}" \
  --time="${walltime}" \
  --account="${account}" \
  --export=ALL,VSC_DATA="$VSC_DATA",REPO_DIR="${remote_repo_dir}",JULIA_VERSION="${julia_version}",JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026",RUN_ROOT="${run_root}",PERSIST_ROOT="${persist_root}",RESULTS_BASENAME="${results_basename}",EXAMPLE="${example}",TASK="${task}",SCRIPT_ARGS="${script_args}",SOURCE_BRANCH="${source_branch}",SOURCE_COMMIT="${source_commit}" \
  vsc/job.slurm
EOF
)"

job_id="${job_submit_output%%;*}"
if [[ -z "${job_id}" ]]; then
  echo "[submit] ERROR: empty sbatch output: ${job_submit_output}" >&2
  exit 1
fi

echo "[submit] Submitted job ${job_id} on ${PARTITION}"
echo "[submit] Queue:      ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"squeue -M wice -j ${job_id}\""
echo "[submit] Accounting: ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"sacct -M wice -j ${job_id} --format=JobID,State,Elapsed,ReqGRES,AllocTRES%80\""
echo "[submit] Logs:       ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"bash -lc 'tail -f ${remote_repo_dir}/logs/cdc2026-${job_id}.out'\""
echo "[submit] Results:    ${PERSIST_ROOT}/${RESULTS_BASENAME}"
