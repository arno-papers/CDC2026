#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Submit budget profiling to VSC as a batch job.
#
# Usage:
#   EXAMPLE=monod ./scripts/submit_profile_budget.sh
#   EXAMPLE=haldane WALLTIME=2:00:00 ./scripts/submit_profile_budget.sh
# ============================================================================

EXAMPLE="${EXAMPLE:?Set EXAMPLE (e.g. monod, haldane, weibull)}"

REMOTE_HOST="${REMOTE_HOST:-tier2}"
SSH_SOCKET="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"
SSH_OPTS=(-S "${SSH_SOCKET}" -o BatchMode=yes)

LOCAL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-CDC2026}"
PARTITION="${PARTITION:-gpu_a100}"
ACCOUNT="${ACCOUNT:-lp_dad}"
JULIA_VERSION="${JULIA_VERSION:-1.12.5}"
WALLTIME="${WALLTIME:-4:00:00}"

ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" >/dev/null

remote_vsc_data="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "source /etc/profile 2>/dev/null; printf %s \"\$VSC_DATA\"")"
if [[ -z "${remote_vsc_data}" ]]; then
  echo "ERROR: remote VSC_DATA is empty" >&2
  exit 1
fi
remote_repo_dir="${remote_vsc_data}/${REMOTE_SUBDIR}"

echo "[profile] Example:     ${EXAMPLE}"
echo "[profile] Remote repo: ${remote_repo_dir}"
echo "[profile] Walltime:    ${WALLTIME}"

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

# Bootstrap Julia
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${JULIA_VERSION}" <<'EOF'
set -euo pipefail
cd "$1"
if [[ -z "${VSC_DATA:-}" ]]; then export VSC_DATA="$(dirname "$1")"; fi
./vsc/bootstrap_julia.sh
export JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026"
"$VSC_DATA/software/julia/$2/bin/julia" --project=. -e 'using Pkg; Pkg.instantiate()'
EOF

# Submit
job_id="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- \
  "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" \
  "${JULIA_VERSION}" "${EXAMPLE}" <<'SUBMIT'
set -euo pipefail
remote_repo_dir="$1"; partition="$2"; walltime="$3"; account="$4"
julia_version="$5"; example="$6"

if [[ -z "${VSC_DATA:-}" ]]; then export VSC_DATA="$(dirname "$1")"; fi
julia_bin="$VSC_DATA/software/julia/${julia_version}/bin/julia"
cd "${remote_repo_dir}"
mkdir -p logs

sbatch --parsable \
  --job-name="profile-${example}" \
  --clusters=wice \
  --partition="${partition}" \
  --account="${account}" \
  --nodes=1 --ntasks=1 --cpus-per-task=18 --gpus-per-node=1 \
  --mem=120G \
  --time="${walltime}" \
  --output="logs/profile-${example}-%j.out" \
  --error="logs/profile-${example}-%j.err" \
  --export=ALL,JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026" \
  --wrap="${julia_bin} --project=. scripts/profile_budget.jl example=${example}"
SUBMIT
)"

job_id="${job_id%%;*}"
echo "[profile] Submitted job ${job_id}"
echo "[profile] Logs:  ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"bash -lc 'tail -f ${remote_repo_dir}/logs/profile-${EXAMPLE}-${job_id}.out'\""
