#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-tier2}"
SSH_SOCKET="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"
SSH_OPTS=(-S "${SSH_SOCKET}" -o BatchMode=yes)

LOCAL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-CDC2026}"
PARTITION="${PARTITION:-gpu_a100_debug}"
WALLTIME="${WALLTIME:-00:30:00}"
ACCOUNT="${ACCOUNT:-intro_vsc32553}"
JULIA_VERSION="${JULIA_VERSION:-1.12.4}"

ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" >/dev/null

remote_vsc_data="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "source /etc/profile 2>/dev/null; printf %s \"\$VSC_DATA\"")"
if [[ -z "${remote_vsc_data}" ]]; then
  echo "[submit] ERROR: remote VSC_DATA is empty" >&2
  exit 1
fi
remote_repo_dir="${remote_vsc_data}/${REMOTE_SUBDIR}"

echo "[submit] Local repo:  ${LOCAL_REPO_DIR}"
echo "[submit] Remote repo: ${remote_repo_dir}"

ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'mkdir -p "${remote_repo_dir}"'"

rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.ruff_cache/' \
  --exclude='results/' \
  --exclude='artifacts/' \
  -e "ssh -S ${SSH_SOCKET}" \
  "${LOCAL_REPO_DIR}/" "${REMOTE_HOST}:${remote_repo_dir}/"

ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${JULIA_VERSION}" <<'EOF'
set -euo pipefail
remote_repo_dir="$1"
julia_version="$2"

cd "${remote_repo_dir}"
if [[ -z "${VSC_DATA:-}" ]]; then
  export VSC_DATA="$(dirname "${remote_repo_dir}")"
fi

./vsc_scripts/bootstrap_julia_1_12.sh
"$VSC_DATA/software/julia/${julia_version}/bin/julia" --version
EOF

job_submit_output="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" "${JULIA_VERSION}" <<'EOF'
set -euo pipefail
remote_repo_dir="$1"
partition="$2"
walltime="$3"
account="$4"
julia_version="$5"

cd "${remote_repo_dir}"
if [[ -z "${VSC_DATA:-}" ]]; then
  export VSC_DATA="$(dirname "${remote_repo_dir}")"
fi
mkdir -p logs

sbatch --parsable \
  --partition="${partition}" \
  --time="${walltime}" \
  --account="${account}" \
  --export=ALL,VSC_DATA="$VSC_DATA",REPO_DIR="${remote_repo_dir}",JULIA_VERSION="${julia_version}",JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026" \
  vsc_scripts/smoke_wice_a100_debug.slurm
EOF
)"

job_id="${job_submit_output%%;*}"
if [[ -z "${job_id}" ]]; then
  echo "[submit] ERROR: empty sbatch output: ${job_submit_output}" >&2
  exit 1
fi

echo "[submit] Submitted job ${job_id} on ${PARTITION}"
echo "[submit] Follow logs: ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"bash -lc 'tail -f ${remote_repo_dir}/logs/cdc2026-smoke-${job_id}.out'\""
echo "[submit] Queue:       ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"squeue -M wice -j ${job_id}\""
echo "[submit] Accounting:  ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"sacct -M wice -j ${job_id} --format=JobID,State,Elapsed,ReqGRES,AllocTRES%80\""
