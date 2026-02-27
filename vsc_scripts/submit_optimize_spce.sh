#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-tier2}"
SSH_SOCKET="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"
SSH_OPTS=(-S "${SSH_SOCKET}" -o BatchMode=yes)

LOCAL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-CDC2026}"
PARTITION="${PARTITION:-gpu_a100}"
WALLTIME="${WALLTIME:-4:00:00}"
ACCOUNT="${ACCOUNT:-intro_vsc32553}"
JULIA_VERSION="${JULIA_VERSION:-1.12.4}"

N_ITERS="${N_ITERS:-300}"
N_RESTARTS="${N_RESTARTS:-4}"
B="${B:-64}"
L="${L:-15}"
M="${M:-128}"
N_SUBSTEPS="${N_SUBSTEPS:-100}"
LR_MAX="${LR_MAX:-0.01}"
LR_MIN="${LR_MIN:-1e-5}"
WARMUP="${WARMUP:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SEED="${SEED:-0}"
RESULTS_BASENAME="${RESULTS_BASENAME:-spce-opt-$(date +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

source_branch="$(git -C "${LOCAL_REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
source_commit="$(git -C "${LOCAL_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || true)"
if [[ -z "${source_branch}" ]]; then
  source_branch="unknown"
fi
if [[ -z "${source_commit}" ]]; then
  source_commit="unknown"
fi

ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" >/dev/null

remote_vsc_data="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'printf %s \"\$VSC_DATA\"'")"
remote_vsc_scratch="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'printf %s \"\$VSC_SCRATCH\"'")"
if [[ -z "${remote_vsc_data}" ]]; then
  echo "[submit-spce-opt] ERROR: remote VSC_DATA is empty" >&2
  exit 1
fi
if [[ -z "${remote_vsc_scratch}" ]]; then
  remote_vsc_scratch="${remote_vsc_data}"
fi

RUN_ROOT="${RUN_ROOT:-${remote_vsc_scratch}/CDC2026-runs}"
PERSIST_ROOT="${PERSIST_ROOT:-${remote_vsc_data}/CDC2026-runs}"
remote_repo_dir="${remote_vsc_data}/${REMOTE_SUBDIR}"

echo "[submit-spce-opt] Local repo:  ${LOCAL_REPO_DIR}"
echo "[submit-spce-opt] Remote repo: ${remote_repo_dir}"
echo "[submit-spce-opt] Run root:    ${RUN_ROOT}"
echo "[submit-spce-opt] Persist root:${PERSIST_ROOT}"
echo "[submit-spce-opt] Result tag:  ${RESULTS_BASENAME}"
echo "[submit-spce-opt] Source:      ${source_branch}@${source_commit}"
echo "[submit-spce-opt] Config:      n_iters=${N_ITERS} n_restarts=${N_RESTARTS} B=${B} L=${L} M=${M}"

ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'mkdir -p "${remote_repo_dir}"'"

rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.ruff_cache/' \
  --exclude='results/' \
  --exclude='artifacts/' \
  --exclude='.remote_logs/' \
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
export JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026"
"$VSC_DATA/software/julia/${julia_version}/bin/julia" --project=. -e 'using Pkg; Pkg.instantiate()'
EOF

echo "[submit-spce-opt] Estimated credits for this sbatch request:"
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" <<'EOF'
set -euo pipefail
remote_repo_dir="$1"
partition="$2"
walltime="$3"
account="$4"

cd "${remote_repo_dir}"
sam-quote sbatch --partition="${partition}" --time="${walltime}" --account="${account}" vsc_scripts/optimize_spce.slurm || true
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[submit-spce-opt] DRY_RUN=1 set, stopping before sbatch submission"
  exit 0
fi

job_submit_output="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" "${JULIA_VERSION}" "${RUN_ROOT}" "${PERSIST_ROOT}" "${RESULTS_BASENAME}" "${N_ITERS}" "${N_RESTARTS}" "${B}" "${L}" "${M}" "${N_SUBSTEPS}" "${LR_MAX}" "${LR_MIN}" "${WARMUP}" "${GRAD_ACCUM}" "${SEED}" "${source_branch}" "${source_commit}" <<'EOF'
set -euo pipefail
remote_repo_dir="$1"
partition="$2"
walltime="$3"
account="$4"
julia_version="$5"
run_root="$6"
persist_root="$7"
results_basename="$8"
n_iters="$9"
n_restarts="${10}"
B="${11}"
L="${12}"
M="${13}"
n_substeps="${14}"
lr_max="${15}"
lr_min="${16}"
warmup="${17}"
grad_accum="${18}"
seed="${19}"
source_branch="${20}"
source_commit="${21}"

cd "${remote_repo_dir}"
if [[ -z "${VSC_DATA:-}" ]]; then
  export VSC_DATA="$(dirname "${remote_repo_dir}")"
fi
mkdir -p logs

sbatch --parsable \
  --partition="${partition}" \
  --time="${walltime}" \
  --account="${account}" \
  --export=ALL,VSC_DATA="$VSC_DATA",REPO_DIR="${remote_repo_dir}",JULIA_VERSION="${julia_version}",JULIA_DEPOT_PATH="$VSC_DATA/julia-depot-cdc2026",RUN_ROOT="${run_root}",PERSIST_ROOT="${persist_root}",RESULTS_BASENAME="${results_basename}",N_ITERS="${n_iters}",N_RESTARTS="${n_restarts}",B="${B}",L="${L}",M="${M}",N_SUBSTEPS="${n_substeps}",LR_MAX="${lr_max}",LR_MIN="${lr_min}",WARMUP="${warmup}",GRAD_ACCUM="${grad_accum}",SEED="${seed}",SOURCE_BRANCH="${source_branch}",SOURCE_COMMIT="${source_commit}" \
  vsc_scripts/optimize_spce.slurm
EOF
)"

job_id="${job_submit_output%%;*}"
if [[ -z "${job_id}" ]]; then
  echo "[submit-spce-opt] ERROR: empty sbatch output: ${job_submit_output}" >&2
  exit 1
fi

echo "[submit-spce-opt] Submitted job ${job_id} on ${PARTITION}"
echo "[submit-spce-opt] Queue:      ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"squeue -M wice -j ${job_id}\""
echo "[submit-spce-opt] Accounting: ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"sacct -M wice -j ${job_id} --format=JobID,State,Elapsed,ReqGRES,AllocTRES%80\""
echo "[submit-spce-opt] Logs:       ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"bash -lc 'tail -f ${remote_repo_dir}/logs/cdc2026-spce-opt-${job_id}.out'\""
echo "[submit-spce-opt] Results:    ${PERSIST_ROOT}/${RESULTS_BASENAME}"
