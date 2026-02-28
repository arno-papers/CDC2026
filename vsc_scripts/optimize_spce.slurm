#!/bin/bash -l
#SBATCH --job-name=cdc2026-spce-opt
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=intro_vsc32553
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
JULIA_VERSION="${JULIA_VERSION:-1.12.4}"
JULIA_BIN="${JULIA_BIN:-${VSC_DATA:-$HOME}/software/julia/${JULIA_VERSION}/bin/julia}"
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${VSC_DATA:-$HOME}/julia-depot-cdc2026}"

N_ITERS="${N_ITERS:-300}"
N_RESTARTS="${N_RESTARTS:-1}"
B="${B:-64}"
L="${L:-124}"
M="${M:-128}"
N_SUBSTEPS="${N_SUBSTEPS:-500}"
LR_MAX="${LR_MAX:-0.01}"
LR_MIN="${LR_MIN:-1e-5}"
WARMUP="${WARMUP:-20}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SEED="${SEED:-0}"

RUN_ROOT="${RUN_ROOT:-${VSC_SCRATCH:-${VSC_DATA:-$HOME}}/CDC2026-runs}"
PERSIST_ROOT="${PERSIST_ROOT:-${VSC_DATA:-$HOME}/CDC2026-runs}"
RESULTS_BASENAME="${RESULTS_BASENAME:-spce-opt-${SLURM_JOB_ID}}"
RESULTS_DIR="${RUN_ROOT}/${RESULTS_BASENAME}"
ARCHIVE_DIR="${PERSIST_ROOT}/${RESULTS_BASENAME}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
SOURCE_BRANCH="${SOURCE_BRANCH:-unknown}"

export JULIA_DEPOT_PATH
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1

mkdir -p "${REPO_DIR}/logs" "${RUN_ROOT}" "${PERSIST_ROOT}"
mkdir -p "${RESULTS_DIR}"
cd "${REPO_DIR}"

echo "[spce-opt] Host: $(hostname)"
echo "[spce-opt] Job ID: ${SLURM_JOB_ID}"
echo "[spce-opt] Partition: ${SLURM_JOB_PARTITION}"
echo "[spce-opt] Repo: ${REPO_DIR}"
echo "[spce-opt] Julia: ${JULIA_BIN}"
echo "[spce-opt] Depot: ${JULIA_DEPOT_PATH}"
echo "[spce-opt] Results (scratch): ${RESULTS_DIR}"
echo "[spce-opt] Results (persistent): ${ARCHIVE_DIR}"
echo "[spce-opt] n_iters=${N_ITERS} n_restarts=${N_RESTARTS} B=${B} L=${L} M=${M}"
echo "[spce-opt] source_branch=${SOURCE_BRANCH} source_commit=${SOURCE_COMMIT}"

cat > "${RESULTS_DIR}/run_metadata.txt" <<EOF
job_id=${SLURM_JOB_ID}
job_name=${SLURM_JOB_NAME}
partition=${SLURM_JOB_PARTITION}
host=$(hostname)
start_time=$(date -Is)
repo_dir=${REPO_DIR}
source_branch=${SOURCE_BRANCH}
source_commit=${SOURCE_COMMIT}
julia_version=${JULIA_VERSION}
julia_bin=${JULIA_BIN}
julia_depot_path=${JULIA_DEPOT_PATH}
n_iters=${N_ITERS}
n_restarts=${N_RESTARTS}
B=${B}
L=${L}
M=${M}
n_substeps=${N_SUBSTEPS}
lr_max=${LR_MAX}
lr_min=${LR_MIN}
warmup=${WARMUP}
grad_accum=${GRAD_ACCUM}
seed=${SEED}
results_dir=${RESULTS_DIR}
archive_dir=${ARCHIVE_DIR}
EOF

if [[ ! -x "${JULIA_BIN}" ]]; then
  echo "[spce-opt] Julia binary not found: ${JULIA_BIN}" >&2
  exit 1
fi

nvidia-smi
"${JULIA_BIN}" --version

"${JULIA_BIN}" --project=. -e 'using Pkg; Pkg.instantiate()'

set +e
"${JULIA_BIN}" --project=. optimize_static_spce.jl \
  n_iters="${N_ITERS}" \
  n_restarts="${N_RESTARTS}" \
  B="${B}" \
  L="${L}" \
  M="${M}" \
  n_substeps="${N_SUBSTEPS}" \
  lr_max="${LR_MAX}" \
  lr_min="${LR_MIN}" \
  warmup="${WARMUP}" \
  grad_accum="${GRAD_ACCUM}" \
  seed="${SEED}" \
  results_dir="${RESULTS_DIR}"
run_rc=$?
set -e

if [[ -d "${RESULTS_DIR}" ]]; then
  mkdir -p "${ARCHIVE_DIR}"
  rsync -a "${RESULTS_DIR}/" "${ARCHIVE_DIR}/"
fi

if [[ ${run_rc} -ne 0 ]]; then
  echo "[spce-opt] Optimization failed with exit code ${run_rc}" >&2
  exit ${run_rc}
fi

echo "[spce-opt] Optimization completed successfully"
echo "[spce-opt] Persisted results: ${ARCHIVE_DIR}"
