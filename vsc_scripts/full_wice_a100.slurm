#!/bin/bash -l
#SBATCH --job-name=cdc2026-full
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=intro_vsc32553
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
JULIA_VERSION="${JULIA_VERSION:-1.12.4}"
JULIA_BIN="${JULIA_BIN:-${VSC_DATA:-$HOME}/software/julia/${JULIA_VERSION}/bin/julia}"
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${VSC_DATA:-$HOME}/julia-depot-cdc2026}"

N_ITERS="${N_ITERS:-1000}"
GRAD_ACCUM="${GRAD_ACCUM:-10}"
LR_MAX="${LR_MAX:-0.003}"
LR_MIN="${LR_MIN:-1e-5}"
WARMUP="${WARMUP:-50}"
CLIP_NORM="${CLIP_NORM:-1.0}"
LOSS_PNG_EVERY="${LOSS_PNG_EVERY:-10}"
SEED="${SEED:-0}"
PLOTTING="${PLOTTING:-false}"

RUN_ROOT="${RUN_ROOT:-${VSC_SCRATCH:-${VSC_DATA:-$HOME}}/CDC2026-runs}"
PERSIST_ROOT="${PERSIST_ROOT:-${VSC_DATA:-$HOME}/CDC2026-runs}"
RESULTS_BASENAME="${RESULTS_BASENAME:-full-${SLURM_JOB_ID}}"
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

echo "[full] Host: $(hostname)"
echo "[full] Job ID: ${SLURM_JOB_ID}"
echo "[full] Partition: ${SLURM_JOB_PARTITION}"
echo "[full] Repo: ${REPO_DIR}"
echo "[full] Julia: ${JULIA_BIN}"
echo "[full] Depot: ${JULIA_DEPOT_PATH}"
echo "[full] Results (scratch): ${RESULTS_DIR}"
echo "[full] Results (persistent): ${ARCHIVE_DIR}"
echo "[full] n_iters=${N_ITERS} grad_accum=${GRAD_ACCUM} seed=${SEED}"
echo "[full] source_branch=${SOURCE_BRANCH} source_commit=${SOURCE_COMMIT}"

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
grad_accum=${GRAD_ACCUM}
lr_max=${LR_MAX}
lr_min=${LR_MIN}
warmup=${WARMUP}
clip_norm=${CLIP_NORM}
loss_png_every=${LOSS_PNG_EVERY}
seed=${SEED}
plotting=${PLOTTING}
results_dir=${RESULTS_DIR}
archive_dir=${ARCHIVE_DIR}
EOF

if [[ ! -x "${JULIA_BIN}" ]]; then
  echo "[full] Julia binary not found: ${JULIA_BIN}" >&2
  exit 1
fi

nvidia-smi
"${JULIA_BIN}" --version

"${JULIA_BIN}" --project=. -e 'using Pkg; Pkg.instantiate()'

set +e
"${JULIA_BIN}" --project=. training.jl \
  n_iters="${N_ITERS}" \
  grad_accum="${GRAD_ACCUM}" \
  lr_max="${LR_MAX}" \
  lr_min="${LR_MIN}" \
  warmup="${WARMUP}" \
  clip_norm="${CLIP_NORM}" \
  loss_png_every="${LOSS_PNG_EVERY}" \
  seed="${SEED}" \
  plotting="${PLOTTING}" \
  results_dir="${RESULTS_DIR}"
run_rc=$?
set -e

if [[ -d "${RESULTS_DIR}" ]]; then
  mkdir -p "${ARCHIVE_DIR}"
  rsync -a "${RESULTS_DIR}/" "${ARCHIVE_DIR}/"
fi

if [[ ${run_rc} -ne 0 ]]; then
  echo "[full] Training failed with exit code ${run_rc}" >&2
  exit ${run_rc}
fi

echo "[full] Training completed successfully"
echo "[full] Persisted results: ${ARCHIVE_DIR}"
