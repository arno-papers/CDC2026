#!/bin/bash -l
#SBATCH --job-name=cdc2026-smoke
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --account=intro_vsc32553
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
JULIA_VERSION="${JULIA_VERSION:-1.12.4}"
JULIA_BIN="${JULIA_BIN:-${VSC_DATA:-$HOME}/software/julia/${JULIA_VERSION}/bin/julia}"
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${VSC_DATA:-$HOME}/julia-depot-cdc2026}"

export JULIA_DEPOT_PATH
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

echo "[smoke] Host: $(hostname)"
echo "[smoke] Repo: ${REPO_DIR}"
echo "[smoke] Julia: ${JULIA_BIN}"
echo "[smoke] Depot: ${JULIA_DEPOT_PATH}"
echo "[smoke] Job ID: ${SLURM_JOB_ID}"

if [[ ! -x "${JULIA_BIN}" ]]; then
  echo "[smoke] Julia binary not found: ${JULIA_BIN}" >&2
  exit 1
fi

nvidia-smi
"${JULIA_BIN}" --version

"${JULIA_BIN}" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(strict=false)'

echo "[smoke] Running eager CUDA first-iter check"
"${JULIA_BIN}" --project=. training_eager_cuda_first_iter.jl seed=0

echo "[smoke] Running tiny end-to-end training"
"${JULIA_BIN}" --project=. training.jl n_iters=2 grad_accum=1 plotting=false loss_png_every=1 seed=0

echo "[smoke] Completed successfully"
