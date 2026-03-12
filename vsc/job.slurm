#!/bin/bash -l
#SBATCH --job-name=cdc2026
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=lp_dad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ============================================================================
# Unified SLURM job template.
#
# Required env vars:
#   EXAMPLE     — example name (e.g. "monod")
#   TASK        — task name (e.g. "train", "eval_bim", "optimize_static", etc.)
#
# Optional env vars (with defaults):
#   SCRIPT_ARGS — extra arguments passed to the Julia script
#   RESULTS_BASENAME, PERSIST_ROOT — result archiving
# ============================================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
JULIA_VERSION="${JULIA_VERSION:-1.12.5}"
JULIA_BIN="${JULIA_BIN:-${VSC_DATA:-$HOME}/software/julia/${JULIA_VERSION}/bin/julia}"
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${VSC_DATA:-$HOME}/julia-depot-cdc2026}"

EXAMPLE="${EXAMPLE:?EXAMPLE must be set (e.g. monod)}"
TASK="${TASK:?TASK must be set (e.g. train, eval_bim, optimize_static)}"
SCRIPT_ARGS="${SCRIPT_ARGS:-}"

PERSIST_ROOT="${PERSIST_ROOT:-${VSC_DATA:-$HOME}/CDC2026-runs}"
RESULTS_BASENAME="${RESULTS_BASENAME:-${EXAMPLE}-${TASK}-${SLURM_JOB_ID}}"
ARCHIVE_DIR="${PERSIST_ROOT}/${RESULTS_BASENAME}"
SCRIPT_RESULTS="${REPO_DIR}/examples/${EXAMPLE}/results"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
SOURCE_BRANCH="${SOURCE_BRANCH:-unknown}"

export JULIA_BIN
export JULIA_DEPOT_PATH
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1

# Resolve script path
SCRIPT_PATH="examples/${EXAMPLE}/${TASK}.jl"

mkdir -p "${REPO_DIR}/logs" "${PERSIST_ROOT}" "${SCRIPT_RESULTS}"
cd "${REPO_DIR}"

echo "[job] Host: $(hostname)"
echo "[job] Job ID: ${SLURM_JOB_ID}"
echo "[job] Partition: ${SLURM_JOB_PARTITION}"
echo "[job] Repo: ${REPO_DIR}"
echo "[job] Julia: ${JULIA_BIN}"
echo "[job] Depot: ${JULIA_DEPOT_PATH}"
echo "[job] Script: ${SCRIPT_PATH}"
echo "[job] Example: ${EXAMPLE}, Task: ${TASK}"
echo "[job] Results (script): ${SCRIPT_RESULTS}"
echo "[job] Results (archive): ${ARCHIVE_DIR}"
echo "[job] source_branch=${SOURCE_BRANCH} source_commit=${SOURCE_COMMIT}"

cat > "${SCRIPT_RESULTS}/run_metadata.txt" <<EOF
job_id=${SLURM_JOB_ID}
job_name=${SLURM_JOB_NAME}
partition=${SLURM_JOB_PARTITION}
host=$(hostname)
start_time=$(date -Is)
repo_dir=${REPO_DIR}
source_branch=${SOURCE_BRANCH}
source_commit=${SOURCE_COMMIT}
julia_version=${JULIA_VERSION}
example=${EXAMPLE}
task=${TASK}
script_path=${SCRIPT_PATH}
script_args=${SCRIPT_ARGS}
results_dir=${SCRIPT_RESULTS}
archive_dir=${ARCHIVE_DIR}
EOF

nvidia-smi
"${JULIA_BIN}" --version

"${JULIA_BIN}" --project=. -e 'using Pkg; Pkg.instantiate()'

set +e
# shellcheck disable=SC2086
"${JULIA_BIN}" --project=. "${SCRIPT_PATH}" \
  ${SCRIPT_ARGS}
run_rc=$?
set -e

if [[ -d "${SCRIPT_RESULTS}" ]]; then
  mkdir -p "${ARCHIVE_DIR}"
  rsync -a "${SCRIPT_RESULTS}/" "${ARCHIVE_DIR}/"
fi

if [[ ${run_rc} -ne 0 ]]; then
  echo "[job] Script failed with exit code ${run_rc}" >&2
  exit ${run_rc}
fi

echo "[job] Completed successfully"
echo "[job] Persisted results: ${ARCHIVE_DIR}"
