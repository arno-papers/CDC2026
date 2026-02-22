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

echo "[smoke] Running tiny end-to-end training (5 iters, grad_accum=10)"

# Background GPU monitor: sample utilization + memory every 5s
GPU_LOG="${REPO_DIR}/logs/gpu_util_${SLURM_JOB_ID}.csv"
echo "timestamp,gpu_util_pct,mem_used_MiB,mem_total_MiB,power_W,temp_C" > "${GPU_LOG}"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "${GPU_LOG}" 2>/dev/null
    sleep 5
  done
) &
GPU_MONITOR_PID=$!

"${JULIA_BIN}" --project=. training.jl n_iters=5 grad_accum=10 plotting=false loss_png_every=1 seed=0

kill "${GPU_MONITOR_PID}" 2>/dev/null || true
wait "${GPU_MONITOR_PID}" 2>/dev/null || true

echo ""
echo "[smoke] === GPU Utilization Summary ==="
if [[ -f "${GPU_LOG}" ]]; then
  # Skip header, extract gpu_util and mem_used columns
  awk -F', *' 'NR>1 && $2+0 > 0 {
    n++; sum_util+=$2; sum_mem+=$3;
    if($2>max_util) max_util=$2;
    if($3>max_mem) max_mem=$3;
    mem_total=$4
  } END {
    if(n>0) {
      printf "[smoke] Samples with GPU active: %d\n", n
      printf "[smoke] GPU Util: mean=%.0f%%, max=%.0f%%\n", sum_util/n, max_util
      printf "[smoke] Memory:   mean=%.0f MiB, max=%.0f MiB / %s MiB\n", sum_mem/n, max_mem, mem_total
    } else {
      print "[smoke] No GPU-active samples found (compilation only?)"
    }
  }' "${GPU_LOG}"
  echo "[smoke] Raw log: ${GPU_LOG}"
fi

echo "[smoke] Completed successfully"
