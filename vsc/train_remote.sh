#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Idempotent remote GPU wrapper for Snakemake.
#
# Called by Snakefile GPU rules. Handles the full lifecycle:
#   1st call: syncs code to VSC cluster, submits SLURM job, exits 1
#   2nd call: checks job status, syncs results back on completion
#
# Usage:  TASK=train ./vsc/train_remote.sh
#         TASK=optimize_static ./vsc/train_remote.sh
#
# State tracked in .pipeline-state/${TASK}.jobid (survives SSH disconnects).
# ============================================================================

EXAMPLE="${EXAMPLE:-monod}"
TASK="${TASK:-train}"
REMOTE_HOST="${REMOTE_HOST:-tier2}"
SSH_SOCKET="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"
SSH_OPTS=(-S "${SSH_SOCKET}" -o BatchMode=yes)
LOCAL_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-CDC2026}"
PARTITION="${PARTITION:-gpu_a100}"
ACCOUNT="${ACCOUNT:-intro_vsc32553}"
JULIA_VERSION="${JULIA_VERSION:-1.12.5}"
WALLTIME="${WALLTIME:-24:00:00}"
SCRIPT_ARGS="${SCRIPT_ARGS:-}"

STATE_DIR="${LOCAL_REPO_DIR}/.pipeline-state"
STATE_FILE="${STATE_DIR}/${TASK}.jobid"

# --- Resolve remote VSC_DATA and repo path ---
resolve_remote() {
    remote_vsc_data="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" \
        "source /etc/profile 2>/dev/null; printf %s \"\$VSC_DATA\"")"
    if [[ -z "${remote_vsc_data}" ]]; then
        echo "[train_remote] ERROR: remote VSC_DATA is empty" >&2
        exit 1
    fi
    remote_vsc_scratch="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" \
        "source /etc/profile 2>/dev/null; printf %s \"\$VSC_SCRATCH\"")"
    if [[ -z "${remote_vsc_scratch}" ]]; then
        remote_vsc_scratch="${remote_vsc_data}"
    fi
    remote_repo_dir="${remote_vsc_data}/${REMOTE_SUBDIR}"
}

# ============================================================================
#  Check existing job OR submit new one
# ============================================================================

if [[ -f "$STATE_FILE" ]]; then
    JOB_ID=$(cat "$STATE_FILE")
    echo "[train_remote] Found saved job $JOB_ID, checking status..."

    STATE=$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" \
        "sacct -M wice -j ${JOB_ID} --format=State -n -P | head -1" || echo "UNKNOWN")
    STATE=$(echo "$STATE" | tr -d '[:space:]')

    case "$STATE" in
        COMPLETED)
            echo "[train_remote] Job $JOB_ID completed successfully."
            resolve_remote
            echo "[train_remote] Syncing results from cluster..."
            rsync -avz \
                -e "ssh -S ${SSH_SOCKET}" \
                "${REMOTE_HOST}:${remote_repo_dir}/examples/${EXAMPLE}/results/" \
                "${LOCAL_REPO_DIR}/examples/${EXAMPLE}/results/"
            rm "$STATE_FILE"
            echo "[train_remote] Done."
            exit 0
            ;;
        FAILED|CANCELLED|CANCELLED+|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            echo "[train_remote] Job $JOB_ID failed: $STATE" >&2
            resolve_remote
            echo "[train_remote] Last 20 lines of job log:" >&2
            ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" \
                "tail -20 ${remote_repo_dir}/logs/cdc2026-${JOB_ID}.out 2>/dev/null" >&2 || true
            rm "$STATE_FILE"
            exit 1
            ;;
        *)
            echo "[train_remote] Job $JOB_ID still running ($STATE)."
            echo "[train_remote] Run 'snakemake' again later."
            exit 1
            ;;
    esac
else
    # --- First call: sync code and submit ---
    if ! ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" 2>/dev/null; then
        echo "[train_remote] ERROR: No SSH control socket. Start one with:" >&2
        echo "  ssh -MNf -o ControlMaster=yes -o ControlPersist=24h -o ControlPath=${SSH_SOCKET} ${REMOTE_HOST}" >&2
        exit 1
    fi

    resolve_remote

    source_branch="$(git -C "${LOCAL_REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    source_commit="$(git -C "${LOCAL_REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    RESULTS_BASENAME="${EXAMPLE}-${TASK}-$(date +%Y%m%d-%H%M%S)"
    RUN_ROOT="${remote_vsc_scratch}/CDC2026-runs"
    PERSIST_ROOT="${remote_vsc_data}/CDC2026-runs"

    # Sync code to cluster
    echo "[train_remote] Syncing code to ${REMOTE_HOST}:${remote_repo_dir}/ ..."
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'mkdir -p \"${remote_repo_dir}\"'"
    rsync -az --delete \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='.ruff_cache/' \
        --exclude='examples/*/results/' \
        --exclude='.pipeline-state/' \
        -e "ssh -S ${SSH_SOCKET}" \
        "${LOCAL_REPO_DIR}/" "${REMOTE_HOST}:${remote_repo_dir}/"

    # Bootstrap Julia + instantiate
    echo "[train_remote] Bootstrapping Julia on cluster..."
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- \
        "${remote_repo_dir}" "${JULIA_VERSION}" <<'BOOTSTRAP'
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
BOOTSTRAP

    # Submit SLURM job
    echo "[train_remote] Submitting job..."
    sbatch_output="$(ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" bash -s -- \
        "${remote_repo_dir}" "${PARTITION}" "${WALLTIME}" "${ACCOUNT}" "${JULIA_VERSION}" \
        "${RUN_ROOT}" "${PERSIST_ROOT}" "${RESULTS_BASENAME}" \
        "${EXAMPLE}" "${TASK}" "${SCRIPT_ARGS}" \
        "${source_branch}" "${source_commit}" <<'SUBMIT' 2>&1
remote_repo_dir="$1"; partition="$2"; walltime="$3"; account="$4"
julia_version="$5"; run_root="$6"; persist_root="$7"; results_basename="$8"
example="$9"; task="${10}"; script_args="${11}"
source_branch="${12}"; source_commit="${13}"

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
SUBMIT
    )" || true

    # Extract job ID from last line (sbatch may emit warnings before the ID)
    JOB_ID="$(echo "$sbatch_output" | grep -oP '^\d+' | tail -1)"
    JOB_ID="${JOB_ID%%;*}"
    if [[ -z "$JOB_ID" ]]; then
        echo "[train_remote] ERROR: empty sbatch output" >&2
        exit 1
    fi

    mkdir -p "$STATE_DIR"
    echo "$JOB_ID" > "$STATE_FILE"

    echo "[train_remote] Submitted training job $JOB_ID"
    echo "[train_remote] Monitor: ssh -S ${SSH_SOCKET} ${REMOTE_HOST} \"sacct -M wice -j ${JOB_ID} --format=JobID,State,Elapsed\""
    echo "[train_remote] Run 'snakemake' again after job completes."
    exit 1  # Fail intentionally — checkpoint not ready yet
fi
