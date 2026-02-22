# CDC2026 Tier-2 (KU Leuven wICE) Workflow

This project is configured for Julia `1.12.x` and can be run on KU Leuven Tier-2 (`wice`) with Slurm account `intro_vsc32553`.

This `VSC_README.md` is the single source of truth for running on Tier-2.

## Prerequisites

- You have SSH alias `tier2` configured for `vsc32553`.
- You can complete keyboard-interactive verification (MFA) for that account.
- You are in this repository root.

## Command locations (important)

- Local commands in this guide (`./vsc_scripts/...`) should be run from the repo root (the directory where this repository is cloned).
- If needed, start with:

```bash
cd /path/to/CDC2026
```

- Remote commands are executed on Tier-2 login nodes via `ssh ... tier2 "..."` and can be launched from any local directory.
- For manual remote job operations, the synced project lives in `$VSC_DATA/CDC2026` (or `$VSC_DATA/$REMOTE_SUBDIR` if overridden).

## 1) Required first verification step

Run from local repo root (`/path/to/CDC2026`).

Before any automated scripts, do one interactive login first:

```bash
ssh tier2
hostname
whoami
exit
```

Why: without this first verification, non-interactive SSH can fail with:

`Permission denied (publickey,keyboard-interactive)`

## 2) Start an SSH control socket (per work session)

Run from local repo root (`/path/to/CDC2026`).

```bash
ssh -MNf -o ControlMaster=yes -o ControlPersist=24h -o ControlPath=~/.ssh/cm-tier2-%r@%h:%p tier2
ssh -S ~/.ssh/cm-tier2-%r@%h:%p -O check tier2
```

All helper scripts assume this socket exists.

## 3) Preflight check

Run from local repo root (`/path/to/CDC2026`).

```bash
./vsc_scripts/vsc_preflight.sh
```

This prints identity, credits, partition status, billing weights, and software availability.

## 4) Optional smoke test

Run from local repo root (`/path/to/CDC2026`).

```bash
./vsc_scripts/submit_smoke.sh
```

This uses `gpu_a100_debug` (single full A100, short run) and validates Julia/CUDA/training wiring.

## 5) Submit a full training run

Run from local repo root (`/path/to/CDC2026`).

```bash
N_ITERS=1000 \
GRAD_ACCUM=10 \
WALLTIME=24:00:00 \
PARTITION=gpu_a100 \
RESULTS_BASENAME=full-$(date +%Y%m%d-%H%M%S) \
./vsc_scripts/submit_full.sh
```

The submit script does:

- rsync this repo to remote `$VSC_DATA/<REMOTE_SUBDIR>`;
- bootstrap Julia `1.12.4` under `$VSC_DATA/software/julia/`;
- `Pkg.instantiate()`;
- print `sam-quote` cost estimate;
- submit `vsc_scripts/full_wice_a100.slurm`.

## 6) Monitor jobs

Run from local repo root (`/path/to/CDC2026`).

```bash
./vsc_scripts/watch_job.sh <JOBID>
```

or manually:

```bash
ssh -S ~/.ssh/cm-tier2-%r@%h:%p tier2 "squeue -M wice -j <JOBID> -o '%i|%j|%T|%M|%L|%P|%R'"
ssh -S ~/.ssh/cm-tier2-%r@%h:%p tier2 "sacct -M wice -j <JOBID> --format=JobID,JobName%24,Partition,State,Elapsed,ExitCode,ReqGRES,AllocTRES%100"
```

## 7) Changing training parameters / increasing budget

You can change training behavior from `submit_full.sh` environment variables.

Common training knobs:

- `N_ITERS`
- `GRAD_ACCUM`
- `LR_MAX`, `LR_MIN`, `WARMUP`, `CLIP_NORM`
- `SEED`
- `LOSS_PNG_EVERY`
- `PLOTTING`

Budget and runtime knobs:

- `WALLTIME`: Slurm time limit.
- `PARTITION`: hardware/cost profile (`gpu_a100` default).
- `RESULTS_BASENAME`: run naming for reproducibility.

Higher-budget example:

```bash
N_ITERS=5000 \
GRAD_ACCUM=10 \
WALLTIME=72:00:00 \
PARTITION=gpu_a100 \
RESULTS_BASENAME=full-$(date +%Y%m%d-%H%M%S)-budgetx5 \
./vsc_scripts/submit_full.sh
```

Before launching expensive runs, estimate cost only:

```bash
DRY_RUN=1 N_ITERS=5000 WALLTIME=72:00:00 PARTITION=gpu_a100 ./vsc_scripts/submit_full.sh
```

If you need different CPU/memory/GPU resources, edit `vsc_scripts/full_wice_a100.slurm` (`--cpus-per-task`, `--mem`, `--gpus-per-node`) and keep values valid for the selected partition.

## 8) Cleanup

Run from local repo root (`/path/to/CDC2026`).

Smoke-artifact cleanup helper:

```bash
./vsc_scripts/cleanup_smoke_artifacts.sh
```

When done with a session:

```bash
ssh -S ~/.ssh/cm-tier2-%r@%h:%p -O exit tier2
```
