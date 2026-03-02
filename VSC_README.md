# CDC2026 Tier-2 (KU Leuven wICE) Workflow

This project is configured for Julia `1.12.x` and can be run on KU Leuven Tier-2 (`wice`) with Slurm account `intro_vsc32553`.

This `VSC_README.md` is the single source of truth for running on Tier-2.

## Prerequisites

- You have SSH alias `tier2` configured for `vsc32553`.
- You can complete keyboard-interactive verification (MFA) for that account.
- You are in this repository root.
- Snakemake is installed locally (`pip install snakemake`).

## 1) Required first verification step

Before any automated scripts, do one interactive login first:

```bash
ssh tier2
hostname
whoami
exit
```

## 2) Start an SSH control socket (per work session)

```bash
ssh -MNf -o ControlMaster=yes -o ControlPersist=24h -o ControlPath=~/.ssh/cm-tier2-%r@%h:%p tier2
ssh -S ~/.ssh/cm-tier2-%r@%h:%p -O check tier2
```

All helper scripts assume this socket exists.

## 3) Run the pipeline with Snakemake

Training runs on the VSC cluster; everything else runs locally. The pipeline is idempotent and SSH-disconnect safe.

```bash
# First run: BIM optimization runs locally, training submits to cluster, stops
snakemake --keep-going

# Second run (after training completes): syncs checkpoint, runs eval/plots locally
snakemake

# Compile paper (figures + LaTeX)
snakemake paper

# Dry run (show what would run)
snakemake -n
```

### How it works

The `train` rule calls `vsc/train_remote.sh`, which:
1. **First call:** Syncs code to the cluster, submits a SLURM job, saves the job ID in `.pipeline-state/train.jobid`, then exits with code 1 (so Snakemake knows the checkpoint isn't ready).
2. **Subsequent calls:** Checks the saved job's status via `sacct`. If COMPLETED, rsyncs results back and exits 0. If still running, exits 1. If failed, prints logs and exits 1.

The job ID file survives SSH disconnects — the SLURM job runs independently on the cluster.

### Direct cluster submission (bypassing Snakemake)

```bash
# Training (24h default walltime)
EXAMPLE=monod TASK=train ./vsc/submit.sh

# With custom parameters
SCRIPT_ARGS="n_iters=500 grad_accum=10" EXAMPLE=monod TASK=train ./vsc/submit.sh

# Cost estimate only
DRY_RUN=1 EXAMPLE=monod TASK=train ./vsc/submit.sh
```

After direct submission, sync results manually:
```bash
rsync -avz tier2:$VSC_DATA/CDC2026/examples/monod/results/ examples/monod/results/
```

### Customizable environment variables

| Variable | Default | Description |
|---|---|---|
| `EXAMPLE` | (required) | Example name (e.g. `monod`) |
| `TASK` | (required) | Task name (e.g. `train`, `eval_bim`, `optimize_static`) |
| `SCRIPT_ARGS` | `""` | Extra arguments passed to the Julia script |
| `WALLTIME` | task-dependent | SLURM time limit |
| `PARTITION` | `gpu_a100` | SLURM partition |
| `RESULTS_BASENAME` | auto-generated | Result archive directory name |
| `DRY_RUN` | `0` | Set to `1` for cost estimate only |

## 4) Monitor jobs

```bash
ssh -S ~/.ssh/cm-tier2-%r@%h:%p tier2 "squeue -M wice -j <JOBID> -o '%i|%j|%T|%M|%L|%P|%R'"
ssh -S ~/.ssh/cm-tier2-%r@%h:%p tier2 "sacct -M wice -j <JOBID> --format=JobID,JobName%24,Partition,State,Elapsed,ExitCode,ReqGRES,AllocTRES%100"
```

## 5) Cleanup

When done with a session:

```bash
ssh -S ~/.ssh/cm-tier2-%r@%h:%p -O exit tier2
```
