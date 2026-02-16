#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job_id> [remote_host]" >&2
  exit 1
fi

job_id="$1"
remote_host="${2:-tier2}"
socket="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"

ssh -S "${socket}" -o BatchMode=yes -O check "${remote_host}" >/dev/null

echo "[watch] Watching ${job_id} on ${remote_host}"
while ssh -S "${socket}" -o BatchMode=yes "${remote_host}" "bash -lc 'squeue -M wice -h -j ${job_id}'" | grep -q .; do
  ssh -S "${socket}" -o BatchMode=yes "${remote_host}" "bash -lc 'squeue -M wice -j ${job_id} -o \"%i|%j|%T|%M|%L|%P|%R\"'"
  sleep 30
done

echo "[watch] Final accounting"
ssh -S "${socket}" -o BatchMode=yes "${remote_host}" \
  "bash -lc 'sacct -M wice -j ${job_id} --format=JobID,JobName%24,Partition,State,Elapsed,ExitCode,ReqGRES,AllocTRES%100'"
