#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-tier2}"
SSH_SOCKET="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"
SSH_OPTS=(-S "${SSH_SOCKET}" -o BatchMode=yes)

ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" >/dev/null

echo "== Tier-2 identity =="
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'hostname; whoami; date'"

echo
echo "== Credit balance =="
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'sam-balance'"

echo
echo "== GPU partitions (wICE) =="
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'sinfo -M wice -p gpu_a100_debug,gpu_a100,gpu_h100 -o \"%P|%a|%l|%D|%C|%G\"'"

echo
echo "== Billing weights =="
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'scontrol show partition gpu_a100 -M wice | tr "\n" " " | sed "s/ TRESBillingWeights=/\nTRESBillingWeights=/"; echo; scontrol show partition gpu_h100 -M wice | tr "\n" " " | sed "s/ TRESBillingWeights=/\nTRESBillingWeights=/"'"

echo
echo "== Julia modules =="
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'module spider Julia 2>&1'"

echo
echo "== CUDA modules =="
ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "bash -lc 'module spider CUDA 2>&1'"
