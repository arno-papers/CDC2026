#!/usr/bin/env bash
set -euo pipefail

remote_host="${REMOTE_HOST:-tier2}"
socket="${SSH_SOCKET:-$HOME/.ssh/cm-tier2-%r@%h:%p}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rm -rf "${repo_root}/.remote_logs"

ssh -S "${socket}" -o BatchMode=yes -O check "${remote_host}" >/dev/null
ssh -S "${socket}" -o BatchMode=yes "${remote_host}" bash -s <<'EOF'
set -euo pipefail
if [[ -z "${VSC_DATA:-}" ]]; then
  exit 0
fi
repo="$VSC_DATA/CDC2026"
mkdir -p "$repo/logs"
rm -f "$repo"/logs/cdc2026-smoke-*.out "$repo"/logs/cdc2026-smoke-*.err
rm -f "$repo/checkpoint.jls" "$repo/diagnostics.jls" "$repo/plot_loss_live.png"
ls -lah "$repo/logs"
EOF

echo "[cleanup] Local and remote smoke artifacts removed"
