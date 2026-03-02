#!/usr/bin/env bash
set -euo pipefail

JULIA_VERSION="${JULIA_VERSION:-1.12.5}"
INSTALL_ROOT="${VSC_DATA:-$HOME}/software/julia"
JULIA_BIN="${INSTALL_ROOT}/${JULIA_VERSION}/bin/julia"
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${VSC_DATA:-$HOME}/julia-depot-cdc2026}"

mkdir -p "${JULIA_DEPOT_PATH}"

echo "[bootstrap] Julia binary: ${JULIA_BIN}"
"${JULIA_BIN}" --version
echo "[bootstrap] Depot path: ${JULIA_DEPOT_PATH}"
