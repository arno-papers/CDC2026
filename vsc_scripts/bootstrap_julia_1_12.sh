#!/usr/bin/env bash
set -euo pipefail

JULIA_VERSION="${JULIA_VERSION:-1.12.4}"
JULIA_SERIES="${JULIA_VERSION%.*}"
INSTALL_ROOT="${JULIA_INSTALL_ROOT:-${VSC_DATA:-$HOME}/software/julia}"
JULIA_PREFIX="${JULIA_PREFIX:-${INSTALL_ROOT}/${JULIA_VERSION}}"
JULIA_BIN="${JULIA_PREFIX}/bin/julia"
JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${VSC_DATA:-$HOME}/julia-depot-cdc2026}"

archive="julia-${JULIA_VERSION}-linux-x86_64.tar.gz"
url="https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_SERIES}/${archive}"

mkdir -p "${INSTALL_ROOT}" "${JULIA_DEPOT_PATH}"

if [[ ! -x "${JULIA_BIN}" ]]; then
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' EXIT

  echo "[bootstrap] Downloading ${url}"
  curl -fL "${url}" -o "${tmpdir}/${archive}"

  echo "[bootstrap] Extracting Julia ${JULIA_VERSION}"
  tar -xzf "${tmpdir}/${archive}" -C "${tmpdir}"
  mv "${tmpdir}/julia-${JULIA_VERSION}" "${JULIA_PREFIX}"
fi

echo "[bootstrap] Julia binary: ${JULIA_BIN}"
"${JULIA_BIN}" --version

echo "[bootstrap] Depot path: ${JULIA_DEPOT_PATH}"
echo "[bootstrap] Export with: export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH}"
