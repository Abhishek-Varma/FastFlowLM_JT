#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Provision the HRX dev package (public headers + CMake package config +
# libhrx.so) by BUILDING it from the public FastFlow SDK (ROCm/FastFlow-SDK)
# via `dev.py build hrx`. This is the sole HRX source for FLM -- HRX is supplied
# via the public FastFlow SDK, with no dependency on any private HRX fork.
#
# The SDK builds HRX from its projects/hrx submodule (github.com/ROCm/hrx-system)
# at the commit pinned in the SDK's manifests/versions.toml [hrx].commit. This
# script pins the SDK snapshot + the expected HRX commit in ./fastflow-sdk.env,
# builds HRX, and stages the resulting build/hrx-install prefix into
# ./.hrx-release/<prefix> so the find_package(hrx CONFIG REQUIRED) flow
# consumes it: it prints HRX_CMAKE_PREFIX and the -DCMAKE_PREFIX_PATH hint, and
# writes `prefix=<prefix>` to $GITHUB_OUTPUT when set.
#
# Env knobs:
#   FASTFLOW_SDK_DIR        reuse an existing FastFlow-SDK checkout (skip clone)
#   FASTFLOW_SDK_SKIP_SETUP =1 to skip `dev.py setup` (deps already installed)
#   FASTFLOW_SDK_SKIP_BUILD =1 to skip `dev.py build hrx` (reuse build/hrx-install)
#   HRX_ALLOW_COMMIT_DRIFT  =1 to downgrade the HRX commit mismatch to a warning
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: provision-hrx-from-fastflow-sdk.sh [out-dir]

Builds HRX from the pinned FastFlow SDK snapshot (./fastflow-sdk.env) and stages
the HRX dev package into out-dir (default ./.hrx-release). Prints the CMake
package prefix for find_package(hrx) / -DCMAKE_PREFIX_PATH.
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[provision-hrx] $*" >&2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$here/fastflow-sdk.env"
[[ -f "$env_file" ]] || die "missing $env_file"
# shellcheck disable=SC1090
source "$env_file"

: "${FASTFLOW_SDK_REPO:?missing FASTFLOW_SDK_REPO in $env_file}"
: "${FASTFLOW_SDK_REF:?missing FASTFLOW_SDK_REF in $env_file}"
: "${HRX_EXPECTED_COMMIT:?missing HRX_EXPECTED_COMMIT in $env_file}"

out_dir="${1:-$here/.hrx-release}"
mkdir -p "$out_dir"

py="${PYTHON:-python3}"
command -v "$py" >/dev/null 2>&1 || die "python3 not found on PATH (set PYTHON=...)"

# 1. Obtain a FastFlow SDK checkout at the pinned ref.
if [[ -n "${FASTFLOW_SDK_DIR:-}" ]]; then
  sdk_dir="$FASTFLOW_SDK_DIR"
  [[ -f "$sdk_dir/dev.py" ]] || die "FASTFLOW_SDK_DIR=$sdk_dir has no dev.py"
  log "reusing existing FastFlow SDK checkout: $sdk_dir"
else
  sdk_dir="$out_dir/.fastflow-sdk-src"
  if [[ ! -d "$sdk_dir/.git" ]]; then
    log "cloning $FASTFLOW_SDK_REPO -> $sdk_dir"
    git clone --filter=blob:none "$FASTFLOW_SDK_REPO" "$sdk_dir"
  fi
  log "checking out FastFlow SDK ref $FASTFLOW_SDK_REF"
  git -C "$sdk_dir" fetch --tags --filter=blob:none origin "$FASTFLOW_SDK_REF" 2>/dev/null || git -C "$sdk_dir" fetch --tags origin
  git -C "$sdk_dir" checkout --quiet "$FASTFLOW_SDK_REF"
fi

# 2. Check out the HRX submodule and verify the pinned commit.
log "initializing projects/hrx submodule"
git -C "$sdk_dir" submodule sync projects/hrx >/dev/null 2>&1 || true
git -C "$sdk_dir" submodule update --init projects/hrx
hrx_head="$(git -C "$sdk_dir/projects/hrx" rev-parse HEAD)"
# Cross-check what the SDK's own manifest claims to pin, for a clear message.
sdk_pin="$(sed -n '/^\[hrx\]/,/^\[/p' "$sdk_dir/manifests/versions.toml" \
             | sed -n 's/^commit *= *"\([0-9a-f]\{40\}\)".*/\1/p' | head -n1 || true)"
log "HRX submodule HEAD : $hrx_head"
log "SDK versions.toml  : ${sdk_pin:-<none>}"
log "expected (our pin) : $HRX_EXPECTED_COMMIT"
if [[ "$hrx_head" != "$HRX_EXPECTED_COMMIT" ]]; then
  msg="HRX commit drift: FastFlow SDK $FASTFLOW_SDK_REF checked out $hrx_head, expected $HRX_EXPECTED_COMMIT"
  if [[ "${HRX_ALLOW_COMMIT_DRIFT:-0}" == "1" ]]; then
    log "WARNING: $msg (continuing: HRX_ALLOW_COMMIT_DRIFT=1)"
  else
    die "$msg (set HRX_ALLOW_COMMIT_DRIFT=1 to override, or update fastflow-sdk.env)"
  fi
fi

# 3. Build HRX via the SDK developer CLI.
if [[ "${FASTFLOW_SDK_SKIP_SETUP:-0}" != "1" ]]; then
  log "dev.py setup --profile fastflowlm (installs pinned build deps into SDK caches)"
  ( cd "$sdk_dir" && "$py" dev.py setup --profile fastflowlm )
fi
if [[ "${FASTFLOW_SDK_SKIP_BUILD:-0}" != "1" ]]; then
  log "dev.py build hrx --fresh"
  ( cd "$sdk_dir" && "$py" dev.py build hrx --fresh )
fi

hrx_install="$sdk_dir/build/hrx-install"
[[ -d "$hrx_install" ]] || die "HRX install prefix not found: $hrx_install (did 'dev.py build hrx' run?)"

# 4. Validate the dev package is complete.
cfg="$hrx_install/lib/cmake/hrx/hrx-config.cmake"
[[ -f "$cfg" ]] || die "HRX CMake package missing: $cfg"
for h in hrx_runtime.h hrx_amdxdna.h; do
  [[ -f "$hrx_install/include/hrx/$h" ]] || die "HRX public header missing: include/hrx/$h"
done
compgen -G "$hrx_install/lib/libhrx.so*" >/dev/null || die "libhrx.so missing under $hrx_install/lib"

# 5. Stage into .hrx-release/<prefix> (one dir deep so the Makefile/CMake
#    auto-locate finds it) with a fresh copy. Normalize mtimes to "now": HRX
#    build artifacts can
#    carry future-dated entries, and CMake records imported package configs as
#    regeneration inputs, which otherwise makes Ninja re-run CMake until it
#    aborts with "build.ninja still dirty after 100 tries".
short="${hrx_head:0:12}"
prefix="$out_dir/hrx-fastflow-sdk-$short"
log "staging HRX dev package -> $prefix"
rm -rf "$prefix"
mkdir -p "$prefix"
cp -a "$hrx_install/." "$prefix/"
find "$prefix" -exec touch -h -c {} + 2>/dev/null || find "$prefix" -exec touch {} +

# Warn on collision: the Makefile/CMake auto-locate globs .hrx-release/*/ and
# takes the first match, so a leftover HRX package dir on disk (e.g. an old
# hrx-amdxdna-* extraction) can shadow this one. Feed the HRX_HOME printed below
# explicitly to be unambiguous.
siblings="$(find "$out_dir" -maxdepth 2 -path '*/lib/libhrx.so' 2>/dev/null \
              | grep -v "/$(basename "$prefix")/lib/libhrx.so$" || true)"
if [[ -n "$siblings" ]]; then
  log "NOTE: other HRX package(s) exist under $out_dir:"
  while IFS= read -r s; do log "        ${s%/lib/libhrx.so}"; done <<< "$siblings"
  log "      Makefile auto-locate is ambiguous; pass the HRX_HOME printed below."
fi

echo "HRX_CMAKE_CONFIG=$prefix/lib/cmake/hrx/hrx-config.cmake"
echo "HRX_CMAKE_PREFIX=$prefix"
echo ""
echo "Configure FLM with: -DCMAKE_PREFIX_PATH=\"$prefix\""
echo "Or for the Makefile engines: HRX_HOME=\"$prefix\""
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "prefix=$prefix" >> "$GITHUB_OUTPUT"
fi
