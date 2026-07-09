#!/usr/bin/env bash
# Fetch and verify the pinned HRX amdxdna release artifact.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: fetch-hrx-release.sh [out-dir]

Downloads the HRX release artifact pinned in ../hrx-release.env, verifies its
checksum, extracts it, and prints the extracted artifact root.

When running in GitHub Actions, the script also writes:
  root=<artifact-root>
to $GITHUB_OUTPUT.
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$here"
release_env="$here/hrx-release.env"
[[ -f "$release_env" ]] || die "missing $release_env"

# shellcheck disable=SC1090
source "$release_env"

: "${HRX_RELEASE_REPO:?missing HRX_RELEASE_REPO in $release_env}"
: "${HRX_RELEASE_TAG:?missing HRX_RELEASE_TAG in $release_env}"
: "${HRX_RELEASE_ASSET:?missing HRX_RELEASE_ASSET in $release_env}"
: "${HRX_RELEASE_SHA256:?missing HRX_RELEASE_SHA256 in $release_env}"

out_dir="${1:-$root/.hrx-release}"
mkdir -p "$out_dir"

base_url="https://github.com/${HRX_RELEASE_REPO}/releases/download/${HRX_RELEASE_TAG}"
tarball="$out_dir/$HRX_RELEASE_ASSET"
sha_file="$tarball.sha256"

download() {
  local url="$1"
  local path="$2"
  echo "download: $url"
  curl -fsSL --retry 3 --retry-delay 2 -o "$path" "$url"
}

download "$base_url/$HRX_RELEASE_ASSET" "$tarball"
download "$base_url/$HRX_RELEASE_ASSET.sha256" "$sha_file"

remote_sha="$(awk '{print $1; exit}' "$sha_file")"
[[ -n "$remote_sha" ]] || die "empty remote sha256 file: $sha_file"
[[ "$remote_sha" == "$HRX_RELEASE_SHA256" ]] ||
  die "remote sha256 mismatch: expected $HRX_RELEASE_SHA256, got $remote_sha"

actual_sha="$(sha256sum "$tarball" | awk '{print $1}')"
[[ "$actual_sha" == "$HRX_RELEASE_SHA256" ]] ||
  die "tarball sha256 mismatch: expected $HRX_RELEASE_SHA256, got $actual_sha"

artifact_root_name="$(tar -tzf "$tarball" | sed -n '1s,/.*,,p')"
[[ -n "$artifact_root_name" ]] || die "could not determine tarball root"
case "$artifact_root_name" in
  .|..) die "invalid tarball root: $artifact_root_name" ;;
esac

artifact_root="$out_dir/$artifact_root_name"
rm -rf "$artifact_root"
tar -C "$out_dir" -xzf "$tarball"

[[ -f "$artifact_root/env.sh" ]] || die "missing $artifact_root/env.sh"
[[ -f "$artifact_root/HRX_BUILD/libhrx/src/libhrx/libhrx.so" ]] ||
  die "missing packaged libhrx.so"
[[ -f "$artifact_root/HRX_BUILD/libflatcc_runtime.a" ]] ||
  die "missing packaged libflatcc_runtime.a"

echo "HRX_ARTIFACT_ROOT=$artifact_root"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "root=$artifact_root" >> "$GITHUB_OUTPUT"
fi