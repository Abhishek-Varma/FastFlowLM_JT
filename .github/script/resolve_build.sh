#!/usr/bin/env bash
# Decides where one platform's portable builds come from for the model sweep,
# and does not return until those builds actually exist. Writes these step
# outputs, each left unset when that build is unavailable:
#
#   run_id                    the CI run holding the artifacts (run mode only)
#   ready                     the default portable build
#   hrx_ready                 the HRX portable build
#   models_from_source_ready  the models-from-source build (run mode only)
#
# Usage: resolve_build.sh <linux|windows>
#
# Environment:
#   MODE            "release" or "run", decided by the resolve-source job
#   RELEASE_TAG     the release to check, in release mode
#   CANDIDATE_SHAS  commits whose build run to wait for, in run mode
#   FINISH_MINUTES  how long to wait for that run to complete, in run mode
#   GH_TOKEN, GITHUB_REPOSITORY, GITHUB_OUTPUT  as set by Actions
set -euo pipefail

platform="$1"
case "${platform}" in
  linux)
    workflow=debian-portable.yml
    label=Linux
    # Release assets are anchored at the start of the name so an
    # HRX_-prefixed asset is never mistaken for the default build.
    release_asset='^fastflowlm_.*_linux\.tar\.gz$'
    release_hrx_asset='^HRX_fastflowlm_.*_linux\.tar\.gz$'
    artifact='^portable$'
    hrx_artifact='^HRX-portable$'
    source_artifact='^models-from-source-portable$'
    ;;
  windows)
    workflow=windows-build.yml
    label=Windows
    release_asset='^fastflowlm_.*_windows_amd64\.zip$'
    release_hrx_asset='^HRX_fastflowlm_.*_windows_amd64\.zip$'
    # Fully anchored, so "fastflowlm-windows-<sha>" picks up neither the
    # HRX- prefixed artifact nor the MSI job's "fastflowlm-windows-msi-<sha>".
    artifact='^fastflowlm-windows-[0-9a-f]+$'
    hrx_artifact='^HRX-fastflowlm-windows-[0-9a-f]+$'
    source_artifact='^models-from-source-fastflowlm-windows-[0-9a-f]+$'
    ;;
  *)
    echo "::error::Unknown platform '${platform}'."
    exit 2
    ;;
esac

# Sets <flag>=true when <pattern> matches one of <names>, and otherwise
# reports the build as missing at the given severity.
check() {
  local names="$1" pattern="$2" flag="$3" what="$4" severity="$5"
  if printf '%s\n' "${names}" | grep -qE "${pattern}"; then
    echo "${flag}=true" >> "$GITHUB_OUTPUT"
    echo "  ${what}: ready"
  else
    echo "::${severity}::No ${what}; skipping that sweep."
  fi
}

# --- Release mode: a dispatch sweeps a shipped build ------------------------
if [ "${MODE}" = "release" ]; then
  # Fails the job on a typo'd tag, which is the right outcome: the operator
  # asked for something specific that does not exist.
  assets=$(gh api "repos/${GITHUB_REPOSITORY}/releases/tags/${RELEASE_TAG}" \
    --jq '.assets[].name')
  echo "Assets on ${RELEASE_TAG}:"
  printf '%s\n' "${assets}" | sed 's/^/  /'

  # A release can legitimately ship one platform and not the other, so a
  # missing asset skips that sweep instead of failing.
  check "${assets}" "${release_asset}" ready \
    "${label} portable asset on ${RELEASE_TAG}" warning
  # HRX assets ship from v1.0.8 on, and a failed HRX build leaves a release
  # without them, so their absence is a notice, not a warning.
  check "${assets}" "${release_hrx_asset}" hrx_ready \
    "${label} HRX portable asset on ${RELEASE_TAG}" notice
  exit 0
fi

# --- Run mode: a PR or a src/** push sweeps this commit's build -------------
POLL_SECONDS=20
# Separate windows: a build that never starts is a lost cause well before one
# that is merely slow or queued behind other runs.
APPEAR_DEADLINE=$(( SECONDS + 10 * 60 ))
FINISH_DEADLINE=$(( SECONDS + FINISH_MINUTES * 60 ))

# Newest run for whichever candidate SHA matches, or empty.
find_run() {
  local sha run_id
  for sha in ${CANDIDATE_SHAS}; do
    run_id=$(gh api \
      "repos/${GITHUB_REPOSITORY}/actions/workflows/${workflow}/runs?head_sha=${sha}&per_page=100" \
      --jq 'if (.workflow_runs | length) == 0 then "" else (.workflow_runs | sort_by(.run_number) | last | .id) end')
    if [ -n "${run_id}" ]; then
      printf '%s' "${run_id}"
      return 0
    fi
  done
}

while :; do
  run_id=$(find_run)
  [ -n "${run_id}" ] && break
  if [ "${SECONDS}" -ge "${APPEAR_DEADLINE}" ]; then
    echo "::warning::No ${workflow} run appeared for this commit; skipping the ${label} sweeps."
    exit 0
  fi
  echo "Waiting for a ${workflow} run to appear..."
  sleep "${POLL_SECONDS}"
done

# Waits for the whole run rather than the individual build jobs: artifacts
# are only listable once their job has uploaded them, and the run completing
# is the one event that guarantees that for all of them.
while :; do
  status=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}" --jq '.status')
  [ "${status}" = "completed" ] && break
  if [ "${SECONDS}" -ge "${FINISH_DEADLINE}" ]; then
    echo "::warning::${workflow} run ${run_id} is still ${status} after ${FINISH_MINUTES} minutes; skipping the ${label} sweeps."
    exit 0
  fi
  echo "${workflow} run ${run_id} is ${status}; checking again in ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
done

artifacts=$(gh api --paginate \
  "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}/artifacts?per_page=100" \
  --jq '.artifacts[] | select(.expired | not) | .name')
echo "${workflow} run ${run_id} published:"
printf '%s\n' "${artifacts}" | sed 's/^/  /'

# Written even when no artifact landed; the ready flags are what gate the
# sweeps, and an unused run id costs nothing.
echo "run_id=${run_id}" >> "$GITHUB_OUTPUT"

# The upload step is the last thing each build job does and carries no
# "if: always()", so an artifact exists only if that job got all the way
# through. That makes artifact presence a better readiness signal than the
# run's conclusion, which is a single verdict over every job in the workflow:
# a failed HRX build, or a failed .deb build, would otherwise take the default
# portable sweep down with it.
check "${artifacts}" "${artifact}" ready \
  "${label} portable artifact in run ${run_id}" warning
check "${artifacts}" "${hrx_artifact}" hrx_ready \
  "${label} HRX portable artifact in run ${run_id}" warning
check "${artifacts}" "${source_artifact}" models_from_source_ready \
  "${label} models-from-source artifact in run ${run_id}" warning
