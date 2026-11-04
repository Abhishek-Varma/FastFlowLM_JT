#!/usr/bin/env bash
# phase_sweep_shim.sh — Linux convenience wrapper around the phase sweep.
#
# Runs every text model through `flm bench <tag> -i <cfg>` with
# FLM_PUREHRX_PROFILE=1 (phase-tagged device-wait accounting) at 1k ctx / 20
# iters, using the *default* (flm_rt SHIM) HRX engines deployed in src/lib/hrx
# that carry our perf_improve cherry-picks + [hrxphase] instrumentation.
# Serial + wedge-guarded (retry 3x, 20s cooldown) so the NPU never wedges.
#
# All paths are env-overridable (defaults match the reference layout in
# REGENERATE.md). For a cross-platform (Windows) run, use phase_sweep.py instead.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# --- reference-layout defaults (override via env) ---
: "${FLM_BIN:=$REPO/src/build/flm}"
: "${FLM_BENCH_CFG:=$HERE/bench_1k_20.json}"
: "${FLM_SWEEP_OUT:=$HERE/sweep_shim}"
: "${FLM_ENGINE_LIBDIR:=$REPO/src/lib/hrx}"
: "${FLM_HRX_LIBDIR:=}"           # e.g. .../hrx-fastflow-sdk-<hash>/lib
export FLM_CONFIG_PATH="${FLM_CONFIG_PATH:-$REPO/src/model_list.json}"
export LD_LIBRARY_PATH="$FLM_HRX_LIBDIR:$FLM_ENGINE_LIBDIR:$REPO/src/lib:${LD_LIBRARY_PATH:-}"
# FLM_MODELINFO_PATH / FLM_XCLBIN_PATH / FLM_MODEL_PATH are passed through if set.

mkdir -p "$FLM_SWEEP_OUT"

TAGS="${FLM_SWEEP_TAGS:-
gemma3:1b gemma3:4b gemma4-it:12b gemma4-it:e2b gemma4-it:e4b gemma4e-flash:e4b
gpt-oss:20b hy-mt2-flash:1.8b lfm2:1.2b lfm2:2.6b llama3.2:1b llama3.2:3b
llama3.1:8b nanbeige4.1:3b phi4-mini-it:4b qwen2.5-it:3b qwen2.5vl-it:3b
qwen3:0.6b qwen3:1.7b qwen3:4b qwen3:8b qwen3vl-it:4b qwen3vl-flash:4b
qwen3.5:0.8b qwen3.5:2b qwen3.5:4b qwen3.5:9b qwen3.6-moe:35b-a3b
}"

wedged () { grep -qaE "ert state|hrx\]\[ERROR\]|did not complete" "$1" && return 0
            ! grep -qa "hrxphase" "$1"; }

for tag in $TAGS; do
  [ -z "$tag" ] && continue
  safe="${tag//[:\/]/_}"
  log="$FLM_SWEEP_OUT/${safe}.log"
  ok=0
  for attempt in 1 2 3; do
    echo "[$(date +%H:%M:%S)] $tag attempt $attempt ..."
    FLM_PUREHRX_PROFILE=1 "$FLM_BIN" bench "$tag" -i "$FLM_BENCH_CFG" > "$log" 2>&1
    if wedged "$log"; then echo "  WEDGE/incomplete -> cooldown 20s"; sleep 20; continue; fi
    echo "  clean"; ok=1; break
  done
  [ "$ok" = 1 ] || echo "  !! $tag FAILED after 3 attempts"
  sleep 3
done
echo "=== SHIM SWEEP DONE -> $FLM_SWEEP_OUT ==="
