#!/usr/bin/env bash
# phase_sweep.sh — run every pure-HRX model through the bench with phase-tagged
# device-wait accounting (FLM_PUREHRX_PROFILE=1) at 1k ctx / 20 iters / default
# flush stride. Captures the per-model [hrxphase] summary block used to build the
# four report tables (prefill-all, per-token decode, all-token decode, Option B).
# Wedge-guarded: retry up to 3x with a 20s cooldown.
set -uo pipefail
ROOT=/home/nod/avarma_repro
export FLM_CONFIG_PATH="$ROOT/FastFlowLM_JT/src/model_list.json"
export FLM_MODELINFO_PATH="$ROOT/pr714_hrx_0921/extracted/model_info.json"
export FLM_XCLBIN_PATH="$ROOT/pr714_hrx_0921/extracted/xclbins"
export FLM_MODEL_PATH="$ROOT/rocm-npu-staging/build/FastFlowLM-models"
FLM="$ROOT/FastFlowLM_JT/src/build/flm"
CFG="$ROOT/bench_1k_20.json"
OUT="$ROOT/perf_prof/sweep"; mkdir -p "$OUT"

TAGS="
gemma3-text-purehrx:1b
gemma3-purehrx:4b
gemma4-12b-purehrx:12b
gemma4e-purehrx:e4b
gemma4e-flash-purehrx:e4b
gpt-oss-purehrx:20b
hunyuan-purehrx:1.8b
lfm2-purehrx:1.2b
llama3-purehrx:1b
nanbeige-purehrx:3b
phi4-purehrx:4b
qwen2-purehrx:3b
qwen2vl-purehrx:3b
qwen3-purehrx:0.6b
qwen3vl-flash-purehrx:4b
qwen3vl-purehrx:4b
qwen3-purehrx:1.7b
qwen3-purehrx:4b
qwen3-purehrx:8b
llama3-purehrx:3b
llama3-purehrx:8b
lfm2-purehrx:2.6b
gemma4e-purehrx:e2b
qwen3.5-purehrx:0.8b
qwen3.5-purehrx:2b
qwen3.5-purehrx:4b
qwen3.5-purehrx:9b
qwen3.6-moe-purehrx:35b-a3b
"

wedged () { grep -qaE "ert state|hrx\]\[ERROR\]|did not complete" "$1" && return 0
            ! grep -qa "hrxphase" "$1"; }

for tag in $TAGS; do
  [ -z "$tag" ] && continue
  safe="${tag//[:\/]/_}"
  log="$OUT/${safe}.log"
  ok=0
  for attempt in 1 2 3; do
    echo "[$(date +%H:%M:%S)] $tag attempt $attempt ..."
    FLM_PUREHRX_PROFILE=1 FLM_PUREHRX_PROFILE_EVERY=128 "$FLM" bench "$tag" -i "$CFG" > "$log" 2>&1
    if wedged "$log"; then echo "  WEDGE/incomplete -> cooldown 20s"; sleep 20; continue; fi
    echo "  clean"; ok=1; break
  done
  [ "$ok" = 1 ] || echo "  !! $tag FAILED after 3 attempts"
  sleep 2
done
echo "=== SWEEP DONE -> $OUT ==="
