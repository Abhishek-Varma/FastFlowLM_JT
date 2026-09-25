# Pure/Native HRX vs. `flm_rt` — Qwen3-0.6B NPU Benchmark

Comparison of the stock **`flm_rt`** dispatch path against the new **pure/native
HRX** dispatch path for the smallest Qwen model (`Qwen3-0.6B-NPU2`), both built
against the pinned HRX release and run on the same NPU from a single `flm.exe`.

## What "pure/native HRX" means here

The stock engine (`qwen3_npu`) drives the NPU through the `flm_rt` compile-time
alias, i.e. the `hrx::run` / `hrx::runlist` C++ shim over the raw HRX C API. The
new engine (`qwen3_npu_pure_hrx`) bypasses that shim on the hot paths and calls the
**native HRX C API directly** — `hrx_stream_dispatch` / `hrx_stream_flush` /
`hrx_stream_wait` — while preserving the same coherence bookkeeping
(`hrx_h2d_bindings` / `hrx_mark_dispatched`) so multi-turn KV-cache correctness is
retained.

Both engines are compiled to their own DLLs, statically linked into the same
`flm.exe`, and selected at runtime via the model tag:

- `qwen3:0.6b` → `Qwen3` → `qwen3_npu` (flm_rt)
- `qwen3-purehrx:0.6b` → `Qwen3PureHrx` → `qwen3_npu_pure_hrx` (native HRX)

## Correctness (pure/native HRX engine)

Verified via the REST server before benchmarking:

- Single-turn: *"What is the capital of France?"* → **"The capital of France is Paris."**
- Multi-turn (KV-cache coherence): remembered the name across turns and answered
  *"12 multiplied by 8 is 96. My name is Abhishek."*

## Environment

| Item | Value |
| --- | --- |
| Device | AMD Ryzen AI MAX+ PRO 395 w/ Radeon 8060S (Strix NPU, XDNA2) |
| Model | Qwen3-0.6B-NPU2 (same weights for both tags) |
| HRX runtime | pinned `flm-hrx-amdxdna-v2026.09.15` (`hrx.dll`) |
| flm.exe | v1.0.5, `FLM_USE_HRX=ON` |
| Config | 1k context (`max_length: 1024`), **20 iterations**, `bench_1k_20.json` |
| Metric | prefill = prompt_tokens / prefill_duration; decode = generated_tokens / decoding_duration |

## Results (1k context, 20 iterations)

Values are `avg ± std` (min / max) over 20 iterations.

| Metric | `flm_rt` (`qwen3:0.6b`) | pure HRX (`qwen3-purehrx:0.6b`) | Delta (pure − flm_rt) |
| --- | --- | --- | --- |
| **Prefill (tok/s)** | 883.75 ± 94.03 (796.83 / 1177.92) | 877.37 ± 95.57 (716.78 / 1196.71) | **−6.38 tok/s (−0.72%)** |
| **Decode (tok/s)** | 39.78 ± 3.11 (33.57 / 48.21) | 39.87 ± 5.08 (30.34 / 55.08) | **+0.09 tok/s (+0.23%)** |
| TTFT (s) | 1.126 ± 0.099 (0.838 / 1.238) | 1.135 ± 0.106 (0.825 / 1.374) | +0.009 s (+0.82%) |

## Takeaway

At 1k context the pure/native HRX path is **performance-neutral** versus the
`flm_rt` shim: prefill is within ~0.7% and decode within ~0.2%, both well inside
run-to-run variance (overlapping std). This is expected — the `flm_rt`
`hrx::run`/`runlist` wrapper is a thin layer over the same `hrx_stream_*` calls the
pure path now issues directly, so removing it neither adds nor removes measurable
NPU work at this size. The value of the pure path is a direct, dependency-light
dispatch surface (no C++ shim on the hot loop) with identical output and
multi-turn coherence.

## Raw CSVs

- `run105_15/bin/bench_qwen3_0.6b_20260925_AMD_RYZEN_AI_MAX__PRO_395_w__Radeon_8060S.csv`
- `run105_15/bin/bench_qwen3-purehrx_0.6b_20260925_AMD_RYZEN_AI_MAX__PRO_395_w__Radeon_8060S.csv`
