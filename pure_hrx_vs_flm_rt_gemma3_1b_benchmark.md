# Pure/Native HRX vs. `flm_rt` — Gemma3-1B NPU Benchmark

Comparison of the stock **`flm_rt`** dispatch path against the new **pure/native
HRX** dispatch path for the smallest Gemma3 text model (`Gemma3-1B-NPU2`), both
built against the pinned HRX release and run on the same NPU from a single
`flm.exe`.

## What "pure/native HRX" means here

The stock engine (`gemma_text_npu`) drives the NPU through the `flm_rt`
compile-time alias, i.e. the `hrx::run` / `hrx::runlist` C++ shim over the raw HRX
C API. The new engine (`gemma_text_npu_pure_hrx`) bypasses that shim on the hot
paths and calls the **native HRX C API directly** — `hrx_stream_dispatch` /
`hrx_stream_flush` / `hrx_stream_wait` — while preserving the same coherence
bookkeeping (`hrx_h2d_bindings` / `hrx_mark_dispatched`) so multi-turn KV-cache
correctness is retained.

Gemma3-text specifics that were ported to native dispatch:

- The per-token **decoder-layer chain** (`layers_run`), which mixes **sliding-window**
  and **full-attention** layers (`is_sliding_window(i)`), is now built as a
  `std::vector<pure_hrx::native_run>` and submitted with a single
  `submit_chain` + `wait`.
- The **prefill dequant chain** (`dequant_all`, the 4 dequant kernels per layer) is
  likewise issued natively, keeping the existing overlap where the host RMS-norm
  runs while the NPU dequants.
- The `lm_head` stays on its own async `execute()`/`wait()` (the `GemmaTextLMHead`
  module) so that `set_context_length` — which recompiles the next token's layer
  sequence and rebuilds the native run vector — overlaps `lm_head`'s NPU
  execution. (Merging that host work after the drain regressed decode badly in the
  qwen3 experiment, so the overlap is preserved here too.)

Both engines are compiled to their own DLLs, statically linked into the same
`flm.exe`, and selected at runtime via the model tag:

- `gemma3:1b` → `Gemma3_Text_Only` → `gemma_text_npu` (flm_rt)
- `gemma3-purehrx:1b` → `Gemma3_Text_OnlyPureHrx` → `gemma_text_npu_pure_hrx` (native HRX)

## Correctness (pure/native HRX engine)

Verified via the REST server before benchmarking, and cross-checked against the
stock `gemma3:1b` with greedy decoding (`temperature = 0`) — the outputs are
**byte-identical**:

- Single-turn: *"What is the capital of France?"* → **"Paris is the capital of France."**
- Multi-turn (KV-cache coherence): correctly recalled *"Your name is Abhishek. And
  your favorite number is 7!"* across turns.

## Environment

| Item | Value |
| --- | --- |
| Device | AMD Ryzen AI MAX+ PRO 395 w/ Radeon 8060S (Strix NPU, XDNA2) |
| Model | Gemma3-1B-NPU2 (same weights for both tags) |
| HRX runtime | pinned `flm-hrx-amdxdna-v2026.09.15` (`hrx.dll`, commit c023c3e) |
| flm.exe | v1.0.5, `FLM_USE_HRX=ON` |
| Config | 1k context (`max_length: 1024`), **20 iterations**, `bench_1k_20.json` |
| Metric | prefill = prompt_tokens / prefill_duration; decode = generated_tokens / decoding_duration |

## Results (1k context, 20 iterations)

Values are `avg ± std` over 20 iterations.

| Metric | `flm_rt` (`gemma3:1b`) | pure HRX (`gemma3-purehrx:1b`) | Delta (pure − flm_rt) |
| --- | --- | --- | --- |
| **Prefill (tok/s)** | 641.00 ± 34.31 | 635.60 ± 27.61 | **−5.40 tok/s (−0.84%)** |
| **Decode (tok/s)** | 34.63 ± 0.58 | 34.45 ± 0.58 | **−0.18 tok/s (−0.52%)** |
| TTFT (s) | 1.533 ± 0.074 | 1.545 ± 0.061 | +0.012 s (+0.78%) |

## Takeaway

At 1k context the pure/native HRX path is **performance-neutral** versus the
`flm_rt` shim: prefill is within ~0.8% and decode within ~0.5%, both well inside
run-to-run variance (overlapping std). This matches the qwen3-0.6B result and is
expected — the `flm_rt` `hrx::run`/`runlist` wrapper is a thin layer over the same
`hrx_stream_*` calls the pure path now issues directly, so removing it neither adds
nor removes measurable NPU work at this size. The NPU kernels (attention, GEMM,
dequant) dominate wall-clock; the dispatch shim was never the bottleneck.

The value of the pure path here is a direct, dependency-light dispatch surface (no
C++ shim on the hot loop) with identical output and multi-turn coherence — and, for
the next step, an explicit `record` / `submit` / `wait` seam that makes it possible
to experiment with **asynchronous overlap** of NPU dispatch and host work at the
engine level.
