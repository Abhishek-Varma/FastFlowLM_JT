# NPU vs Host Compute — Phase Breakdown (FLM pure-HRX models)

For every model runnable on FastFlowLM's pure/native-HRX path we measure how much
of the wall-clock time is spent **waiting on the NPU** vs. doing **host-side work**
(tokenization, sampling, KV bookkeeping, dispatch/launch overhead, C++ glue, etc.).

The four tables below cover the four scopes requested:

1. **Prefill (whole prefill phase)**
2. **Per-token decode**
3. **All-token decode** (full 32-token generation per request)
4. **Whole-run / Option B** (prefill + 32-token decode, end-to-end)

---

## Methodology

- **Harness:** `flm bench <tag> -i bench_1k_20.json` with `FLM_PUREHRX_PROFILE=1`.
- **Workload:** **1k context**, **20 iterations** per model (to stabilize profiled data),
  **default stride** (`FLM_PUREHRX_FLUSH_STRIDE` default = 1). Each iteration runs one
  prefill (`insert`) followed by a 32-token decode (`generate`).
- **NPU time** = wall time spent inside device stream waits (`hrx_stream_wait`). Every
  NPU dispatch in the native-HRX engines funnels through a single phase-tagged
  `hrx::timed_stream_wait()`, tagged PREFILL / DECODE from the bench loop. This captures
  all layer compute, attention, and lm_head launches.
- **Host time** = wall − NPU for the same phase: everything the CPU does that is *not*
  blocked on the accelerator.
- `% NPU-bound = NPU / wall`, `% Host = 1 − % NPU-bound`.
- Per-request numbers are 20-iteration totals ÷ 20; per-token numbers divide the decode
  phase by generated tokens.

> **Reading the split:** high **% NPU-bound** = accelerator-limited (host keeps up, little
> headroom from faster CPU work). Higher **% Host** = a meaningful slice of wall time is
> spent on CPU-side orchestration between NPU dispatches.

> **Important:** this instrumentation only sees the **native/pure-HRX** device-wait path.
> The regular (non-pure-HRX) engines dispatch through the `flm_rt` shim, which does **not**
> hit the instrumented wait — a regular tag reports 0 NPU / 100% host. So every row below
> was run through a pure-HRX engine. The NPU/host split is primarily a function of
> **architecture × size**, not the specific fine-tune, so one row represents all fine-tunes
> that share that architecture and size (see Coverage).

---

## Table 1 — Prefill (whole prefill phase, per request)

| model | prefill wall (ms) | prefill NPU (ms) | % NPU-bound | % Host | prompt tok |
|---|---|---|---|---|---|
| gemma3:4b | 980.05 | 848.33 | 86.6% | 13.4% | 53 |
| gemma3-text:1b | 587.89 | 496.31 | 84.4% | 15.6% | 53 |
| gemma4-12b:12b | 1695.63 | 1535.73 | 90.6% | 9.4% | 53 |
| gemma4e-flash:e4b | 355.29 | 298.16 | 83.9% | 16.1% | 53 |
| gemma4e:e2b | 868.20 | 794.84 | 91.6% | 8.4% | 53 |
| gemma4e:e4b | 1252.15 | 1135.20 | 90.7% | 9.3% | 53 |
| gpt-oss:20b | 3885.63 | 2652.32 | 68.3% | 31.7% | 114 |
| hunyuan:1.8b | 130.51 | 111.49 | 85.4% | 14.6% | 49 |
| lfm2:1.2b | 402.90 | 338.43 | 84.0% | 16.0% | 65 |
| lfm2:2.6b | 784.50 | 672.55 | 85.7% | 14.3% | 65 |
| llama3:1b | 376.57 | 330.89 | 87.9% | 12.1% | 81 |
| llama3:3b | 773.55 | 688.11 | 89.0% | 11.0% | 81 |
| llama3:8b | 1441.53 | 1299.23 | 90.1% | 9.9% | 81 |
| nanbeige:3b | 951.50 | 840.99 | 88.4% | 11.6% | 76 |
| phi4:4b | 913.04 | 824.51 | 90.3% | 9.7% | 50 |
| qwen2:3b | 931.12 | 856.65 | 92.0% | 8.0% | 75 |
| qwen2vl:3b | 942.85 | 829.70 | 88.0% | 12.0% | 65 |
| qwen3:0.6b | 487.00 | 436.06 | 89.5% | 10.5% | 59 |
| qwen3:1.7b | 637.57 | 572.54 | 89.8% | 10.2% | 59 |
| qwen3:4b | 1052.08 | 939.82 | 89.3% | 10.7% | 59 |
| qwen3:8b | 1490.04 | 1346.18 | 90.3% | 9.7% | 59 |
| qwen3vl-flash:4b | 239.53 | 202.04 | 84.3% | 15.7% | 54 |
| qwen3vl:4b | 1047.68 | 942.05 | 89.9% | 10.1% | 54 |

> `gpt-oss:20b` is the clear outlier at **31.7% host** — larger prompt (114 tok) plus MoE
> routing/dispatch overhead leave the biggest host-side slice during prefill.

---

## Table 2 — Per-token decode

| model | decode wall/token (ms) | decode NPU/token (ms) | % NPU-bound | % Host | tok/s |
|---|---|---|---|---|---|
| gemma3:4b | 57.452 | 53.751 | 93.6% | 6.4% | 17.4 |
| gemma3-text:1b | 26.989 | 23.824 | 88.3% | 11.7% | 37.1 |
| gemma4-12b:12b | 166.107 | 156.097 | 94.0% | 6.0% | 6.0 |
| gemma4e-flash:e4b | 76.904 | 71.965 | 93.6% | 6.4% | 13.0 |
| gemma4e:e2b | 43.820 | 39.476 | 90.1% | 9.9% | 22.8 |
| gemma4e:e4b | 77.264 | 72.309 | 93.6% | 6.4% | 12.9 |
| gpt-oss:20b | 57.266 | 55.968 | 97.7% | 2.3% | 17.5 |
| hunyuan:1.8b | 22.596 | 21.362 | 94.5% | 5.5% | 44.3 |
| lfm2:1.2b | 16.581 | 14.225 | 85.8% | 14.2% | 60.3 |
| lfm2:2.6b | 34.228 | 31.156 | 91.0% | 9.0% | 29.2 |
| llama3:1b | 16.789 | 16.136 | 96.1% | 3.9% | 59.6 |
| llama3:3b | 47.398 | 46.449 | 98.0% | 2.0% | 21.1 |
| llama3:8b | 92.529 | 91.681 | 99.1% | 0.9% | 10.8 |
| nanbeige:3b | 45.924 | 42.739 | 93.1% | 6.9% | 21.8 |
| phi4:4b | 50.185 | 49.296 | 98.2% | 1.8% | 19.9 |
| qwen2:3b | 41.513 | 40.533 | 97.6% | 2.4% | 24.1 |
| qwen2vl:3b | 42.234 | 39.382 | 93.2% | 6.8% | 23.7 |
| qwen3:0.6b | 11.562 | 10.881 | 94.1% | 5.9% | 86.5 |
| qwen3:1.7b | 23.405 | 22.313 | 95.3% | 4.7% | 42.7 |
| qwen3:4b | 54.015 | 52.518 | 97.2% | 2.8% | 18.5 |
| qwen3:8b | 94.688 | 93.027 | 98.2% | 1.8% | 10.6 |
| qwen3vl-flash:4b | 52.901 | 49.557 | 93.7% | 6.3% | 18.9 |
| qwen3vl:4b | 51.782 | 48.612 | 93.9% | 6.1% | 19.3 |

> Decode is overwhelmingly NPU-bound, and the host share **shrinks as model size grows**:
> `llama3` 1b→3b→8b = 96.1%→98.0%→**99.1%** NPU; `qwen3` 0.6b→8b = 94.1%→**98.2%**. The fixed
> per-token host overhead becomes negligible once each token's NPU compute is large.

---

## Table 3 — All-token decode (per request, 32 tokens)

| model | decode wall (ms) | decode NPU (ms) | % NPU-bound | % Host | tokens/iter |
|---|---|---|---|---|---|
| gemma3:4b | 1838.47 | 1720.03 | 93.6% | 6.4% | 32 |
| gemma3-text:1b | 863.63 | 762.36 | 88.3% | 11.7% | 32 |
| gemma4-12b:12b | 5315.42 | 4995.10 | 94.0% | 6.0% | 32 |
| gemma4e-flash:e4b | 2441.70 | 2284.90 | 93.6% | 6.4% | 31 |
| gemma4e:e2b | 1402.25 | 1263.22 | 90.1% | 9.9% | 32 |
| gemma4e:e4b | 2472.45 | 2313.88 | 93.6% | 6.4% | 32 |
| gpt-oss:20b | 1832.51 | 1790.98 | 97.7% | 2.3% | 32 |
| hunyuan:1.8b | 723.08 | 683.58 | 94.5% | 5.5% | 32 |
| lfm2:1.2b | 530.58 | 455.20 | 85.8% | 14.2% | 32 |
| lfm2:2.6b | 1095.30 | 997.00 | 91.0% | 9.0% | 32 |
| llama3:1b | 444.07 | 426.80 | 96.1% | 3.9% | 26 |
| llama3:3b | 1516.74 | 1486.36 | 98.0% | 2.0% | 32 |
| llama3:8b | 2896.17 | 2869.62 | 99.1% | 0.9% | 31 |
| nanbeige:3b | 1469.57 | 1367.64 | 93.1% | 6.9% | 32 |
| phi4:4b | 1605.92 | 1577.46 | 98.2% | 1.8% | 32 |
| qwen2:3b | 1328.42 | 1297.04 | 97.6% | 2.4% | 32 |
| qwen2vl:3b | 1309.26 | 1220.83 | 93.2% | 6.8% | 31 |
| qwen3:0.6b | 369.99 | 348.19 | 94.1% | 5.9% | 32 |
| qwen3:1.7b | 748.94 | 714.02 | 95.3% | 4.7% | 32 |
| qwen3:4b | 1728.48 | 1680.58 | 97.2% | 2.8% | 32 |
| qwen3:8b | 3030.03 | 2976.86 | 98.2% | 1.8% | 32 |
| qwen3vl-flash:4b | 1692.82 | 1585.81 | 93.7% | 6.3% | 32 |
| qwen3vl:4b | 1657.03 | 1555.58 | 93.9% | 6.1% | 32 |

> Percentages match Table 2 (same phase, aggregated). A few models hit EOS slightly before
> 32 tokens in some iterations, so their tokens/iter average below 32.

---

## Table 4 — Whole-run (Option B: prefill + 32-token decode, per request)

| model | e2e wall (ms) | e2e NPU (ms) | % NPU-bound | % Host |
|---|---|---|---|---|
| gemma3:4b | 2818.52 | 2568.35 | 91.1% | 8.9% |
| gemma3-text:1b | 1451.53 | 1258.66 | 86.7% | 13.3% |
| gemma4-12b:12b | 7011.05 | 6530.82 | 93.2% | 6.8% |
| gemma4e-flash:e4b | 2796.99 | 2583.06 | 92.4% | 7.6% |
| gemma4e:e2b | 2270.44 | 2058.06 | 90.6% | 9.4% |
| gemma4e:e4b | 3724.60 | 3449.08 | 92.6% | 7.4% |
| gpt-oss:20b | 5718.13 | 4443.30 | 77.7% | 22.3% |
| hunyuan:1.8b | 853.59 | 795.07 | 93.1% | 6.9% |
| lfm2:1.2b | 933.48 | 793.62 | 85.0% | 15.0% |
| lfm2:2.6b | 1879.80 | 1669.55 | 88.8% | 11.2% |
| llama3:1b | 820.63 | 757.69 | 92.3% | 7.7% |
| llama3:3b | 2290.30 | 2174.46 | 94.9% | 5.1% |
| llama3:8b | 4337.70 | 4168.85 | 96.1% | 3.9% |
| nanbeige:3b | 2421.07 | 2208.63 | 91.2% | 8.8% |
| phi4:4b | 2518.96 | 2401.98 | 95.4% | 4.6% |
| qwen2:3b | 2259.54 | 2153.69 | 95.3% | 4.7% |
| qwen2vl:3b | 2252.12 | 2050.53 | 91.0% | 9.0% |
| qwen3:0.6b | 856.99 | 784.24 | 91.5% | 8.5% |
| qwen3:1.7b | 1386.52 | 1286.57 | 92.8% | 7.2% |
| qwen3:4b | 2780.56 | 2620.40 | 94.2% | 5.8% |
| qwen3:8b | 4520.07 | 4323.03 | 95.6% | 4.4% |
| qwen3vl-flash:4b | 1932.36 | 1787.85 | 92.5% | 7.5% |
| qwen3vl:4b | 2704.71 | 2497.63 | 92.3% | 7.7% |

> Whole-run sits at **90–96% NPU-bound** for most models. `gpt-oss:20b` (22.3% host) and the
> small/fast decoders (`lfm2:1.2b` 15.0%, `gemma3-text:1b` 13.3%) carry the biggest host
> slices.

---

## Coverage

**23 arch×size configurations** are profiled above, spanning **all 11 architecture families
that have a native pure-HRX engine**, at every model size whose weights are available
locally:

| architecture (pure-HRX engine) | sizes profiled |
|---|---|
| gemma3 (vision 4b) | 4b |
| gemma3-text | 1b |
| gemma4-12b | 12b |
| gemma4e | e2b, e4b |
| gemma4e-flash | e4b |
| gpt-oss | 20b |
| hunyuan | 1.8b |
| lfm2 | 1.2b, 2.6b |
| llama3 | 1b, 3b, 8b |
| nanbeige | 3b |
| phi4 | 4b |
| qwen2 | 3b |
| qwen2vl | 3b |
| qwen3 | 0.6b, 1.7b, 4b, 8b |
| qwen3vl | 4b |
| qwen3vl-flash | 4b |

### Reconciling against the full catalog

`model_list.json` exposes ~46 top-level tags / ~39 distinct checkpoints. The remainder are
**not** separate measurements for one of two reasons:

1. **Same architecture × size as a row above** (identical NPU/host split; only the fine-tune
   weights differ). These are covered by proxy:
   - `deepseek-r1-0528:8b`, `qwen3-tk:4b`, `qwen3-it:4b` → **qwen3** (8b / 4b)
   - `deepseek-r1:8b` → **llama3** 8b
   - `translategemma:4b`, `medgemma:4b`, `medgemma1.5:4b` → **gemma3** 4b
   - `gpt-oss-sg:20b` → **gpt-oss** 20b
   - `lfm2-trans:2.6b` → **lfm2** 2.6b; `lfm2.5-it/tk:1.2b` → **lfm2** 1.2b
   - the non-purehrx aliases (`qwen3:*`, `lfm2:*`, `gpt-oss:20b`, `gemma3:*`, `qwen2.5-it`,
     `qwen2.5vl-it`, `llama3.2`, `phi4-mini-it`, `hy-mt2`, `nanbeige4.1`, `qwen3vl-it`, …)
     point to the exact checkpoints already measured via their `-purehrx` twins.

2. **Distinct architectures with no pure-HRX engine yet** (would require a from-scratch
   engine port, like the gemma4e-flash port):
   - **`qwen3.5`** (0.8b / 2b / 4b / 9b) — hybrid *linear-attention + full-attention*
     architecture (`attn_output_gate`, `full_attention_interval`, `layer_types`), VL-capable.
     Its engine (`qwen3_5vl_npu`) is ~250 KB of custom prefill/sequence/rotary code; porting
     to pure-HRX is a substantial, higher-risk effort.
   - **`qwen3.6-moe:35b-a3b`** — Mixture-of-Experts (expert routing on top of the above);
     larger still (~21 GB) and the most involved port.
   - **`qwen3.5-omni`** — omni (audio+vision+text), likewise unported.

3. **Not prefill+decode LLMs** (don't fit this bench shape at all):
   `embed-gemma:300m` (embeddings) and `whisper-v3:turbo` (ASR).

### Raw per-model totals
20-iteration raw totals (prefill/decode wall & NPU ms, wait counts, token counts, TTFT,
tok/s) are embedded as a JSON comment by `perf_prof/parse_phase.py`; per-model bench logs
are in `perf_prof/sweep/*.log`.
