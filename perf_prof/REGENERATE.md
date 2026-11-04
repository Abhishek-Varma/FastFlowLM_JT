# Regenerating the NPU-vs-Host phase tables

This directory contains everything an agent needs to reproduce the
`NPU_VS_HOST_PHASE_TABLES_SHIPPED.md` report (host/device split for every
shipped default-HRX text engine) on their own machine — the sweep driver, the
parser, the bench workload, and this guide.

> **Branch:** this harness lives on `app_total_perf_improve`, which carries the
> `[hrxphase]` profiler (`src/include/hrx_cpp/hrx_cpp.hpp`,
> `src/src/benchmarking.hpp`, `src/CMakeLists.txt`) plus the optimized +
> instrumented default HRX engines in `src/lib/hrx/*.so`.

---

## 0. What gets measured

Each `flm bench` run (with `FLM_PUREHRX_PROFILE=1`) prints one `[hrxphase]`
block. Every NPU dispatch in both the engines and the app funnels through a
single phase-tagged `hrx::timed_stream_wait()`, which times the host-blocked
wall inside `hrx_stream_wait` and accumulates it into a process-wide
`PhaseAccum` (a function-local static with STB_GNU_UNIQUE linkage, merged across
the `flm` exe and every dlopen'd engine `.so`). The bench loop tags each phase
PREFILL / DECODE / OTHER and prints the per-phase totals.

### Metrics / formula set

| quantity       | definition                      | meaning |
|----------------|---------------------------------|---------|
| `wall`         | measured wall time per phase    | elapsed |
| `npu_wait`     | host time blocked in `hrx_stream_wait` | host-waiting-for-device (today's "NPU ms") |
| `npu_busy`     | **true device-busy time** (device timeline) | NPU actually executing — *optional, see §4* |
| `pure_host`    | `wall - npu_wait`               | host not blocked on the stream |
| `reclaimable`  | `wall - npu_busy`               | host active **while the NPU is genuinely idle** → the real target |
| `host_overlap` | `npu_busy - npu_wait` (= `pure_host - reclaimable`) | host work **already hidden** behind NPU execution |

**How to read it for targeting:** rank models by `reclaimable` (absolute ms,
not %). High `reclaimable` with few waits ⇒ a fat redundant sync to delete; high
`reclaimable` with many waits ⇒ per-dispatch overhead ⇒ pipeline / ping-pong.
`host_overlap` tells you how well a model is *already* pipelined (higher = less
to gain).

> **Important:** `npu_wait` is host-waiting-for-device. It does **not** include
> "NPU idle waiting on host" (that time is in the host bucket), and it can
> *under*-count true NPU work when an op is submitted non-blocking (async
> lm_head in decode). That gap is exactly `host_overlap`, which is why it needs
> `npu_busy` (§4) to be non-zero.

---

## 1. Build the instrumented `flm` (HRX)

```bash
cd src
cmake --preset linux-default -DFLM_USE_HRX=ON -Dhrx_DIR=/path/to/hrx-fastflow-sdk-<hash>
cmake --build --preset linux-default -j
# -> src/build/flm   (Windows: configure the msvc preset; -> src/build/flm.exe)
```

`CMakeLists.txt` sets `ENABLE_EXPORTS ON` on the `flm` target under
`FLM_USE_HRX` so the `PhaseAccum` symbol lands in the dynamic symbol table and
merges with the engine `.so`s.

## 2. Engines

The optimized + instrumented default HRX engines are already committed under
`src/lib/hrx/*.so` on this branch — no rebuild needed to reproduce the current
report. To regenerate them from source, build `FLM_DLL` on the IRON
`total_perf_improve` branch and copy the `.so` into `src/lib/hrx/`.

## 3. Run the sweep + parse

**Linux:**
```bash
export FLM_HRX_LIBDIR=/path/to/hrx-fastflow-sdk-<hash>/lib
export FLM_MODEL_PATH=/path/to/FastFlowLM-models
export FLM_MODELINFO_PATH=/path/to/model_info.json
export FLM_XCLBIN_PATH=/path/to/xclbins
bash perf_prof/phase_sweep_shim.sh            # -> perf_prof/sweep_shim/*.log
python3 perf_prof/parse_phase.py perf_prof/sweep_shim > perf_prof/tables_fragment.md
```

**Windows / cross-platform:**
```bat
set FLM_HRX_LIBDIR=C:\path\to\hrx-fastflow-sdk-<hash>\lib
set FLM_MODEL_PATH=C:\path\to\FastFlowLM-models
python perf_prof\phase_sweep.py
python perf_prof\parse_phase.py perf_prof\sweep_shim > perf_prof\tables_fragment.md
```

`parse_phase.py` emits the four tables (prefill / per-token decode / all-token
decode / whole-run) plus a `<!-- raw: [...] -->` JSON dump for the report author.
The columns adapt to the data:

- **no device-busy data** (today): compact schema
  `wall | total_npu | pure_host | % host`, where `pure_host = wall - total_npu`.
- **device-busy present** (after §4): full schema
  `wall | npu_busy | pure_host | reclaimable | host_overlap | % reclaimable`.

It is pure Python and tolerates CRLF / mixed encodings, so it runs identically on
both OSes. Paste the fragment into `NPU_VS_HOST_PHASE_TABLES_SHIPPED.md`.

### Environment variables

| var | default | purpose |
|-----|---------|---------|
| `FLM_BIN` | `src/build/flm[.exe]` | flm binary |
| `FLM_BENCH_CFG` | `perf_prof/bench_1k_20.json` | workload (1k ctx, 20 iters) |
| `FLM_SWEEP_OUT` | `perf_prof/sweep_shim` | per-model log dir |
| `FLM_SWEEP_TAGS` | built-in 28-tag list | models to sweep |
| `FLM_HRX_LIBDIR` | — | HRX runtime lib dir (loader path) |
| `FLM_ENGINE_LIBDIR` | `src/lib/hrx` | deployed engine lib dir (loader path) |
| `FLM_MODEL_PATH` / `FLM_MODELINFO_PATH` / `FLM_XCLBIN_PATH` / `FLM_CONFIG_PATH` | — | passed through to flm |

---

## 4. TODO: light up `npu_busy` (true device-busy → real `host_overlap`)

Today the engines emit only `npu_wait` (host-blocked). So the parser falls back
to `npu_busy = npu_wait`, which forces `host_overlap = 0` and
`reclaimable = pure_host`. That is exact for serial prefill but **over-estimates**
reclaimable for the async-lm_head decode window.

To make `host_overlap` real, add a device-timeline busy measurement and have the
harness print `prefill_npu_busy_ms_total=` / `decode_npu_busy_ms_total=` in the
`[hrxphase]` block. `parse_phase.py` already parses those optional fields and
will switch to real `reclaimable` / `host_overlap` automatically once present.
Candidate sources: HRX hardware/ERT packet timestamps per dispatch, or
submit→completion timestamps for async ops. This requires an engine/app change,
a rebuild, and a fresh sweep (it **cannot** be derived from existing logs).
