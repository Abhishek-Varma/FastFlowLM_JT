#!/usr/bin/env python3
"""Parse the [hrxphase] blocks from the phase sweep logs and emit the four
report tables (prefill-all, per-token decode, all-token decode, whole-run).

Usage: parse_phase.py <sweep_dir> > report_fragment.md
"""
import sys, os, re, glob, json

def parse_log(path):
    txt = open(path, errors="replace").read()
    m = re.search(r"\[hrxphase\] engine=(\S+) iters=(\d+)", txt)
    if not m:
        return None
    d = {"tag": m.group(1), "iters": int(m.group(2))}
    def g(pat, cast=float):
        mm = re.search(pat, txt)
        return cast(mm.group(1)) if mm else None
    d["prompt_tokens"] = g(r"prompt_tokens_total=(\d+)", int)
    d["gen_tokens"]    = g(r"gen_tokens_total=(\d+)", int)
    d["pf_wall"] = g(r"prefill_wall_ms_total=([\d.]+)")
    d["pf_npu"]  = g(r"prefill_npu_ms_total=([\d.]+)")
    d["pf_waits"]= g(r"prefill_waits=(\d+)", int)
    d["de_wall"] = g(r"decode_wall_ms_total=([\d.]+)")
    d["de_npu"]  = g(r"decode_npu_ms_total=([\d.]+)")
    d["de_waits"]= g(r"decode_waits=(\d+)", int)
    d["other_npu"] = g(r"other_npu_ms_total=([\d.]+)")
    # bench summary (cross-check)
    sm = re.search(r"1k \|\s*([\d.]+)\s*±\s*[\d.]+\s*\|\s*([\d.]+)\s*±\s*[\d.]+\s*\|\s*([\d.]+)\s*±\s*[\d.]+", txt)
    if sm:
        d["ttft_s"] = float(sm.group(1)); d["prefill_tps"] = float(sm.group(2)); d["decode_tps"] = float(sm.group(3))
    return d

def pct(n, w):
    return 100.0 * n / w if w else 0.0

def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else "perf_prof/sweep"
    rows = []
    for f in sorted(glob.glob(os.path.join(sweep, "*.log"))):
        d = parse_log(f)
        if d: rows.append(d)
    if not rows:
        print("no [hrxphase] blocks found in", sweep); return

    def label(tag):
        return tag.replace("-purehrx", "").replace("_npu", "")

    # Table 1 — prefill all (per request / per iter)
    print("### Table 1 — Prefill (whole prefill phase, per request)\n")
    print("| model | prefill wall (ms) | prefill NPU (ms) | % NPU-bound | % Host | prompt tok |")
    print("|---|---|---|---|---|---|")
    for d in rows:
        it = d["iters"]
        w = d["pf_wall"]/it; n = d["pf_npu"]/it
        print(f"| {label(d['tag'])} | {w:.2f} | {n:.2f} | {pct(d['pf_npu'],d['pf_wall']):.1f}% | {pct(d['pf_wall']-d['pf_npu'],d['pf_wall']):.1f}% | {d['prompt_tokens']//it} |")

    # Table 2 — per-token decode
    print("\n### Table 2 — Per-token decode\n")
    print("| model | decode wall/token (ms) | decode NPU/token (ms) | % NPU-bound | % Host | tok/s |")
    print("|---|---|---|---|---|---|")
    for d in rows:
        g = d["gen_tokens"]
        w = d["de_wall"]/g; n = d["de_npu"]/g
        print(f"| {label(d['tag'])} | {w:.3f} | {n:.3f} | {pct(d['de_npu'],d['de_wall']):.1f}% | {pct(d['de_wall']-d['de_npu'],d['de_wall']):.1f}% | {1000.0/w:.1f} |")

    # Table 3 — all-token decode (per request = 32 tokens)
    print("\n### Table 3 — All-token decode (per request, 32 tokens)\n")
    print("| model | decode wall (ms) | decode NPU (ms) | % NPU-bound | % Host | tokens/iter |")
    print("|---|---|---|---|---|---|")
    for d in rows:
        it = d["iters"]
        w = d["de_wall"]/it; n = d["de_npu"]/it
        print(f"| {label(d['tag'])} | {w:.2f} | {n:.2f} | {pct(d['de_npu'],d['de_wall']):.1f}% | {pct(d['de_wall']-d['de_npu'],d['de_wall']):.1f}% | {d['gen_tokens']//it} |")

    # Table 4 — whole run (Option B): prefill + all decode
    print("\n### Table 4 — Whole-run (Option B: prefill + 32-token decode, per request)\n")
    print("| model | e2e wall (ms) | e2e NPU (ms) | % NPU-bound | % Host |")
    print("|---|---|---|---|---|")
    for d in rows:
        it = d["iters"]
        wall = (d["pf_wall"]+d["de_wall"])/it
        npu  = (d["pf_npu"]+d["de_npu"]+(d["other_npu"] or 0))/it
        print(f"| {label(d['tag'])} | {wall:.2f} | {npu:.2f} | {pct(npu*it,d['pf_wall']+d['de_wall']):.1f}% | {pct((d['pf_wall']+d['de_wall'])-npu*it,d['pf_wall']+d['de_wall']):.1f}% |")

    # dump raw json for the report author
    print("\n<!-- raw: " + json.dumps(rows) + " -->")

if __name__ == "__main__":
    main()
