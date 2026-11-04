#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse the ``[hrxphase]`` blocks emitted by ``flm bench`` (with
``FLM_PUREHRX_PROFILE=1``) and emit the NPU-vs-host phase tables.

Cross-platform: pure Python, no shell-isms, tolerant of CRLF / mixed encodings,
so it runs identically on Linux and Windows (``py parse_phase.py sweep_dir``).

New formula set
---------------
For each phase we now report three host quantities instead of a single ``% Host``:

  * ``npu_wait``  = wall time the *host* is blocked inside ``hrx_stream_wait``
                    (host-waiting-for-device). This is what the current
                    instrumentation records as "NPU ms".
  * ``npu_busy``  = *true* device-busy time, read from the device timeline
                    (optional field ``*_npu_busy_ms_total`` in the log). When the
                    engines/app have not yet been instrumented to emit it, it is
                    absent and we fall back to ``npu_busy = npu_wait``.

  * ``pure_host``     = wall - npu_wait              (host not blocked on the stream)
  * ``reclaimable``   = wall - npu_busy              (host active while NPU genuinely idle)
  * ``host_overlap``  = npu_busy - npu_wait          (host work already hidden behind NPU)
                        == pure_host - reclaimable

Interpretation:
  * ``reclaimable`` is the real "which model to target next" lever — exposed host
    time where the NPU is idle (pipeline / ping-pong / delete redundant syncs).
  * ``host_overlap`` tells you how much host work is *already* hidden (good);
    it is only non-zero once ``npu_busy`` is measured from the device timeline.
  * Until ``npu_busy`` is instrumented, ``npu_busy == npu_wait`` so
    ``host_overlap == 0`` and ``reclaimable == pure_host`` (an upper bound on the
    true reclaimable pool — correct for serial prefill, an over-estimate for the
    async-lm_head decode window).

Usage: parse_phase.py <sweep_dir> [> report_fragment.md]
"""
import sys
import os
import re
import glob
import json


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    txt = txt.replace("\r\n", "\n")
    m = re.search(r"\[hrxphase\] engine=(\S+) iters=(\d+)", txt)
    if not m:
        return None
    d = {"tag": m.group(1), "iters": int(m.group(2))}

    def g(pat, cast=float):
        mm = re.search(pat, txt)
        return cast(mm.group(1)) if mm else None

    d["prompt_tokens"] = g(r"prompt_tokens_total=(\d+)", int)
    d["gen_tokens"] = g(r"gen_tokens_total=(\d+)", int)
    d["pf_wall"] = g(r"prefill_wall_ms_total=([\d.]+)")
    d["pf_npu"] = g(r"prefill_npu_ms_total=([\d.]+)")      # host-blocked wait
    d["pf_waits"] = g(r"prefill_waits=(\d+)", int)
    d["de_wall"] = g(r"decode_wall_ms_total=([\d.]+)")
    d["de_npu"] = g(r"decode_npu_ms_total=([\d.]+)")       # host-blocked wait
    d["de_waits"] = g(r"decode_waits=(\d+)", int)
    d["other_npu"] = g(r"other_npu_ms_total=([\d.]+)")
    # Optional true device-busy fields (present only once the engines/app are
    # instrumented with device-timeline timestamps). Absent -> None -> fall back.
    d["pf_npu_busy"] = g(r"prefill_npu_busy_ms_total=([\d.]+)")
    d["de_npu_busy"] = g(r"decode_npu_busy_ms_total=([\d.]+)")
    return d


def pct(n, w):
    return 100.0 * n / w if w else 0.0


def has_busy(rows):
    """True once at least one log carried device-busy timestamps."""
    return any(r.get("pf_npu_busy") is not None or r.get("de_npu_busy") is not None
               for r in rows)


def label(tag):
    return tag.replace("-purehrx", "").replace("_npu", "")


def phase_metrics(wall, npu_wait, npu_busy):
    """Return the host decomposition for one phase (all in ms)."""
    if npu_busy is None:
        npu_busy = npu_wait           # degrade gracefully (serial assumption)
    pure_host = wall - npu_wait
    reclaimable = wall - npu_busy
    host_overlap = npu_busy - npu_wait
    return {
        "wall": wall,
        "npu_wait": npu_wait,
        "npu_busy": npu_busy,
        "pure_host": pure_host,
        "reclaimable": reclaimable,
        "host_overlap": host_overlap,
    }


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else os.path.join("perf_prof", "sweep")
    rows = []
    for f in sorted(glob.glob(os.path.join(sweep, "*.log"))):
        d = parse_log(f)
        if d:
            rows.append(d)
    if not rows:
        print("no [hrxphase] blocks found in", sweep)
        return

    busy = has_busy(rows)
    note = ("device-busy timestamps PRESENT -> host_overlap / reclaimable are real."
            if busy else
            "device-busy timestamps ABSENT -> npu_busy=npu_wait, so host_overlap=0 "
            "and reclaimable=pure_host (upper bound; exact for serial prefill, "
            "over-estimate for the async-lm_head decode window).")
    print("<!-- formula note: " + note + " -->\n")

    # ---- Table 1: Prefill (per request / per iter) ----
    print("### Table 1 — Prefill (whole prefill phase, per request)\n")
    print("| model | wall (ms) | npu_busy (ms) | pure_host (ms) | reclaimable (ms) "
          "| host_overlap (ms) | % reclaimable | prompt tok |")
    print("|---|---|---|---|---|---|---|---|")
    for d in rows:
        it = d["iters"]
        p = phase_metrics(d["pf_wall"], d["pf_npu"], d["pf_npu_busy"])
        print("| {m} | {w:.2f} | {b:.2f} | {ph:.2f} | {r:.2f} | {o:.2f} | {pr:.1f}% | {tok} |".format(
            m=label(d["tag"]), w=p["wall"]/it, b=p["npu_busy"]/it, ph=p["pure_host"]/it,
            r=p["reclaimable"]/it, o=p["host_overlap"]/it,
            pr=pct(p["reclaimable"], p["wall"]), tok=d["prompt_tokens"]//it))

    # ---- Table 2: Per-token decode ----
    print("\n### Table 2 — Per-token decode\n")
    print("| model | wall/tok (ms) | npu_busy/tok (ms) | pure_host/tok (ms) "
          "| reclaimable/tok (ms) | host_overlap/tok (ms) | % reclaimable | tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for d in rows:
        gtok = d["gen_tokens"]
        p = phase_metrics(d["de_wall"], d["de_npu"], d["de_npu_busy"])
        w = p["wall"]/gtok
        print("| {m} | {w:.3f} | {b:.3f} | {ph:.3f} | {r:.3f} | {o:.3f} | {pr:.1f}% | {tps:.1f} |".format(
            m=label(d["tag"]), w=w, b=p["npu_busy"]/gtok, ph=p["pure_host"]/gtok,
            r=p["reclaimable"]/gtok, o=p["host_overlap"]/gtok,
            pr=pct(p["reclaimable"], p["wall"]), tps=1000.0/w if w else 0.0))

    # ---- Table 3: All-token decode (per request) ----
    print("\n### Table 3 — All-token decode (per request)\n")
    print("| model | wall (ms) | npu_busy (ms) | pure_host (ms) | reclaimable (ms) "
          "| host_overlap (ms) | % reclaimable | tokens/iter |")
    print("|---|---|---|---|---|---|---|---|")
    for d in rows:
        it = d["iters"]
        p = phase_metrics(d["de_wall"], d["de_npu"], d["de_npu_busy"])
        print("| {m} | {w:.2f} | {b:.2f} | {ph:.2f} | {r:.2f} | {o:.2f} | {pr:.1f}% | {tok} |".format(
            m=label(d["tag"]), w=p["wall"]/it, b=p["npu_busy"]/it, ph=p["pure_host"]/it,
            r=p["reclaimable"]/it, o=p["host_overlap"]/it,
            pr=pct(p["reclaimable"], p["wall"]), tok=d["gen_tokens"]//it))

    # ---- Table 4: Whole run (prefill + all decode, per request) ----
    print("\n### Table 4 — Whole-run (prefill + decode, per request)\n")
    print("| model | wall (ms) | npu_busy (ms) | pure_host (ms) | reclaimable (ms) "
          "| host_overlap (ms) | % reclaimable |")
    print("|---|---|---|---|---|---|---|")
    for d in rows:
        it = d["iters"]
        wall = d["pf_wall"] + d["de_wall"]
        npu_wait = d["pf_npu"] + d["de_npu"] + (d["other_npu"] or 0)
        npu_busy = None
        if d["pf_npu_busy"] is not None or d["de_npu_busy"] is not None:
            npu_busy = ((d["pf_npu_busy"] or d["pf_npu"])
                        + (d["de_npu_busy"] or d["de_npu"])
                        + (d["other_npu"] or 0))
        p = phase_metrics(wall, npu_wait, npu_busy)
        print("| {m} | {w:.2f} | {b:.2f} | {ph:.2f} | {r:.2f} | {o:.2f} | {pr:.1f}% |".format(
            m=label(d["tag"]), w=p["wall"]/it, b=p["npu_busy"]/it, ph=p["pure_host"]/it,
            r=p["reclaimable"]/it, o=p["host_overlap"]/it,
            pr=pct(p["reclaimable"], p["wall"])))

    # dump raw json for the report author
    print("\n<!-- raw: " + json.dumps(rows) + " -->")


if __name__ == "__main__":
    main()
