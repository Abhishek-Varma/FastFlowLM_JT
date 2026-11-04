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
            "device-busy timestamps ABSENT -> only pure_host = wall - total_npu is "
            "derivable. reclaimable/host_overlap need true device-busy time (see "
            "REGENERATE.md s4); until then reclaimable==pure_host, host_overlap==0, "
            "so those columns are omitted.")
    print("<!-- formula note: " + note + " -->\n")

    def emit(title, extra_hdr, make_extra, wall_of, wait_of, busy_of, unit):
        """Print one table. Columns depend on whether device-busy data exists.

        unit(value, d) scales a ms total to the row's unit (per iter / per token).
        """
        print("### " + title + "\n")
        xh = (" " + extra_hdr + " |") if extra_hdr else ""
        xs = "---|" if extra_hdr else ""
        if busy:
            print("| model | wall (ms) | npu_busy (ms) | pure_host (ms) | "
                  "reclaimable (ms) | host_overlap (ms) | % reclaimable |" + xh)
            print("|---|---|---|---|---|---|---|" + xs)
        else:
            print("| model | wall (ms) | total_npu (ms) | pure_host (ms) | "
                  "% host |" + xh)
            print("|---|---|---|---|---|" + xs)
        for d in rows:
            p = phase_metrics(wall_of(d), wait_of(d), busy_of(d))
            u = lambda v: unit(v, d)
            xv = (" {} |".format(make_extra(d, p))) if extra_hdr else ""
            if busy:
                print("| {m} | {w:.3f} | {b:.3f} | {ph:.3f} | {r:.3f} | {o:.3f} | {pr:.1f}% |{x}".format(
                    m=label(d["tag"]), w=u(p["wall"]), b=u(p["npu_busy"]), ph=u(p["pure_host"]),
                    r=u(p["reclaimable"]), o=u(p["host_overlap"]),
                    pr=pct(p["reclaimable"], p["wall"]), x=xv))
            else:
                print("| {m} | {w:.3f} | {n:.3f} | {ph:.3f} | {pr:.1f}% |{x}".format(
                    m=label(d["tag"]), w=u(p["wall"]), n=u(p["npu_wait"]), ph=u(p["pure_host"]),
                    pr=pct(p["pure_host"], p["wall"]), x=xv))

    per_iter = lambda v, d: v / d["iters"]
    per_gtok = lambda v, d: v / d["gen_tokens"]

    # ---- Table 1: Prefill (per request) ----
    emit("Table 1 — Prefill (whole prefill phase, per request)",
         "prompt tok", lambda d, p: d["prompt_tokens"] // d["iters"],
         lambda d: d["pf_wall"], lambda d: d["pf_npu"], lambda d: d["pf_npu_busy"],
         per_iter)

    # ---- Table 2: Per-token decode ----
    print()
    emit("Table 2 — Per-token decode",
         "tok/s", lambda d, p: "{:.1f}".format(1000.0 / (p["wall"] / d["gen_tokens"])),
         lambda d: d["de_wall"], lambda d: d["de_npu"], lambda d: d["de_npu_busy"],
         per_gtok)

    # ---- Table 3: All-token decode (per request) ----
    print()
    emit("Table 3 — All-token decode (per request)",
         "tokens/iter", lambda d, p: d["gen_tokens"] // d["iters"],
         lambda d: d["de_wall"], lambda d: d["de_npu"], lambda d: d["de_npu_busy"],
         per_iter)

    # ---- Table 4: Whole run (prefill + all decode, per request) ----
    def wr_busy(d):
        if d["pf_npu_busy"] is None and d["de_npu_busy"] is None:
            return None
        return ((d["pf_npu_busy"] or d["pf_npu"]) + (d["de_npu_busy"] or d["de_npu"])
                + (d["other_npu"] or 0))
    print()
    emit("Table 4 — Whole-run (prefill + decode, per request)",
         "", lambda d, p: "",
         lambda d: d["pf_wall"] + d["de_wall"],
         lambda d: d["pf_npu"] + d["de_npu"] + (d["other_npu"] or 0),
         wr_busy, per_iter)

    # dump raw json for the report author
    print("\n<!-- raw: " + json.dumps(rows) + " -->")


if __name__ == "__main__":
    main()
