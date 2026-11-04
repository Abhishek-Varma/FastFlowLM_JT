#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-platform phase sweep driver (Linux + Windows).

Runs every text model through ``flm bench <tag> -i <cfg>`` with
``FLM_PUREHRX_PROFILE=1`` so each run prints an ``[hrxphase]`` block, and
captures one log per model under the output dir. Serial (one model at a time)
and wedge-guarded (retry up to 3x with a cooldown) so the NPU is never wedged.

Configuration is read from environment variables so any agent can reproduce the
sweep on their own box without editing the script. All paths default to the
reference layout documented in REGENERATE.md; override as needed.

  FLM_BIN            path to the flm binary        (default: ./src/build/flm[.exe])
  FLM_BENCH_CFG      bench config json             (default: ./perf_prof/bench_1k_20.json)
  FLM_SWEEP_OUT      output dir for *.log          (default: ./perf_prof/sweep_shim)
  FLM_SWEEP_TAGS     space/newline/comma separated list of model tags to run
                     (default: the built-in TEXT_TAGS list below)
  FLM_HRX_LIBDIR     dir with the HRX runtime .so/.dll (prepended to the loader path)
  FLM_ENGINE_LIBDIR  dir with the deployed engine libs (prepended to the loader path)
  FLM_CONFIG_PATH / FLM_MODELINFO_PATH / FLM_XCLBIN_PATH / FLM_MODEL_PATH
                     passed straight through to flm if set.

Usage:  python phase_sweep.py            # full sweep
        python phase_sweep.py qwen3:0.6b lfm2:1.2b   # just these tags
"""
import os
import sys
import time
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # FastFlowLM_JT

TEXT_TAGS = """
gemma3:1b gemma3:4b gemma4-it:12b gemma4-it:e2b gemma4-it:e4b gemma4e-flash:e4b
gpt-oss:20b hy-mt2-flash:1.8b lfm2:1.2b lfm2:2.6b llama3.2:1b llama3.2:3b
llama3.1:8b nanbeige4.1:3b phi4-mini-it:4b qwen2.5-it:3b qwen2.5vl-it:3b
qwen3:0.6b qwen3:1.7b qwen3:4b qwen3:8b qwen3vl-it:4b qwen3vl-flash:4b
qwen3.5:0.8b qwen3.5:2b qwen3.5:4b qwen3.5:9b qwen3.6-moe:35b-a3b
""".split()

WEDGE_MARKERS = ("ert state", "hrx][ERROR]", "did not complete")


def default_flm_bin():
    exe = "flm.exe" if os.name == "nt" else "flm"
    cand = REPO / "src" / "build" / exe
    return str(cand)


def tags_from_env():
    raw = os.environ.get("FLM_SWEEP_TAGS")
    if not raw:
        return TEXT_TAGS
    return [t for t in raw.replace(",", " ").split() if t]


def is_wedged(log_text):
    if any(mark in log_text for mark in WEDGE_MARKERS):
        return True
    return "hrxphase" not in log_text  # incomplete run


def build_env(hrx_libdir, engine_libdir):
    env = dict(os.environ)
    env["FLM_PUREHRX_PROFILE"] = "1"
    # Prepend HRX runtime + engine dirs to the platform loader path.
    extra = [p for p in (hrx_libdir, engine_libdir) if p]
    if extra:
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        else:
            key = "LD_LIBRARY_PATH"
            env[key] = os.pathsep.join(extra + [env.get(key, "")])
    return env


def main():
    flm = os.environ.get("FLM_BIN", default_flm_bin())
    cfg = os.environ.get("FLM_BENCH_CFG", str(HERE / "bench_1k_20.json"))
    out = Path(os.environ.get("FLM_SWEEP_OUT", str(HERE / "sweep_shim")))
    hrx_libdir = os.environ.get("FLM_HRX_LIBDIR", "")
    engine_libdir = os.environ.get(
        "FLM_ENGINE_LIBDIR", str(REPO / "src" / "lib" / "hrx"))
    tags = sys.argv[1:] or tags_from_env()

    out.mkdir(parents=True, exist_ok=True)
    env = build_env(hrx_libdir, engine_libdir)

    if not Path(flm).exists():
        print("ERROR: flm binary not found at", flm,
              "- set FLM_BIN", file=sys.stderr)
        return 2
    if not Path(cfg).exists():
        print("ERROR: bench config not found at", cfg,
              "- set FLM_BENCH_CFG", file=sys.stderr)
        return 2

    print("flm       :", flm)
    print("bench cfg :", cfg)
    print("out dir   :", out)
    print("tags      :", len(tags), "models")
    failures = []
    for tag in tags:
        safe = tag.replace(":", "_").replace("/", "_")
        log = out / (safe + ".log")
        ok = False
        for attempt in (1, 2, 3):
            print("[{t}] {tag} attempt {a} ...".format(
                t=time.strftime("%H:%M:%S"), tag=tag, a=attempt))
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                subprocess.run([flm, "bench", tag, "-i", cfg],
                               stdout=fh, stderr=subprocess.STDOUT, env=env)
            text = open(log, "r", encoding="utf-8", errors="replace").read()
            if is_wedged(text):
                print("  WEDGE/incomplete -> cooldown 20s")
                time.sleep(20)
                continue
            print("  clean")
            ok = True
            break
        if not ok:
            failures.append(tag)
            print("  !! {} FAILED after 3 attempts".format(tag))
        time.sleep(3)

    print("=== SWEEP DONE ->", out, "===")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
