#!/usr/bin/env python3
"""Render ASR test results as one Markdown report, a section per model.

Reads every *_results.json that asr_test.py wrote under the given directory
(only those in an asr-results folder when the artifacts also hold baseline
runs) and prints GitHub-flavoured Markdown to stdout, for $GITHUB_STEP_SUMMARY.
Each model gets its own section with the platforms side by side, so a reader
can see at once which model failed where and why.

asr_test.py also calls render() for the report of a single run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLATFORM_ORDER = ["linux", "linux-hrx", "windows", "windows-hrx",
                  "linux-models-from-source", "windows-models-from-source"]
PLATFORM_LABELS = {
    "linux": "Linux", "linux-hrx": "Linux HRX", "windows": "Windows",
    "windows-hrx": "Windows HRX", "linux-models-from-source": "Linux (models from source)",
    "windows-models-from-source": "Windows (models from source)",
}
WHISPER_PREFIX = "whisper-v3"


def pct(value) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def points(current, base) -> str:
    if current is None or base is None:
        return ""
    return f" ({(current - base) * 100:+.2f})"


def label(platform: str) -> str:
    return PLATFORM_LABELS.get(platform, platform)


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def reference_name(result: dict) -> str:
    base = result.get("baseline") or {}
    if base.get("kind") == "build":
        return "main"
    if base.get("kind") == "expected":
        return "expected"
    return ""


def model_checks(result: dict, model: str) -> list[dict]:
    return [c for c in result.get("checks", []) if c["model"] == model]


def verdict(result: dict | None, model: str) -> str:
    if result is None:
        return "no results"
    checks = model_checks(result, model)
    if not checks:
        return "not run"
    return "**FAIL**" if any(c["status"] == "FAIL" for c in checks) else "**PASS**"


def whisper_models(results: list[dict]) -> list[str]:
    models = []
    for result in results:
        for check in result.get("checks", []):
            if check["model"].startswith(WHISPER_PREFIX) and check["model"] not in models:
                models.append(check["model"])
    return models or [f"{WHISPER_PREFIX}:turbo"]


def chat_models(results: list[dict]) -> list[str]:
    models = []
    for result in results:
        for model in result.get("chat", {}):
            if model not in models:
                models.append(model)
    return models


def problems(lines: list[str], results: list[dict], model: str) -> None:
    """Failed checks and warnings for one model, grouped by platform."""
    lines += ["**Failed checks and warnings**", ""]
    any_listed = False
    for result in results:
        rows = [c for c in model_checks(result, model) if c["status"] != "ok"]
        if not rows:
            continue
        any_listed = True
        lines.append(f"*{label(result['platform'])}*")
        lines.append("")
        for row in rows:
            status = "FAIL" if row["status"] == "FAIL" else "warning"
            group = "" if row["group"] == "core" else f" [{row['group']}]"
            detail = f": {cell(row['detail'])}" if row["detail"] else ""
            lines.append(f"- **{status}**{group} {cell(row['area'])}: "
                         f"{cell(row['check'])}{detail}")
        lines.append("")
    if not any_listed:
        lines += ["_None._", ""]


def whisper_section(lines: list[str], results: list[dict], model: str) -> None:
    columns = [label(r["platform"]) for r in results]
    ref = next((reference_name(r) for r in results if reference_name(r)), "")
    lines += [f"## {model}: speech-to-text (`/v1/audio/transcriptions`)", "",
              "**Verdict:** " + " · ".join(f"{label(r['platform'])} {verdict(r, model)}"
                                          for r in results), ""]

    lines += ["**Accuracy**", "",
              "Error rate on each FLEURS set: WER, or CER for ja and zh. "
              + (f"In brackets, the change in points against {ref}. " if ref else "")
              + "Lower is better.", "",
              "| Set | Metric | " + " | ".join(columns) + " |",
              "|---|---|" + "---|" * len(columns)]
    set_names: list[str] = []
    for result in results:
        set_names += [s for s in result.get("sets", {}) if s not in set_names]
    for set_name in set_names:
        metric = next((r["sets"][set_name]["metric"].upper() for r in results
                       if set_name in r.get("sets", {})), "")
        cells = []
        for result in results:
            current = result.get("sets", {}).get(set_name)
            base = ((result.get("baseline") or {}).get("sets", {}).get(set_name)
                    or {}).get("error_rate")
            cells.append("-" if current is None else
                         pct(current["error_rate"]) + points(current["error_rate"], base))
        lines.append(f"| {set_name} | {metric} | " + " | ".join(cells) + " |")
    cells = []
    for result in results:
        current = (result.get("overall") or {}).get("error_rate")
        base = ((result.get("baseline") or {}).get("overall") or {}).get("error_rate")
        cells.append(f"**{pct(current)}**{points(current, base)}")
    lines.append("| **pooled** | all | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["**Behaviour**", "",
              "| Measure | " + " | ".join(columns) + " |",
              "|---|" + "---|" * len(columns)]

    def row(name: str, render) -> None:
        lines.append(f"| {name} | " + " | ".join(render(r) for r in results) + " |")

    def runaways(result: dict) -> str:
        current = (result.get("overall") or {}).get("runaways")
        base = ((result.get("baseline") or {}).get("overall") or {}).get("runaways")
        if current is None:
            return "-"
        return f"{current}" + (f" ({ref} {base})" if base is not None else "")

    def changed(result: dict) -> str:
        sets = result.get("sets", {}).values()
        if not any("changed" in s for s in sets):
            return "-"
        return f"{sum(s.get('changed', 0) for s in sets)}/{sum(s['files'] for s in sets)}"

    def speed(result: dict) -> str:
        sets = result.get("sets", {})
        audio = sum(s.get("audio_s", 0) for s in sets.values())
        wall = sum(s.get("wall_s", 0) for s in sets.values())
        base_sets = (result.get("baseline") or {}).get("sets", {})
        base_audio = sum(s.get("audio_s", 0) for s in base_sets.values())
        base_wall = sum(s.get("wall_s", 0) for s in base_sets.values())
        text = f"{wall / audio:.3f}" if audio else "-"
        if base_audio and ref == "main":
            text += f" (main {base_wall / base_audio:.3f})"
        return text

    row("Runaway transcripts (repetition loops)", runaways)
    row("Files whose transcript changed vs main", changed)
    row("Processing seconds per audio second", speed)
    lines.append("")

    groups: list[str] = []
    for result in results:
        groups += [g for g in result.get("features", {}) if g not in groups]
    lines += ["**Checks by group** (passed/total; required groups fail the run)", "",
              "| Group | Required | " + " | ".join(columns) + " |",
              "|---|---|" + "---|" * len(columns)]
    for group in groups:
        required = "yes" if group == "core" or group in results[0].get("required", []) else "no"
        cells = []
        for result in results:
            checks = [c for c in model_checks(result, model) if c["group"] == group]
            if not checks:
                cells.append("-")
                continue
            passed = sum(c["status"] == "ok" for c in checks)
            cells.append(f"{passed}/{len(checks)}")
        lines.append(f"| {group} | {required} | " + " | ".join(cells) + " |")
    lines.append("")
    problems(lines, results, model)


def chat_section(lines: list[str], results: list[dict], model: str) -> None:
    columns = [label(r["platform"]) for r in results]
    lines += [f"## {model}: chat transcription (audio sent to the chat model)", "",
              "**Verdict:** " + " · ".join(f"{label(r['platform'])} {verdict(r, model)}"
                                          for r in results), "",
              "| Measure | " + " | ".join(columns) + " |",
              "|---|" + "---|" * len(columns)]
    cells, clips = [], []
    for result in results:
        current = result.get("chat", {}).get(model)
        base = ((result.get("baseline") or {}).get("chat") or {}).get(model)
        if current is None:
            cells.append("-")
            clips.append("-")
            continue
        text = pct(current["error_rate"])
        if base:
            text += f" (main {pct(base['error_rate'])})"
        cells.append(text)
        clips.append(str(current["files"]))
    lines.append("| WER on English clips | " + " | ".join(cells) + " |")
    lines.append("| Clips | " + " | ".join(clips) + " |")
    lines.append("")
    problems(lines, results, model)


def render(results: list[dict], title: str = "ASR Test") -> str:
    results = sorted(results, key=lambda r: (PLATFORM_ORDER.index(r["platform"])
                                             if r["platform"] in PLATFORM_ORDER else 99,
                                             r["platform"]))
    lines = [f"# {title}", ""]
    if not results:
        return "\n".join(lines + ["_No results were produced._", ""])

    whisper = whisper_models(results)
    chats = chat_models(results)
    refs = {reference_name(r) for r in results} - {""}
    required = results[0].get("required", [])
    lines += [
        "Builds under test: flm " + ", ".join(sorted({str(r.get('version')) for r in results}))
        + ". Compared against: " + (" and ".join(
            "main's latest build, run first on the same machine" if r == "main"
            else "the expected rates in asr_eval/expected.json" for r in sorted(refs))
            or "nothing") + ".",
        "",
        "Required feature groups: " + (", ".join(required) or "none")
        + ". Other groups are measured and reported as warnings.",
        "",
        "| Model | " + " | ".join(label(r["platform"]) for r in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for model in whisper + chats:
        lines.append(f"| {model} | " + " | ".join(verdict(r, model) for r in results) + " |")
    lines.append("")

    for model in whisper:
        whisper_section(lines, results, model)
    for model in chats:
        chat_section(lines, results, model)

    other = [c for r in results for c in r.get("checks", [])
             if c["status"] == "FAIL" and c["model"] not in whisper + chats]
    if other:
        lines += ["## Other failures", ""]
        lines += [f"- {cell(c['model'])}: {cell(c['check'])} {cell(c['detail'])}" for c in other]
        lines.append("")
    return "\n".join(lines)


def load_results(root: Path) -> list[dict]:
    paths = sorted(root.rglob("*_results.json"))
    # Artifacts also carry the baseline's own run; report the builds under test.
    if any("asr-results" in p.parts for p in paths):
        paths = [p for p in paths if "asr-results" in p.parts]
    results = []
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"<!-- skipped {path}: {exc} -->")
            continue
        if "platform" in result and "checks" in result:
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", help="Directory holding the downloaded artifacts.")
    args = parser.parse_args(argv)
    print(render(load_results(Path(args.directory))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
