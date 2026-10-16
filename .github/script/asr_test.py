#!/usr/bin/env python3
"""Check Whisper transcriptions through /v1/audio/transcriptions.

sweep_audio.py only asks for a non-empty transcript. This checks that the
transcript is right, and that every response format agrees with the others:

    json          close to the reference transcript, by word error rate
    default       no response_format at all: the same {model, text} as json
    verbose_json  the same text as json, the clip's duration and language, and
                  segments that are ordered, non-empty, inside the clip and
                  join back into the text

The bundled clip runs about 110 s, so its transcript spans four 30 s Whisper
windows and every run exercises the chunk-boundary handling.

A clip passed with --audio gets the format checks only, unless a
<clip>.reference.json sits next to it.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from sweep_common import SweepTask, run_task

REFERENCE_SUFFIX = ".reference.json"
# Whisper reports times to 0.01 s and the clip length to the sample.
DURATION_TOLERANCE_S = 0.05


def words(text: str) -> list[str]:
    """Lowercased words without punctuation, which Whisper varies run to run."""
    return re.findall(r"[\w']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the reference length."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1] / len(ref)


class AsrTest(SweepTask):
    name = "asr"
    needs_asr = True
    csv_header = ["Model", "Format", "Check", "Status", "Detail"]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--audio",
            default=None,
            metavar="PATH",
            help="Clip to transcribe. Defaults to the bundled test clip.",
        )
        parser.add_argument(
            "--max-wer",
            type=float,
            default=0.15,
            help="Highest word error rate against the reference (default: 0.15).",
        )
        parser.add_argument(
            "--require-verbose-json",
            action="store_true",
            help=(
                "Fail when the server answers verbose_json without segments. "
                "By default that is recorded as skipped, for builds that do "
                "not support verbose_json yet."
            ),
        )

    def __init__(self, args):
        super().__init__(args)
        self.rows: list[list[str]] = []

    def audio_path(self) -> Path:
        if self.args.audio:
            path = Path(self.args.audio)
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            return path
        return self.asset("test_audio.mp3")

    @staticmethod
    def load_reference(path: Path) -> dict | None:
        reference = path.with_name(path.stem + REFERENCE_SUFFIX)
        if not reference.is_file():
            return None
        return json.loads(reference.read_text(encoding="utf-8"))

    def select_models(self, catalog: list[dict]) -> list[str]:
        models = [e["id"] for e in catalog if e["id"].startswith("whisper-v3")]
        if not models:
            print(
                "[asr] No Whisper models found. The /v1/models fallback hides "
                "Whisper, so pass --models explicitly if 'flm list' failed."
            )
        return models

    # -- checks ------------------------------------------------------------

    def check(self, writer, model: str, fmt: str, name: str, ok: bool, detail: str = "") -> None:
        status = "ok" if ok else "FAIL"
        row = [model, fmt, name, status, detail]
        self.rows.append(row)
        self.record(writer, row, model=model,
                    error=None if ok else f"{fmt}: {name} ({detail})")
        print(f"    {status:4}  {fmt}: {name}" + (f"  [{detail}]" if detail else ""))

    def skip(self, writer, model: str, fmt: str, name: str, detail: str) -> None:
        row = [model, fmt, name, "skip", detail]
        self.rows.append(row)
        self.record(writer, row, model=model)
        print(f"    skip  {fmt}: {name}  [{detail}]")
        # Surfaces as an annotation on the workflow run, where a skip is
        # otherwise easy to mistake for a pass.
        print(f"::warning::{model} {fmt}: {name} skipped: {detail}")

    def transcribe(self, model: str, path: Path, response_format: str | None) -> dict:
        """The raw response body, so fields the SDK does not model survive."""
        kwargs = {"model": model}
        if response_format:
            kwargs["response_format"] = response_format
        with open(path, "rb") as handle:
            raw = self.client.audio.transcriptions.with_raw_response.create(
                file=handle, **kwargs
            )
        # The server answers with an empty body when ASR is not loaded.
        return raw.http_response.json()

    def request(self, writer, model: str, path: Path, fmt: str) -> dict | None:
        started = time.monotonic()
        try:
            body = self.transcribe(model, path, None if fmt == "default" else fmt)
        except Exception as exc:
            self.check(writer, model, fmt, "request succeeds", False, str(exc))
            return None
        print(f"  {fmt}: {time.monotonic() - started:.1f}s")
        return body

    def check_json(self, writer, model: str, body: dict, reference: dict | None) -> None:
        text = body.get("text") or ""
        self.check(writer, model, "json", "text is non-empty", bool(text.strip()))
        if reference:
            wer = word_error_rate(reference["text"], text)
            self.check(writer, model, "json", "matches the reference transcript",
                       wer <= self.args.max_wer,
                       f"WER {wer:.1%}, limit {self.args.max_wer:.0%}")

    def check_default(self, writer, model: str, body: dict, text: str) -> None:
        self.check(writer, model, "default", "returns only model and text",
                   set(body) == {"model", "text"}, ", ".join(sorted(body)))
        self.check(writer, model, "default", "text matches json",
                   body.get("text") == text)

    def check_verbose(self, writer, model: str, body: dict, text: str,
                      reference: dict | None) -> None:
        fmt = "verbose_json"
        segments = body.get("segments")
        if segments is None:
            detail = "response has no segments; server ignores verbose_json"
            if self.args.require_verbose_json:
                self.check(writer, model, fmt, "returns segments", False, detail)
            else:
                self.skip(writer, model, fmt, "all checks", detail)
            return

        self.check(writer, model, fmt, "text matches json", body.get("text") == text)

        duration = body.get("duration")
        if not isinstance(duration, (int, float)):
            self.check(writer, model, fmt, "duration is a number", False, repr(duration))
            return
        if reference:
            self.check(writer, model, fmt, "duration is the clip length",
                       abs(duration - reference["duration"]) <= DURATION_TOLERANCE_S,
                       f"{duration:.2f}s, clip {reference['duration']:.2f}s")
            self.check(writer, model, fmt, "language is detected",
                       body.get("language") == reference["language"],
                       str(body.get("language")))

        self.check(writer, model, fmt, "segments are present",
                   bool(segments) or not text.strip(), f"{len(segments)} segments")
        self.check(writer, model, fmt, "segment ids count up from 0",
                   [s.get("id") for s in segments] == list(range(len(segments))))
        outside = [s["id"] for s in segments
                   if not 0 <= s["start"] <= s["end"] <= duration + DURATION_TOLERANCE_S]
        self.check(writer, model, fmt, "segments lie inside the clip",
                   not outside, f"ids {outside[:5]}" if outside else "")
        out_of_order = [b["id"] for a, b in zip(segments, segments[1:])
                        if b["start"] < a["start"]]
        self.check(writer, model, fmt, "segments are in order",
                   not out_of_order, f"ids {out_of_order[:5]}" if out_of_order else "")
        empty = [s["id"] for s in segments
                 if s["end"] - s["start"] < 0.01 or not s["text"].strip()]
        self.check(writer, model, fmt, "no empty or zero-length segments",
                   not empty, f"ids {empty[:5]}" if empty else "")
        joined = "".join(s["text"] for s in segments).strip()
        self.check(writer, model, fmt, "segments join into the text", joined == text)

    # -- entry point -------------------------------------------------------

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{self.name}_{self.args.platform}_v{self.version}.md"

    def write_report(self) -> None:
        """Markdown table for the workflow's job summary."""
        failed = sum(row[3] == "FAIL" for row in self.rows)
        skipped = sum(row[3] == "skip" for row in self.rows)
        lines = [
            f"### ASR test: {self.args.platform}, flm {self.version}",
            "",
            f"{len(self.rows) - failed - skipped} passed, {failed} failed, "
            f"{skipped} skipped",
            "",
            "| Model | Format | Check | Status | Detail |",
            "|---|---|---|---|---|",
        ]
        for row in self.rows:
            cells = [str(cell).replace("|", "\\|") for cell in row]
            lines.append("| " + " | ".join(cells) + " |")
        for failure in self.failures:
            if not any(failure.startswith(row[0]) for row in self.rows):
                lines.append(f"\n- {failure}")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        path = self.audio_path()
        reference = self.load_reference(path)
        print(f"[asr] Clip: {path}" + ("" if reference else " (no reference transcript)"))

        models = self.resolve_models()
        if not models:
            self.failures.append("no Whisper model to test")
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                print(f"\n--- ASR: {model} ---")
                plain = self.request(writer, model, path, "json")
                if plain is None:
                    continue
                text = plain.get("text") or ""
                self.check_json(writer, model, plain, reference)
                default = self.request(writer, model, path, "default")
                if default is not None:
                    self.check_default(writer, model, default, text)
                verbose = self.request(writer, model, path, "verbose_json")
                if verbose is not None:
                    self.check_verbose(writer, model, verbose, text, reference)
        finally:
            handle.close()
            self.write_report()
        print(f"\nASR test complete. Saved to {self.csv_path}")
        self.write_summary(models)


if __name__ == "__main__":
    sys.exit(run_task(AsrTest))
