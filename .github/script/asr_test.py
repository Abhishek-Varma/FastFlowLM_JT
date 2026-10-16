#!/usr/bin/env python3
"""Measure and check ASR through /v1/audio/transcriptions and chat audio.

sweep_audio.py only asks for a non-empty transcript. This runs the FLEURS
evaluation set in assets/asr_eval/ (see build_asr_eval.py) and measures:

    accuracy      WER per set, CER for ja and zh, against human transcripts,
                  each file's errors capped at its reference length so one
                  runaway transcript cannot swamp a set; and a pooled rate
                  over every set
    runaways      transcripts over 1.5 times the reference length, which is
                  what a decoder stuck in a repetition loop produces
    speed         seconds of processing per second of audio
    formats       json, no response_format and verbose_json agree
    chat audio    WER of chat models that transcribe audio: Gemma 4 E2B, whose
                  audio front end E4B and the flash variants share, and 12B,
                  which has its own

and checks these feature groups:

    verbose_json     segments that are ordered, inside the clip, non-empty and
                     join into the text; duration equal to the clip length
    timestamp_rules  segments never overlap; the first one starts within 1 s
    window_tail      no text repeated across 30 s window boundaries
    language         the language field is honoured; unknown codes fail
    prompt           a prompt is accepted and helps with the words it names
    quiet_log        transcripts stay out of the server log

A failed check in a group fails the run only if the group is listed in
--require; otherwise it is reported as "warn". Core checks always count. The
groups exist so a feature can be required once main has it, and measured
before then.

With --baseline, the results of another build (normally main's, run on the
same machine just before) become the reference. The run fails if the pooled
error rate rises by more than --tolerance points, the rate of any set of ten
or more files by more than --set-tolerance, or a file turns into a runaway, or
repeats more phrases, while also getting more errors than in the baseline.
The long-form sets are left to the pooled rate and those last two checks: a
single file there moves by many points whenever one decode flips, which HRX
builds do between identical runs. The report also shows how
many transcripts changed. Even a numerically equivalent change, such as a
different FFT, flips a few borderline decodes, so a few changed files and
small per-set moves are expected; --max-drift is for builds that should
reproduce the baseline exactly. Without a baseline, --expected gives fixed
error rates to stay under, with the same --tolerance on the pooled rate and
--set-tolerance per set; the workflow passes --tolerance 2 there, since those
rates come from one machine per platform.

Every response body, and a results JSON that --baseline can read back, are
written to --output-dir.
"""

from __future__ import annotations

import base64
import json
import re
import socket
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from asr_summary import label as platform_label, render
from sweep_common import ASSETS_DIR, SweepTask, run_task

FEATURE_GROUPS = [
    "verbose_json", "timestamp_rules", "window_tail", "language", "prompt", "quiet_log",
]
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "zh": "Chinese", "ar": "Arabic",
}
CHAT_PROMPT = (
    "Transcribe the speech in this audio exactly, in its original language. "
    "Reply with the transcript only."
)
# MP3 encoder padding and the server's own rounding.
DURATION_TOLERANCE_S = 0.1
# The encoder is not bit-exact between passes, so two requests for the same
# clip can differ in a word or two (see the PR #794 notes).
SAME_TEXT_MAX_WER = 0.02
REPEAT_NGRAM = 6
RUNAWAY_RATIO = 1.5
# Smaller sets are judged through the pooled rate only; see the docstring.
SET_CHECK_MIN_FILES = 10
ARABIC_DIACRITICS = re.compile("[\u064b-\u065f\u0670\u06d6-\u06ed]")


# -- scoring ------------------------------------------------------------------

def normalize(text: str, language: str) -> str:
    """Lowercase, no punctuation or symbols, single spaces, NFKC."""
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "ar":
        text = ARABIC_DIACRITICS.sub("", text)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def tokens(text: str, language: str, metric: str) -> list[str]:
    text = normalize(text, language)
    return list(text.replace(" ", "")) if metric == "cer" else text.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        current = [i]
        for j, hyp_token in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_token != hyp_token)))
        previous = current
    return previous[-1]


def error_rate(ref: str, hyp: str, language: str, metric: str) -> tuple[int, int]:
    """(edits, reference length), so rates can be pooled over a set."""
    ref_tokens = tokens(ref, language, metric)
    return edit_distance(ref_tokens, tokens(hyp, language, metric)), len(ref_tokens)


def pooled(pairs) -> float:
    """Error rate over many files, each file's edits capped at its length."""
    edits = sum(min(e, n) for e, n in pairs)
    length = sum(n for _, n in pairs)
    return edits / length if length else 0.0


def repeats(text: str, n: int = REPEAT_NGRAM) -> int:
    """n-word sequences that occur more than once."""
    words = normalize(text, "en").split()
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return sum(c - 1 for c in counts.values() if c > 1)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


# -- test ---------------------------------------------------------------------

class AsrTest(SweepTask):
    name = "asr"
    needs_asr = True
    csv_header = ["Model", "Area", "Group", "Check", "Status", "Detail"]

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--eval-dir", default=str(ASSETS_DIR / "asr_eval"),
                            help="Directory holding manifest.json and the clips.")
        parser.add_argument("--sets", nargs="+", default=None, metavar="SET",
                            help="Run only these sets (default: all).")
        parser.add_argument("--require", nargs="*", default=[], metavar="GROUP",
                            choices=FEATURE_GROUPS,
                            help="Feature groups whose failures fail the run.")
        parser.add_argument("--baseline", default=None, metavar="RESULTS_JSON",
                            help="Results of a reference build to compare against.")
        parser.add_argument("--expected", default=None, metavar="JSON",
                            help="Error rates per platform and set to stay under when there "
                                 "is no baseline. Builds decode differently per platform, "
                                 "so only --platform's entry is used.")
        parser.add_argument("--tolerance", type=float, default=1.0,
                            help="Allowed rise in the pooled error rate, in points (default: 1.0).")
        parser.add_argument("--set-tolerance", type=float, default=5.0,
                            help="Allowed rise in the error rate of any set of "
                                 f"{SET_CHECK_MIN_FILES} or more files, in points "
                                 "(default: 5.0). One flipped decode moves a ten-file "
                                 "set by a point or more.")
        parser.add_argument("--max-drift", type=float, default=None,
                            help="Fail a set whose transcripts differ from the baseline's "
                                 "by more than this fraction, for builds meant to change "
                                 "nothing. Unset: drift is only reported.")
        parser.add_argument("--chat-models", nargs="*",
                            default=["gemma4-it:e2b", "gemma4-it:12b"], metavar="TAG",
                            help="Chat models to transcribe with (default: gemma4-it:e2b "
                                 "gemma4-it:12b). Pass none to skip.")
        parser.add_argument("--chat-clips", type=int, default=5,
                            help="en-short clips sent to each chat model (default: 5).")
        parser.add_argument("--chat-tolerance", type=float, default=3.0,
                            help="Allowed rise in chat WER against the baseline, in points.")

    def __init__(self, args):
        super().__init__(args)
        self.rows: list[list[str]] = []
        self.check_failures: set[str] = set()
        self.results: dict = {"sets": {}, "items": {}, "chat": {}, "features": {},
                              "repeats": {}, "language": {}}
        self.baseline = (json.loads(Path(args.baseline).read_text(encoding="utf-8"))
                         if args.baseline else None)
        self.expected = None
        if args.expected:
            platforms = json.loads(Path(args.expected).read_text(encoding="utf-8"))["platforms"]
            self.expected = platforms.get(args.platform)
            if self.expected is None:
                print(f"::warning::{args.expected} has no expected rates for "
                      f"{args.platform}; accuracy is reported but not checked.")
        self.responses_dir = self.output_dir / "responses"
        self.responses_dir.mkdir(exist_ok=True)

    def start_server(self, append_log: bool = False) -> None:
        """Refuses to start when another server already holds the port.

        flm exits at once when it cannot bind, but the other server answers
        /v1/version, so the run would carry on against a server without ASR,
        or one that is running someone else's job.
        """
        if not self.args.no_server:
            try:
                socket.create_connection((self.args.host, self.args.port), timeout=2).close()
            except OSError:
                pass
            else:
                raise RuntimeError(f"Port {self.args.port} on {self.args.host} is already in "
                                   "use by another server; pass --port to use a free one.")
        super().start_server(append_log)

    def select_models(self, catalog: list[dict]) -> list[str]:
        models = [e["id"] for e in catalog if e["id"].startswith("whisper-v3")]
        if not models:
            print("[asr] No Whisper models found. The /v1/models fallback hides "
                  "Whisper, so pass --models explicitly if 'flm list' failed.")
        return models

    # -- recording ---------------------------------------------------------

    def check(self, writer, model: str, area: str, name: str, ok: bool,
              detail: str = "", group: str = "core") -> None:
        required = group == "core" or group in self.args.require
        status = "ok" if ok else ("FAIL" if required else "warn")
        row = [model, area, group, name, status, detail]
        self.rows.append(row)
        error = None if ok or not required else f"{area}: {name} ({detail})"
        if error:
            self.check_failures.add(f"{model}: {error}")
        self.record(writer, row, model=model, error=error)
        feature = self.results["features"].setdefault(group, {"passed": 0, "total": 0})
        feature["total"] += 1
        feature["passed"] += ok
        if not ok or group == "core":
            print(f"    {status:4}  [{group}] {area}: {name}" + (f"  ({detail})" if detail else ""))

    def save_body(self, model: str, label: str, text: str) -> None:
        safe = re.sub(r"[^\w.-]", "_", f"{model}_{label}")
        (self.responses_dir / f"{safe}.json").write_text(text, encoding="utf-8")

    # -- requests ----------------------------------------------------------

    def transcribe(self, model: str, path: Path, label: str, **fields) -> tuple[int, dict | None, float]:
        """(HTTP status, parsed body or None, seconds). The body is saved first."""
        kwargs = {key: value for key, value in fields.items() if value is not None}
        started = time.monotonic()
        with open(path, "rb") as handle:
            raw = self.client.audio.transcriptions.with_raw_response.create(
                model=model, file=handle, **kwargs)
        seconds = time.monotonic() - started
        response = raw.http_response
        self.save_body(model, label, response.text)
        try:
            body = json.loads(response.text)
        except json.JSONDecodeError:
            body = None
        return response.status_code, body, seconds

    def request(self, writer, model: str, path: Path, label: str, area: str,
                **fields) -> tuple[dict | None, float]:
        """A request that is expected to succeed; failures become core checks."""
        try:
            status, body, seconds = self.transcribe(model, path, label, **fields)
        except Exception as exc:
            self.check(writer, model, area, f"{label}: request succeeds", False, str(exc))
            return None, 0.0
        if status >= 400 or body is None or "error" in body:
            # The server answers with an empty body when ASR is not loaded.
            self.check(writer, model, area, f"{label}: request succeeds", False,
                       f"HTTP {status}: {json.dumps(body)[:120] if body else 'not JSON'}")
            return None, seconds
        return body, seconds

    # -- formats -----------------------------------------------------------

    def same_text(self, a: str, b: str) -> tuple[bool, str]:
        edits, length = error_rate(a, b, "en", "wer")
        rate = edits / length if length else float(bool(b.strip()))
        return rate <= SAME_TEXT_MAX_WER, f"{rate:.1%} apart" if rate else "identical"

    def check_segments(self, writer, model: str, area: str, body: dict, duration: float,
                       failures: Counter) -> None:
        """verbose_json and timestamp_rules checks, tallied into `failures`."""
        segments = body.get("segments") or []
        text = body.get("text") or ""
        problems = {
            "duration is the clip length":
                not isinstance(body.get("duration"), (int, float))
                or abs(body["duration"] - duration) > DURATION_TOLERANCE_S,
            "segment ids count up from 0":
                [s.get("id") for s in segments] != list(range(len(segments))),
            "segments lie inside the clip":
                any(not 0 <= s["start"] <= s["end"] <= duration + DURATION_TOLERANCE_S
                    for s in segments),
            "segments are in order":
                any(b["start"] < a["start"] for a, b in zip(segments, segments[1:])),
            "no empty or zero-length segments":
                any(s["end"] - s["start"] < 0.01 or not s["text"].strip() for s in segments),
            "segments join into the text":
                "".join(s["text"] for s in segments).strip() != text.strip(),
        }
        for name, bad in problems.items():
            failures[("verbose_json", name)] += bad
        rules = {
            "segments never overlap":
                any(b["start"] < a["end"] - 0.005 for a, b in zip(segments, segments[1:])),
            "first segment starts within 1 s":
                bool(segments) and segments[0]["start"] > 1.0,
        }
        for name, bad in rules.items():
            failures[("timestamp_rules", name)] += bad

    def run_formats(self, writer, model: str, item: dict, path: Path) -> None:
        """json, default and verbose_json on one clip must agree."""
        area = f"formats ({item['file']})"
        plain, _ = self.request(writer, model, path, "formats_json", area, response_format="json")
        default, _ = self.request(writer, model, path, "formats_default", area)
        verbose, _ = self.request(writer, model, path, "formats_verbose_json", area,
                                  response_format="verbose_json")
        if plain is None:
            return
        text = plain.get("text") or ""
        self.check(writer, model, area, "json text is non-empty", bool(text.strip()))
        if default is not None:
            self.check(writer, model, area, "default returns only model and text",
                       set(default) == {"model", "text"}, ", ".join(sorted(default)))
            ok, detail = self.same_text(text, default.get("text") or "")
            self.check(writer, model, area, "default text matches json", ok, detail)
        if verbose is not None:
            has_segments = "segments" in verbose
            self.check(writer, model, area, "verbose_json returns segments", has_segments,
                       "" if has_segments else "server ignores verbose_json",
                       group="verbose_json")
            if has_segments:
                ok, detail = self.same_text(text, verbose.get("text") or "")
                self.check(writer, model, area, "verbose_json text matches json", ok, detail,
                           group="verbose_json")

    # -- evaluation --------------------------------------------------------

    def run_eval(self, writer, model: str, items: list[dict], eval_dir: Path) -> None:
        """One verbose_json request per clip: accuracy, speed and segments."""
        segment_failures: Counter = Counter()
        verbose_items = 0
        by_set: dict[str, list[dict]] = {}
        for item in items:
            by_set.setdefault(item["set"], []).append(item)

        for set_name, set_items in by_set.items():
            print(f"  set {set_name}: {len(set_items)} files")
            pairs, audio_s, wall_s, detected, runaways = [], 0.0, 0.0, 0, 0
            for item in set_items:
                body, seconds = self.request(writer, model, eval_dir / item["file"],
                                             f"eval_{item['file']}", f"eval {set_name}",
                                             response_format="verbose_json")
                if body is None:
                    continue
                text = body.get("text") or ""
                edits, length = error_rate(item["reference"], text, item["language"],
                                           item["metric"])
                pairs.append((edits, length))
                runaway = len(tokens(text, item["language"], item["metric"])) > RUNAWAY_RATIO * length
                runaways += runaway
                audio_s += item["duration"]
                wall_s += seconds
                self.results["items"][item["file"]] = {
                    "set": set_name, "language": item["language"], "metric": item["metric"],
                    "text": text, "seconds": seconds, "edits": edits, "length": length,
                    "runaway": runaway}
                if body.get("language") == LANGUAGE_NAMES[item["language"]]:
                    detected += 1
                if "segments" in body:
                    verbose_items += 1
                    self.check_segments(writer, model, f"eval {set_name}", body,
                                        item["duration"], segment_failures)
                if item["set"] in ("en-long", "en-longform"):
                    self.results["repeats"][item["file"]] = {
                        "reference": repeats(item["reference"]), "text": repeats(text)}

            rate = pooled(pairs)
            self.results["sets"][set_name] = {
                "metric": set_items[0]["metric"], "error_rate": rate, "files": len(pairs),
                "runaways": runaways,
                "audio_s": audio_s, "wall_s": wall_s,
                "rtf": wall_s / audio_s if audio_s else None,
                "language_detected": detected if verbose_items else None,
            }
            print(f"    {set_items[0]['metric'].upper()} {pct(rate)}, {runaways} runaways, "
                  f"{audio_s:.0f} s of audio in {wall_s:.1f} s")
            self.check(writer, model, f"eval {set_name}", "every file transcribed",
                       len(pairs) == len(set_items), f"{len(pairs)}/{len(set_items)}")

        all_items = self.results["items"].values()
        self.results["overall"] = {
            "error_rate": pooled([(r["edits"], r["length"]) for r in all_items]),
            "runaways": sum(r["runaway"] for r in all_items),
            "files": len(self.results["items"]),
        }
        print(f"  pooled error rate {pct(self.results['overall']['error_rate'])}, "
              f"{self.results['overall']['runaways']} runaways")

        # Without segments, run_formats has already reported verbose_json.
        for (group, name), bad in sorted(segment_failures.items()):
            self.check(writer, model, "eval", name, bad == 0,
                       f"{bad} of {verbose_items} files" if bad else f"{verbose_items} files",
                       group=group)

        for name, counts in self.results["repeats"].items():
            self.check(writer, model, f"eval {name}",
                       f"no {REPEAT_NGRAM}-word phrase repeated beyond the reference",
                       counts["text"] <= counts["reference"],
                       f"{counts['text']} repeats, reference {counts['reference']}",
                       group="window_tail")

    # -- feature probes ----------------------------------------------------

    def run_language(self, writer, model: str, items: list[dict], eval_dir: Path) -> None:
        """Forcing the language: same or better accuracy, and actually honoured."""
        group, area = "language", "language"
        for language in sorted({i["language"] for i in items} - {"en"}):
            probe = [i for i in items if i["language"] == language][:3]
            pairs, reported = [], []
            for item in probe:
                body, _ = self.request(writer, model, eval_dir / item["file"],
                                       f"language_{item['file']}", area,
                                       response_format="verbose_json", language=language)
                if body is not None:
                    pairs.append(error_rate(item["reference"], body.get("text") or "",
                                            language, item["metric"]))
                    if "segments" in body:
                        reported.append(body.get("language"))
            auto = pooled([(self.results["items"][i["file"]]["edits"],
                            self.results["items"][i["file"]]["length"])
                           for i in probe if i["file"] in self.results["items"]])
            forced = pooled(pairs)
            self.results["language"][language] = {"auto": auto, "forced": forced}
            self.check(writer, model, area, f"{language}: forcing it does not hurt",
                       len(pairs) == len(probe)
                       and forced <= auto + self.args.set_tolerance / 100,
                       f"{pct(forced)} forced, {pct(auto)} detected", group=group)
            # Only verbose_json reports the language. Forcing a wrong language
            # is no test: Whisper often keeps transcribing what it hears.
            if reported:
                self.check(writer, model, area, f"{language}: response reports it",
                           all(r == LANGUAGE_NAMES[language] for r in reported),
                           ", ".join(sorted({str(r) for r in reported})), group=group)

        english = next((i for i in items if i["set"] == "en-short"), None)
        if english is None:
            return
        path = eval_dir / english["file"]
        try:
            status, bad_body, _ = self.transcribe(model, path, "language_invalid", language="xx")
            rejected = status >= 400 or bad_body is None or "error" in bad_body
            detail = f"HTTP {status}"
        except Exception as exc:
            rejected, detail = True, type(exc).__name__
        self.check(writer, model, area, "an unknown language code is rejected", rejected,
                   detail, group=group)

    def run_prompt(self, writer, model: str, items: list[dict], eval_dir: Path) -> None:
        """A prompt naming the clip's proper nouns should not hurt, and should help them."""
        group, area = "prompt", "prompt"
        probe = [i for i in items if i["set"] == "en-short"][:5]
        hits_without = hits_with = names_total = 0
        pairs_without, pairs_with = [], []
        for item in probe:
            names = sorted({w.strip(".,;:!?\"'()") for w in item["raw_reference"].split()[1:]
                            if w[:1].isupper()} - {""})
            if not names:
                continue
            body, _ = self.request(writer, model, eval_dir / item["file"],
                                   f"prompt_{item['file']}", area, prompt=" ".join(names))
            before = self.results["items"].get(item["file"])
            if body is None or before is None:
                continue
            after = body.get("text") or ""
            pairs_with.append(error_rate(item["reference"], after, "en", "wer"))
            pairs_without.append((before["edits"], before["length"]))
            names_total += len(names)
            hits_without += sum(n in before["text"] for n in names)
            hits_with += sum(n in after for n in names)
        if not names_total:
            return
        with_rate, without_rate = pooled(pairs_with), pooled(pairs_without)
        self.results["prompt"] = {"names": names_total, "hits_with": hits_with,
                                  "hits_without": hits_without,
                                  "wer_with": with_rate, "wer_without": without_rate}
        self.check(writer, model, area, "a prompt does not hurt accuracy",
                   with_rate <= without_rate + self.args.set_tolerance / 100,
                   f"WER {pct(with_rate)} with, {pct(without_rate)} without", group=group)
        self.check(writer, model, area, "names in the prompt are spelled as given",
                   hits_with >= hits_without,
                   f"{hits_with}/{names_total} with, {hits_without}/{names_total} without",
                   group=group)

    def run_quiet_log(self, writer, model: str) -> None:
        # Streamed output interleaves timestamp markers with the text.
        log = re.sub(r"<\|[\d.]+\|>", "", self.server_log_path.read_text(
            encoding="utf-8", errors="replace"))
        log = "".join(log.split())
        leaked = [f for f, r in self.results["items"].items()
                  if len("".join(r["text"].split())) >= 20
                  and "".join(r["text"].split())[:20] in log]
        self.check(writer, model, "server log", "transcripts are not logged", not leaked,
                   f"{len(leaked)} of {len(self.results['items'])} transcripts found"
                   if leaked else "", group="quiet_log")

    # -- chat audio --------------------------------------------------------

    def run_chat(self, writer, items: list[dict], eval_dir: Path) -> None:
        probe = [i for i in items if i["set"] == "en-short"][:self.args.chat_clips]
        for model in self.args.chat_models:
            print(f"\n--- Chat transcription: {model} ---")
            pairs, failed = [], 0
            for item in probe:
                encoded = base64.b64encode((eval_dir / item["file"]).read_bytes()).decode()
                try:
                    response = self.client.chat.completions.create(
                        model=model, temperature=0.0, stream=False,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": CHAT_PROMPT},
                            {"type": "input_audio",
                             "input_audio": {"data": encoded, "format": "mp3"}},
                        ]}],
                        **self.completion_kwargs())
                    text = response.choices[0].message.content or ""
                except Exception as exc:
                    failed += 1
                    text = ""
                    print(f"    ERROR {item['file']}: {exc}")
                self.save_body(model, f"chat_{item['file']}", json.dumps({"text": text}))
                pairs.append(error_rate(item["reference"], text, "en", "wer"))
            rate = pooled(pairs)
            self.results["chat"][model] = {"error_rate": rate, "files": len(probe)}
            print(f"    WER {pct(rate)} over {len(probe)} clips")
            self.check(writer, model, "chat", "every clip transcribed", failed == 0,
                       f"{len(probe) - failed}/{len(probe)}")
            self.check(writer, model, "chat", "WER is under 50%", rate < 0.5, pct(rate))
            base = (self.baseline or {}).get("chat", {}).get(model)
            if base:
                self.check(writer, model, "chat", "WER no worse than the baseline",
                           rate - base["error_rate"] <= self.args.chat_tolerance / 100,
                           f"{pct(rate)} vs {pct(base['error_rate'])}")

    # -- comparison --------------------------------------------------------

    def baseline_overall(self) -> dict | None:
        """The baseline's pooled rate over the same files, so --sets compares like with like."""
        base_items = (self.baseline or {}).get("items", {})
        shared = [base_items[f] for f in self.results["items"] if f in base_items]
        if not shared:
            return None
        return {"error_rate": pooled([(r["edits"], r["length"]) for r in shared]),
                "runaways": sum(r.get("runaway", False) for r in shared)}

    def compare(self, writer, model: str) -> None:
        """Error rates against the baseline build, or the expected values."""
        tolerance = self.args.tolerance / 100
        set_tolerance = self.args.set_tolerance / 100
        overall = self.results["overall"]
        if self.baseline:
            base_items = self.baseline.get("items", {})
            base_overall = self.baseline_overall()
            if base_overall:
                self.check(writer, model, "eval", "pooled error rate no worse than the baseline",
                           overall["error_rate"] - base_overall["error_rate"] <= tolerance,
                           f"{pct(overall['error_rate'])} vs {pct(base_overall['error_rate'])}")
                # A file that crosses the runaway length while making no more
                # errors, for instance by also transcribing what the baseline
                # dropped, is not a regression.
                new_runaways = sorted(
                    f for f, r in self.results["items"].items()
                    if r.get("runaway") and f in base_items
                    and not base_items[f].get("runaway")
                    and r["edits"] > base_items[f]["edits"])
                self.check(writer, model, "eval", "no new runaway transcripts",
                           not new_runaways,
                           ", ".join(new_runaways) if new_runaways else
                           f"{overall['runaways']} runaways, baseline {base_overall['runaways']}")
            for set_name, current in self.results["sets"].items():
                base = self.baseline["sets"].get(set_name)
                if not base:
                    continue
                delta = current["error_rate"] - base["error_rate"]
                if current["files"] >= SET_CHECK_MIN_FILES:
                    self.check(writer, model, f"eval {set_name}",
                               "error rate no worse than the baseline", delta <= set_tolerance,
                               f"{pct(current['error_rate'])} vs {pct(base['error_rate'])}")
                compared = [(f, r) for f, r in self.results["items"].items()
                            if r["set"] == set_name and f in base_items]
                drift = pooled([error_rate(base_items[f]["text"], r["text"], r["language"],
                                           r["metric"]) for f, r in compared])
                current["drift"] = drift
                # Normalized, so whitespace and punctuation alone do not count.
                current["changed"] = sum(
                    normalize(base_items[f]["text"], r["language"])
                    != normalize(r["text"], r["language"]) for f, r in compared)
                if self.args.max_drift is not None:
                    self.check(writer, model, f"eval {set_name}",
                               "transcripts match the baseline's", drift <= self.args.max_drift,
                               f"{pct(drift)} of words differ")
                if base.get("rtf") and current.get("rtf") and current["rtf"] > base["rtf"] * 1.2:
                    print(f"::warning::{set_name}: {current['rtf']:.3f} s per audio second, "
                          f"baseline {base['rtf']:.3f}")
            for name, counts in self.results["repeats"].items():
                base = self.baseline.get("repeats", {}).get(name)
                if base and name in base_items:
                    more_errors = self.results["items"][name]["edits"] > base_items[name]["edits"]
                    self.check(writer, model, f"eval {name}",
                               "no more repeated phrases than the baseline",
                               counts["text"] <= base["text"] or not more_errors,
                               f"{counts['text']} vs {base['text']}")
        elif self.expected:
            limit = self.expected.get("overall")
            if limit is not None:
                self.check(writer, model, "eval", "pooled error rate within the expected value",
                           overall["error_rate"] <= limit + tolerance,
                           f"{pct(overall['error_rate'])} vs expected {pct(limit)}")
            for set_name, current in self.results["sets"].items():
                limit = self.expected.get("sets", {}).get(set_name)
                if limit is not None and current["files"] >= SET_CHECK_MIN_FILES:
                    self.check(writer, model, f"eval {set_name}",
                               "error rate within the expected value",
                               current["error_rate"] <= limit + set_tolerance,
                               f"{pct(current['error_rate'])} vs expected {pct(limit)}")

    # -- report ------------------------------------------------------------

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{self.name}_{self.args.platform}_v{self.version}.md"

    @property
    def results_path(self) -> Path:
        return self.output_dir / f"{self.name}_{self.args.platform}_v{self.version}_results.json"

    def baseline_snapshot(self) -> dict | None:
        """What this run was compared against, in one shape for asr_summary.py."""
        if self.baseline:
            return {
                "kind": "build",
                "version": self.baseline.get("version"),
                "overall": self.baseline_overall(),
                "sets": {name: self.baseline["sets"][name] for name in self.results["sets"]
                         if name in self.baseline.get("sets", {})},
                "chat": self.baseline.get("chat", {}),
            }
        if self.expected:
            return {
                "kind": "expected",
                "overall": {"error_rate": self.expected.get("overall")},
                "sets": {name: {"error_rate": rate}
                         for name, rate in self.expected.get("sets", {}).items()},
            }
        return None

    def finish_results(self) -> None:
        checks = [dict(zip(("model", "area", "group", "check", "status", "detail"), row))
                  for row in self.rows]
        # Failures that are not a check, such as the server crashing, still
        # belong in the report.
        checks += [{"model": "server", "area": "run", "group": "core", "check": failure,
                    "status": "FAIL", "detail": ""}
                   for failure in self.failures if failure not in self.check_failures]
        self.results.update({
            "platform": self.args.platform,
            "version": self.version,
            "required": list(self.args.require),
            "checks": checks,
            "baseline": self.baseline_snapshot(),
        })

    def write_report(self) -> None:
        self.report_path.write_text(
            render([self.results], title=f"ASR test: {platform_label(self.args.platform)}")
            + "\n", encoding="utf-8")

    # -- entry point -------------------------------------------------------

    def run(self) -> None:
        eval_dir = Path(self.args.eval_dir)
        manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
        items = [i for i in manifest["items"]
                 if not self.args.sets or i["set"] in self.args.sets]
        print(f"[asr] {len(items)} files from {eval_dir}, "
              f"required groups: {', '.join(self.args.require) or 'none'}")
        self.results["version"] = self.version

        models = self.resolve_models()
        if not models:
            self.failures.append("no Whisper model to test")
        handle, writer = self.open_csv()
        try:
            for model in self.iter_models(models):
                print(f"\n--- ASR: {model} ---")
                formats_item = next((i for i in items if i["set"] == "en-long"), items[0])
                self.run_formats(writer, model, formats_item, eval_dir / formats_item["file"])
                self.run_eval(writer, model, items, eval_dir)
                self.run_language(writer, model, items, eval_dir)
                self.run_prompt(writer, model, items, eval_dir)
                self.run_quiet_log(writer, model)
                self.compare(writer, model)
            self.run_chat(writer, items, eval_dir)
        finally:
            handle.close()
            self.finish_results()
            self.results_path.write_text(json.dumps(self.results, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
            self.write_report()
        print(f"\nASR test complete. Report: {self.report_path}")
        self.write_summary(models)


if __name__ == "__main__":
    sys.exit(run_task(AsrTest))
